"""Minimal PPO training for Gin Rummy against simple bot.

All on GPU using JAX. Single file for simplicity.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from typing import NamedTuple
import time

# Import from existing implementations
import gin_rummy_core as gin  # Standalone game logic (no pyspiel dependency)

# =============================================================================
# Constants
# =============================================================================
NUM_ACTIONS = 241
OBS_DIM = 63  # 52 (hand) + 8 (phase) + 1 (upcard) + 1 (deck) + 1 (deadwood)
GIN_BONUS = 25
UNDERCUT_BONUS = 25
KNOCK_THRESHOLD = 10


# =============================================================================
# Transition storage
# =============================================================================
class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


# =============================================================================
# Environment Wrapper
# =============================================================================

@jax.jit
def make_observation(game_state, agent_player):
    """Create observation vector from agent's perspective."""
    # Get agent's hand
    hand = jnp.where(agent_player == 0,
                     game_state['player0_hand'],
                     game_state['player1_hand'])

    # Phase one-hot (8 phases)
    phase_onehot = jax.nn.one_hot(game_state['phase'], 8)

    # Upcard normalized (-1 to 51 -> 0 to 1)
    upcard_norm = (game_state['upcard'] + 1) / 53.0

    # Deck count normalized
    deck_count = jnp.sum(game_state['deck']) / 52.0

    # Deadwood normalized (expensive but valuable)
    deadwood = gin.calculate_deadwood_lut(hand) / 100.0

    obs = jnp.concatenate([
        hand.astype(jnp.float32),  # 52
        phase_onehot,               # 8
        jnp.array([upcard_norm]),   # 1
        jnp.array([deck_count]),    # 1
        jnp.array([deadwood]),      # 1
    ])
    return obs


@jax.jit
def is_chance_node(state):
    """Check if current state is a chance node (deal or waiting for stock draw)."""
    in_deal = state['phase'] == gin.PHASE_DEAL
    waiting_stock = state['waiting_stock_draw']
    return in_deal | waiting_stock


@jax.jit
def get_legal_mask(game_state, agent_player):
    """Get legal action mask for agent."""
    # Only valid when it's agent's turn
    is_agent_turn = game_state['current_player'] == agent_player
    mask = gin.legal_actions_mask_fast(game_state)  # Use fast version
    # Return zeros if not agent's turn (shouldn't happen in wrapper)
    return jnp.where(is_agent_turn, mask, jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_))


@jax.jit
def compute_game_score(game_state, agent_player):
    """Compute score from agent's perspective at game end.

    Scoring:
    - Gin: winner gets opponent_deadwood + 25
    - Undercut: defender gets knocker_deadwood - defender_deadwood + 25
    - Normal knock: knocker gets knocker_deadwood - opponent_deadwood
    """
    knocker = game_state['knocker']
    winner = game_state['winner']

    # Get hands
    p0_hand = game_state['player0_hand']
    p1_hand = game_state['player1_hand']

    # Compute deadwood for both players
    p0_dw = gin.calculate_deadwood_lut(p0_hand)
    p1_dw = gin.calculate_deadwood_lut(p1_hand)

    knocker_dw = jnp.where(knocker == 0, p0_dw, p1_dw)
    opponent_dw = jnp.where(knocker == 0, p1_dw, p0_dw)

    is_gin = knocker_dw == 0
    is_undercut = (opponent_dw <= knocker_dw) & ~is_gin

    # Compute score from knocker's perspective
    gin_score = opponent_dw + GIN_BONUS
    undercut_score = -(knocker_dw - opponent_dw + UNDERCUT_BONUS)  # Negative for knocker
    normal_score = knocker_dw - opponent_dw  # Negative is good for knocker

    knocker_score = jnp.where(is_gin, gin_score,
                    jnp.where(is_undercut, undercut_score, -normal_score))

    # Convert to p0's perspective
    p0_score = jnp.where(knocker == 0, knocker_score, -knocker_score)

    # Convert to agent's perspective
    agent_score = jnp.where(agent_player == 0, p0_score, -p0_score)

    return agent_score.astype(jnp.float32)


def handle_chance_and_opponent(game_state, agent_player, key):
    """Handle chance nodes and opponent moves until agent's turn or terminal.

    Uses lax.fori_loop with max iterations for faster JIT compilation.
    """
    MAX_ITERS = 100  # Max steps (deal=21, then alternating turns)

    def should_continue(state):
        is_terminal = state['done']
        is_agent_turn = state['current_player'] == agent_player
        is_chance = is_chance_node(state)
        return ~is_terminal & (~is_agent_turn | is_chance)

    def body_fn(i, carry):
        state, key = carry
        key, subkey1 = jax.random.split(key)

        # Check if we should act
        should_act = should_continue(state)

        is_chance = is_chance_node(state)

        # Chance action: sample from deck
        deck = state['deck'].astype(jnp.float32)
        deck_probs = deck / jnp.maximum(deck.sum(), 1.0)
        chance_action = jax.random.choice(subkey1, 52, p=deck_probs)

        # Opponent action (use simple bot)
        opponent_action = gin.simple_bot_action_opt(state)

        # Select appropriate action
        action = jnp.where(is_chance, chance_action, opponent_action)
        new_state = gin.step(state, action)

        # Only update if we should act
        state = jax.tree.map(
            lambda new, old: jnp.where(should_act, new, old),
            new_state, state
        )

        return (state, key)

    final_state, key = jax.lax.fori_loop(0, MAX_ITERS, body_fn, (game_state, key))
    return final_state, key


@partial(jax.jit, static_argnums=())
def env_init(key):
    """Initialize environment: deal cards, handle opponent if they go first."""
    key, player_key, init_key = jax.random.split(key, 3)

    # Randomly assign agent to player 0 or 1
    agent_player = jax.random.randint(player_key, (), 0, 2, dtype=jnp.int8)

    # Initialize game
    game_state = gin.init_state()

    # Handle dealing and opponent's first move if needed
    game_state, key = handle_chance_and_opponent(game_state, agent_player, init_key)

    # Create observation
    obs = make_observation(game_state, agent_player)
    legal_mask = get_legal_mask(game_state, agent_player)

    return {
        'game_state': game_state,
        'agent_player': agent_player,
        'obs': obs,
        'legal_mask': legal_mask,
        'done': game_state['done'],
        'key': key,
    }


@jax.jit
def check_and_end_game(game_state):
    """Check if game should end (knock phase) and set done flag.

    gin_rummy_core.py step is simplified - it doesn't handle knock/layoff phases.
    We end the game immediately when knock phase is entered.
    """
    in_knock_phase = game_state['phase'] == gin.PHASE_KNOCK
    in_wall_phase = game_state['phase'] == gin.PHASE_WALL

    # End game if in knock phase or wall phase (simplified)
    should_end = in_knock_phase | in_wall_phase

    # Set winner based on current deadwood
    p0_hand = game_state['player0_hand']
    p1_hand = game_state['player1_hand']
    p0_dw = gin.calculate_deadwood_lut(p0_hand)
    p1_dw = gin.calculate_deadwood_lut(p1_hand)

    # Current player is the knocker
    knocker = game_state['current_player']
    winner = jnp.where(p0_dw < p1_dw, jnp.int8(0),
             jnp.where(p1_dw < p0_dw, jnp.int8(1), jnp.int8(-1)))

    new_state = {**game_state}
    new_state['done'] = jnp.where(should_end, jnp.bool_(True), game_state['done'])
    new_state['winner'] = jnp.where(should_end, winner, game_state['winner'])
    new_state['knocker'] = jnp.where(should_end & (game_state['knocker'] < 0),
                                      knocker, game_state['knocker'])
    new_state['phase'] = jnp.where(should_end, jnp.int8(gin.PHASE_GAME_OVER), game_state['phase'])

    return new_state


@jax.jit
def env_step(env_state, action):
    """Take agent action, handle opponent, return next state and reward."""
    game_state = env_state['game_state']
    agent_player = env_state['agent_player']
    key = env_state['key']

    # Apply agent's action
    new_game_state = gin.step(game_state, action)

    # Check if game should end (knock phase)
    new_game_state = check_and_end_game(new_game_state)

    # Handle chance nodes and opponent moves (if not done)
    key, step_key = jax.random.split(key)
    new_game_state, key = handle_chance_and_opponent(
        new_game_state, agent_player, step_key
    )

    # Check again after opponent moves
    new_game_state = check_and_end_game(new_game_state)

    # Compute reward (from agent's perspective, normalized)
    # Only non-zero at terminal
    agent_score = compute_game_score(new_game_state, agent_player)
    reward = jnp.where(
        new_game_state['done'],
        agent_score / 100.0,  # Normalize to ~[-1, 1]
        jnp.float32(0.0)
    )

    # Create observation
    obs = make_observation(new_game_state, agent_player)
    legal_mask = get_legal_mask(new_game_state, agent_player)

    return {
        'game_state': new_game_state,
        'agent_player': agent_player,
        'obs': obs,
        'legal_mask': legal_mask,
        'done': new_game_state['done'],
        'key': key,
    }, reward


@jax.jit
def env_reset_if_done(env_state):
    """Reset environment if done, otherwise keep current state."""
    key = env_state['key']
    key, reset_key = jax.random.split(key)

    fresh_state = env_init(reset_key)

    # Use fresh state if done, otherwise keep current
    return jax.tree.map(
        lambda fresh, current: jnp.where(env_state['done'], fresh, current),
        fresh_state,
        env_state
    )


# =============================================================================
# Actor-Critic Network
# =============================================================================

class ActorCritic(nn.Module):
    """Dense Actor-Critic network for card games."""
    action_dim: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, obs):
        # Shared layers
        x = nn.Dense(256)(obs)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Actor head
        logits = nn.Dense(self.action_dim)(x)

        # Critic head
        value = nn.Dense(1)(x)

        return logits, jnp.squeeze(value, axis=-1)


# =============================================================================
# PPO Functions
# =============================================================================

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute GAE advantages."""
    # Append bootstrap value (0 for simplicity since we reset)
    values_extended = jnp.concatenate([values, jnp.zeros((1,) + values.shape[1:])])

    def gae_step(carry, t):
        gae = carry
        delta = rewards[t] + gamma * values_extended[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        gae_step,
        jnp.zeros(values.shape[1:]),
        jnp.arange(rewards.shape[0] - 1, -1, -1),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


def ppo_loss(params, apply_fn, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """Compute PPO loss."""
    obs, actions, old_log_probs, advantages, returns, legal_masks = batch

    # Forward pass
    logits, values = apply_fn(params, obs)

    # Mask illegal actions
    logits = jnp.where(legal_masks, logits, -1e9)

    # New log probs
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(
        log_probs, actions[:, None], axis=-1
    ).squeeze(-1)

    # Policy loss
    ratio = jnp.exp(action_log_probs - old_log_probs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    value_loss = 0.5 * ((values - returns) ** 2).mean()

    # Entropy bonus
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * jnp.where(probs > 0, jnp.log(probs + 1e-8), 0), axis=-1).mean()

    total_loss = pg_loss + vf_coef * value_loss - ent_coef * entropy

    return total_loss, {'pg_loss': pg_loss, 'value_loss': value_loss, 'entropy': entropy}


# =============================================================================
# Training
# =============================================================================

def train(
    num_envs=4096,
    num_steps=128,
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    num_minibatches=8,
    update_epochs=4,
    max_grad_norm=0.5,
    seed=42,
):
    """Main training function."""
    print(f"Training PPO on Gin Rummy", flush=True)
    print(f"  num_envs={num_envs}, num_steps={num_steps}", flush=True)
    print(f"  total_timesteps={total_timesteps}", flush=True)
    print(f"  Devices: {jax.devices()}", flush=True)

    key = jax.random.PRNGKey(seed)

    # Initialize network
    network = ActorCritic()
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, OBS_DIM))
    params = network.init(init_key, dummy_obs)

    # Optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5)
    )
    opt_state = tx.init(params)

    # Initialize environments
    print("Initializing environments...", flush=True)
    key, *env_keys = jax.random.split(key, num_envs + 1)
    env_keys = jnp.stack(env_keys)
    env_states = jax.vmap(env_init)(env_keys)
    print(f"  Environments initialized. Obs shape: {env_states['obs'].shape}", flush=True)

    # JIT compile core functions
    apply_fn = jax.jit(network.apply)

    @jax.jit
    def get_action(params, obs, legal_mask, key):
        """Sample action from policy."""
        logits, value = apply_fn(params, obs)
        # Mask illegal actions
        logits = jnp.where(legal_mask, logits, -1e9)
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]
        return action, value, log_prob

    @jax.jit
    def collect_rollout(params, env_states, key):
        """Collect num_steps of experience."""
        def step_fn(carry, _):
            env_states, key = carry
            key, *action_keys = jax.random.split(key, num_envs + 1)
            action_keys = jnp.stack(action_keys)

            # Get observations and masks
            obs = env_states['obs']
            legal_masks = env_states['legal_mask']

            # Get actions
            actions, values, log_probs = jax.vmap(
                lambda o, m, k: get_action(params, o, m, k)
            )(obs, legal_masks, action_keys)

            # Step environments
            env_states, rewards = jax.vmap(env_step)(env_states, actions)
            dones = env_states['done']

            # Reset done environments
            env_states = jax.vmap(env_reset_if_done)(env_states)

            transition = Transition(
                obs=obs,
                action=actions,
                reward=rewards,
                done=dones,
                value=values,
                log_prob=log_probs,
            )

            return (env_states, key), transition

        (env_states, key), trajectory = jax.lax.scan(
            step_fn, (env_states, key), None, num_steps
        )

        return env_states, trajectory, key

    @jax.jit
    def update_step(params, opt_state, trajectory):
        """PPO update step."""
        # Compute advantages
        advantages, returns = compute_gae(
            trajectory.reward, trajectory.value, trajectory.done,
            gamma, gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten batch
        batch_size = num_steps * num_envs
        minibatch_size = batch_size // num_minibatches

        obs = trajectory.obs.reshape(batch_size, -1)
        actions = trajectory.action.reshape(batch_size)
        old_log_probs = trajectory.log_prob.reshape(batch_size)
        advs = advantages.reshape(batch_size)
        rets = returns.reshape(batch_size)

        # Get legal masks (need to store them in trajectory for this)
        # For now, use all-ones mask (legal actions already enforced)
        legal_masks = jnp.ones((batch_size, NUM_ACTIONS), dtype=jnp.bool_)

        def epoch_step(carry, _):
            params, opt_state, key = carry
            key, perm_key = jax.random.split(key)

            # Shuffle
            perm = jax.random.permutation(perm_key, batch_size)

            def minibatch_step(carry, start_idx):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))

                batch = (
                    obs[idx],
                    actions[idx],
                    old_log_probs[idx],
                    advs[idx],
                    rets[idx],
                    legal_masks[idx],
                )

                (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                    params, apply_fn, batch, clip_eps, vf_coef, ent_coef
                )

                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

                return (params, opt_state), (loss, aux)

            starts = jnp.arange(0, batch_size, minibatch_size)
            (params, opt_state), (losses, auxs) = jax.lax.scan(
                minibatch_step, (params, opt_state), starts
            )

            return (params, opt_state, key), (losses.mean(), {k: v.mean() for k, v in auxs.items()})

        key = jax.random.PRNGKey(0)  # Deterministic for reproducibility
        (params, opt_state, _), (losses, auxs) = jax.lax.scan(
            epoch_step, (params, opt_state, key), None, update_epochs
        )

        return params, opt_state, losses.mean(), {k: v.mean() for k, v in auxs.items()}

    # Training loop
    num_updates = total_timesteps // (num_envs * num_steps)
    print(f"\nStarting training: {num_updates} updates", flush=True)

    total_games = 0
    total_wins = 0
    start_time = time.time()

    for update in range(num_updates):
        # Collect rollout
        key, rollout_key = jax.random.split(key)
        env_states, trajectory, _ = collect_rollout(params, env_states, rollout_key)

        # Count completed games
        games_this_update = trajectory.done.sum()
        wins_this_update = (trajectory.reward > 0).sum()
        total_games += int(games_this_update)
        total_wins += int(wins_this_update)

        # Update
        params, opt_state, loss, aux = update_step(params, opt_state, trajectory)

        # Logging
        if update % 10 == 0:
            elapsed = time.time() - start_time
            steps_done = (update + 1) * num_envs * num_steps
            steps_per_sec = steps_done / elapsed
            win_rate = total_wins / max(total_games, 1)

            print(f"Update {update}/{num_updates} | "
                  f"Steps: {steps_done:,} | "
                  f"Steps/sec: {steps_per_sec:,.0f} | "
                  f"Loss: {float(loss):.4f} | "
                  f"Win rate: {win_rate:.2%} ({total_wins}/{total_games})", flush=True)

    # Final stats
    elapsed = time.time() - start_time
    total_steps = num_updates * num_envs * num_steps
    print(f"\nTraining complete!", flush=True)
    print(f"  Total time: {elapsed:.1f}s", flush=True)
    print(f"  Total steps: {total_steps:,}", flush=True)
    print(f"  Steps/sec: {total_steps / elapsed:,.0f}", flush=True)
    print(f"  Final win rate: {total_wins / max(total_games, 1):.2%}", flush=True)

    return params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = train(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        seed=args.seed,
    )

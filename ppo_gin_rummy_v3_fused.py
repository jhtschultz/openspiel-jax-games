"""PPO Training for Gin Rummy (V3 Fused) - Unrolled Environment Loop.

This version replaces the `handle_chance_and_opponent` loop with a fixed,
unrolled sequence of steps. This allows XLA to fuse the entire environment
dynamics into a single kernel, potentially increasing FPS.

Unrolled Sequence:
1. Agent acts (Draw/Discard).
2. Bot acts (responds to Agent).
3. Bot acts (next turn Draw).
4. Bot acts (next turn Discard).
... until next Agent turn.

Since we automated the Agent's endgame (Knock/Layoff), we can guarantee
a strict turn structure for the "Strategic" phases (Draw/Discard).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from typing import NamedTuple
import time

# Use bfloat16 for speed
DTYPE = jnp.bfloat16

# Use gin_rummy_jax for full game logic
import gin_rummy_jax as gin

# =============================================================================
# Constants
# =============================================================================
NUM_ACTIONS = 241
OBS_MODE = "standard"  # Keep standard for now to isolate speedup
OBS_DIM = 167

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
    legal_mask: jnp.ndarray


# =============================================================================
# Logic for Known Cards
# =============================================================================
@jax.jit
def update_known_cards(state, action, known_cards):
    """Update belief state about which cards each player holds."""
    player = state['current_player']
    phase = state['phase']
    
    is_draw_upcard = (((phase == gin.PHASE_FIRST_UPCARD) | (phase == gin.PHASE_DRAW)) &
                      (action == gin.ACTION_DRAW_UPCARD))
    is_discard = (phase == gin.PHASE_DISCARD) & (action < 52)
    
    def do_update(k):
        # Handle Draw Upcard
        card_drawn = state['upcard'] 
        k = jax.lax.cond(
            is_draw_upcard,
            lambda x: x.at[player, card_drawn].set(True),
            lambda x: x,
            k
        )
        # Handle Discard
        card_discarded = action
        k = jax.lax.cond(
            is_discard,
            lambda x: x.at[player, card_discarded].set(False),
            lambda x: x,
            k
        )
        return k

    should_update = is_draw_upcard | is_discard
    return jax.lax.cond(should_update, do_update, lambda x: x, known_cards)


# =============================================================================
# Environment Wrapper
# =============================================================================

@jax.jit
def make_observation(game_state, known_cards, agent_player):
    """Standard 167-dim observation."""
    hand = jnp.where(
        agent_player == 0,
        game_state['player0_hand'],
        game_state['player1_hand'])
    discard_pile = game_state['discard_pile']
    opponent = 1 - agent_player
    opp_known = known_cards[opponent]
    phase_onehot = jax.nn.one_hot(game_state['phase'], 8)
    upcard_norm = (game_state['upcard'] + 1) / 53.0
    deck_count = jnp.sum(game_state['deck']) / 52.0
    deadwood = gin.calculate_deadwood_lut(hand) / 100.0

    return jnp.concatenate([
        hand.astype(DTYPE),               # 52
        discard_pile.astype(DTYPE),       # 52
        opp_known.astype(DTYPE),          # 52
        phase_onehot.astype(DTYPE),       # 8
        jnp.array([upcard_norm], dtype=DTYPE),   # 1
        jnp.array([deck_count], dtype=DTYPE),    # 1
        jnp.array([deadwood], dtype=DTYPE),      # 1
    ])


@jax.jit
def get_legal_mask(game_state, agent_player):
    """Get legal action mask for agent."""
    is_agent_turn = game_state['current_player'] == agent_player
    mask = gin.legal_actions_mask(game_state) 
    return jnp.where(is_agent_turn, mask, jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_))


@jax.jit
def compute_game_score(game_state, agent_player):
    """Get score from agent's perspective at game end."""
    p0_score = game_state['p0_score']
    agent_score = jnp.where(agent_player == 0, p0_score, -p0_score)
    return agent_score.astype(jnp.float32)


# =============================================================================
# Fused Step Logic (Unrolled)
# =============================================================================

@jax.jit
def is_chance_node(state):
    in_deal = state['phase'] == gin.PHASE_DEAL
    waiting_stock = state['waiting_stock_draw']
    return in_deal | waiting_stock

@jax.jit
def is_agent_strategic_turn(state, agent_player):
    """Check if it's agent's turn AND a strategic phase (not endgame)."""
    is_agent_turn = state['current_player'] == agent_player
    phase = state['phase']
    # Agent makes decisions for these phases only
    is_strategic_phase = (
        (phase == gin.PHASE_FIRST_UPCARD) |
        (phase == gin.PHASE_DRAW) |
        (phase == gin.PHASE_DISCARD)
    )
    return is_agent_turn & is_strategic_phase

@jax.jit
def single_auto_step(state, known_cards, agent_player, key):
    """Execute ONE step of auto-play (Chance or Bot).
    
    Returns: (new_state, new_known_cards, key, done_masked)
    """
    key, subkey = jax.random.split(key)
    
    is_chance = is_chance_node(state)
    
    # Chance action
    deck = state['deck'].astype(jnp.float32)
    deck_probs = deck / jnp.maximum(deck.sum(), 1.0)
    chance_action = jax.random.choice(subkey, 52, p=deck_probs)
    
    # Bot action
    bot_action = gin.simple_bot_action_opt(state)
    
    # Select action
    action = jnp.where(is_chance, chance_action, bot_action)
    
    # Update known cards (only for opponent actions)
    is_opponent = state['current_player'] != agent_player
    should_update = ~state['done'] & ~is_chance & is_opponent
    known_cards = jax.lax.cond(
        should_update,
        lambda k: update_known_cards(state, action, k),
        lambda k: k,
        known_cards
    )
    
    # Step env (only if not done)
    # If done, step returns state as-is (mostly), but we guard it anyway
    # to prevent weird side effects in the logic
    # Actually gin.step doesn't check 'done', so we must not call it if done
    new_state = gin.step(state, action)
    
    # Mask update if already done
    state = jax.tree.map(
        lambda new, old: jnp.where(state['done'], old, new),
        new_state, state
    )
    
    return state, known_cards, key

@jax.jit
def env_step_fused(env_state, action):
    """Execute Agent Step + Sequence of Auto-Steps until next Agent Turn.
    
    This unrolls the loop to a fixed depth (e.g. 20) which covers >99% of 
    inter-turn sequences. If the sequence is longer, it pauses and waits 
    for the next 'env_step' call (which is fine, the agent just sees the 
    intermediate state, realizes it's not its turn, and calls this again).
    
    Actually, for PPO, we MUST return an observation where it IS the agent's turn.
    So we must unroll enough to reach the agent's turn or game end.
    
    Typical sequence:
    1. Agent Discards.
    2. Opponent Draws (Chance/Strategic).
    3. Opponent Discards.
    4. Agent Draws (Strategic) -> RETURN.
    
    Max depth needed is small (~5) except for DEAL phase (21).
    We use a fixed unroll of 25 to be safe.
    """
    game_state = env_state['game_state']
    known_cards = env_state['known_cards']
    agent_player = env_state['agent_player']
    key = env_state['key']

    # --- 1. Agent's Action (The PPO Action) ---
    known_cards = update_known_cards(game_state, action, known_cards)
    game_state = gin.step(game_state, action)
    
    # --- 2. Unrolled Auto-Play Sequence ---
    # We unroll 25 steps. This covers the Deal (21) + buffer.
    # Most turns use only 2-3 steps.
    
    def loop_body(carry, _):
        state, k_cards, k = carry
        
        # Check stop condition: Game Done OR Agent Strategic Turn
        should_stop = state['done'] | is_agent_strategic_turn(state, agent_player)
        
        # If stopped, identity update. Else, step.
        # We compute the step anyway (for fusion) and mask the result.
        next_state, next_k, next_k_key = single_auto_step(state, k_cards, agent_player, k)
        
        state = jax.tree.map(
            lambda new, old: jnp.where(should_stop, old, new),
            next_state, state
        )
        k_cards = jax.tree.map(
            lambda new, old: jnp.where(should_stop, old, new),
            next_k, k_cards
        )
        # Always update key to maintain randomness sync
        k = next_k_key
        
        return (state, k_cards, k), None

    # Unroll 25 times
    (game_state, known_cards, key), _ = jax.lax.scan(
        loop_body, (game_state, known_cards, key), None, length=25
    )

    # --- 3. Finalize ---
    agent_score = compute_game_score(game_state, agent_player)
    reward = jnp.where(game_state['done'], agent_score / 100.0, jnp.float32(0.0))

    obs = make_observation(game_state, known_cards, agent_player)
    legal_mask = get_legal_mask(game_state, agent_player)

    return {
        'game_state': game_state,
        'known_cards': known_cards,
        'agent_player': agent_player,
        'obs': obs,
        'legal_mask': legal_mask,
        'done': game_state['done'],
        'key': key,
    }, reward


@partial(jax.jit, static_argnums=())
def env_init(key):
    """Initialize environment and fast-forward to first agent turn."""
    key, player_key, init_key = jax.random.split(key, 3)
    agent_player = jax.random.randint(player_key, (), 0, 2, dtype=jnp.int8)
    
    game_state = gin.init_state()
    known_cards = jnp.zeros((2, 52), dtype=jnp.bool_)
    
    # Fast forward (Deal + Opponent) using the SAME unrolled logic
    # We create a dummy "init" state and call the scan loop
    # Ideally we'd reuse the code, but for now we copy the scan pattern 
    # slightly modified (start from scratch)
    
    def loop_body(carry, _):
        state, k_cards, k = carry
        should_stop = state['done'] | is_agent_strategic_turn(state, agent_player)
        next_state, next_known, next_key = single_auto_step(state, k_cards, agent_player, k)

        state = jax.tree.map(lambda n, o: jnp.where(should_stop, o, n), next_state, state)
        k_cards = jax.tree.map(lambda n, o: jnp.where(should_stop, o, n), next_known, k_cards)
        return (state, k_cards, next_key), None

    # Length 30 covers Deal (21) + Initial play
    (game_state, known_cards, key), _ = jax.lax.scan(
        loop_body, (game_state, known_cards, init_key), None, length=30
    )

    obs = make_observation(game_state, known_cards, agent_player)
    legal_mask = get_legal_mask(game_state, agent_player)

    return {
        'game_state': game_state,
        'known_cards': known_cards,
        'agent_player': agent_player,
        'obs': obs,
        'legal_mask': legal_mask,
        'done': game_state['done'],
        'key': key,
    }


@jax.jit
def env_reset_if_done(env_state):
    key = env_state['key']
    key, reset_key = jax.random.split(key)
    fresh_state = env_init(reset_key)
    return jax.tree.map(
        lambda fresh, current: jnp.where(env_state['done'], fresh, current),
        fresh_state, env_state
    )


# =============================================================================
# Network (Standard CNN + bf16)
# =============================================================================

class ActorCriticCNN(nn.Module):
    action_dim: int = NUM_ACTIONS
    dtype: jnp.dtype = DTYPE

    @nn.compact
    def __call__(self, obs):
        is_batched = obs.ndim == 2
        if not is_batched: obs = obs[None, :]
        batch_size = obs.shape[0]
        obs = obs.astype(self.dtype)

        cards_flat = obs[:, :156]
        other = obs[:, 156:]
        cards = cards_flat.reshape(batch_size, 3, 4, 13)
        x = jnp.transpose(cards, (0, 2, 3, 1))

        x = nn.Conv(32, (1, 3), padding='SAME', dtype=self.dtype)(x); x = nn.relu(x)
        x = nn.Conv(32, (3, 1), padding='SAME', dtype=self.dtype)(x); x = nn.relu(x)
        x = nn.Conv(64, (2, 2), padding='SAME', dtype=self.dtype)(x); x = nn.relu(x)
        x = x.reshape(batch_size, -1)
        
        card_feat = nn.Dense(128, dtype=self.dtype)(x); card_feat = nn.relu(card_feat)
        other_feat = nn.Dense(32, dtype=self.dtype)(other); other_feat = nn.relu(other_feat)
        combined = jnp.concatenate([card_feat, other_feat], axis=-1)

        actor_h = nn.Dense(128, dtype=self.dtype)(combined); actor_h = nn.relu(actor_h)
        logits = nn.Dense(self.action_dim, dtype=self.dtype)(actor_h)

        critic_h = nn.Dense(128, dtype=self.dtype)(combined); critic_h = nn.relu(critic_h)
        value = nn.Dense(1, dtype=self.dtype)(critic_h)

        value = jnp.squeeze(value, axis=-1)
        if not is_batched: logits = logits[0]; value = value[0]
        return logits, value

ActorCritic = ActorCriticCNN


# =============================================================================
# PPO Functions
# =============================================================================

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    values_extended = jnp.concatenate([values, jnp.zeros((1,) + values.shape[1:])])
    def gae_step(carry, t):
        gae = carry
        delta = rewards[t] + gamma * values_extended[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return gae, gae
    _, advantages = jax.lax.scan(gae_step, jnp.zeros(values.shape[1:]), jnp.arange(rewards.shape[0]-1, -1, -1))
    return advantages[::-1], advantages[::-1] + values

def ppo_loss(params, apply_fn, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    obs, actions, old_log_probs, advantages, returns, legal_masks, old_values = batch
    logits, values = apply_fn(params, obs)
    logits = jnp.where(legal_masks, logits, -1e9)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)

    ratio = jnp.exp(action_log_probs - old_log_probs)
    pg_loss = -jnp.minimum(advantages * ratio, advantages * jnp.clip(ratio, 1-clip_eps, 1+clip_eps)).mean()
    
    v_loss_unclipped = (values - returns) ** 2
    v_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    v_loss_clipped = (v_clipped - returns) ** 2
    value_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * jnp.where(legal_masks, log_probs, 0.0), axis=-1).mean()
    
    return pg_loss + vf_coef * value_loss - ent_coef * entropy, {'pg_loss': pg_loss, 'value_loss': value_loss, 'entropy': entropy}


# =============================================================================
# Training
# =============================================================================

def train(
    num_envs=4096,
    num_steps=128,
    total_timesteps=50_000_000,  # More steps to see steady-state
    learning_rate=3e-4,
    seed=42,
):
    print(f"Training PPO (V3 Fused) on Gin Rummy", flush=True)
    print(f"  Unrolled Environment Loop (25 steps)", flush=True)
    
    key = jax.random.PRNGKey(seed)
    network = ActorCritic()
    key, init_key = jax.random.split(key)
    params = network.init(init_key, jnp.zeros((1, OBS_DIM), dtype=DTYPE))
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate, eps=1e-5))
    opt_state = tx.init(params)
    apply_fn = jax.jit(network.apply)

    # Init envs
    print("Initializing...", flush=True)
    key, *keys = jax.random.split(key, num_envs + 1)
    env_states = jax.vmap(env_init)(jnp.stack(keys))

    @jax.jit
    def collect_rollout(params, env_states, key):
        def step_fn(carry, _):
            env_states, key = carry
            key, *keys = jax.random.split(key, num_envs + 1)
            keys = jnp.stack(keys)
            obs, masks = env_states['obs'], env_states['legal_mask']

            logits, value = apply_fn(params, obs)
            logits = jnp.where(masks, logits, -1e9)
            # Use vmap for batched categorical sampling
            action = jax.vmap(lambda k, l: jax.random.categorical(k, l))(keys, logits)
            log_prob = jax.nn.log_softmax(logits)
            action_log_prob = jnp.take_along_axis(log_prob, action[:, None], axis=-1).squeeze(-1)

            # Fused Step
            next_env_states, reward = jax.vmap(env_step_fused)(env_states, action)
            dones = next_env_states['done']
            next_env_states = jax.vmap(env_reset_if_done)(next_env_states)

            return (next_env_states, key), Transition(obs, action, reward, dones, value, action_log_prob, masks)

        (env_states, key), traj = jax.lax.scan(step_fn, (env_states, key), None, num_steps)
        return env_states, traj, key

    @jax.jit
    def update(params, opt_state, traj, key):
        adv, ret = compute_gae(traj.reward, traj.value, traj.done)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # Flatten
        batch_size = num_steps * num_envs
        batch = (
            traj.obs.reshape(batch_size, -1),
            traj.action.reshape(batch_size),
            traj.log_prob.reshape(batch_size),
            adv.reshape(batch_size),
            ret.reshape(batch_size),
            traj.legal_mask.reshape(batch_size, -1),
            traj.value.reshape(batch_size)
        )
        
        # 4 Epochs, 8 Minibatches
        def epoch_fn(carry, _):
            params, opt_state, key = carry
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, batch_size)
            
            def mb_fn(carry, idx):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (idx,), (batch_size//8,))
                mb = tuple(x[idx] for x in batch)
                (l, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, apply_fn, mb)
                updates, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), (l, aux)
                
            (params, opt_state), (l, aux) = jax.lax.scan(mb_fn, (params, opt_state), jnp.arange(0, batch_size, batch_size//8))
            return (params, opt_state, key), (l.mean(), {k: v.mean() for k,v in aux.items()})

        (params, opt_state, _), (l, aux) = jax.lax.scan(epoch_fn, (params, opt_state, key), None, 4)
        return params, opt_state, l.mean(), {k: v.mean() for k,v in aux.items()}

    # Train loop
    num_updates = total_timesteps // (num_envs * num_steps)
    start = time.time()
    for i in range(num_updates):
        key, r_key, u_key = jax.random.split(key, 3)
        env_states, traj, _ = collect_rollout(params, env_states, r_key)
        params, opt_state, loss, aux = update(params, opt_state, traj, u_key)
        
        if i % 1 == 0:  # Print every update
            elapsed = time.time() - start
            steps = (i+1) * num_envs * num_steps
            games = max(int(traj.done.sum()), 1)
            wins = int((traj.reward > 0).sum())
            print(f"Update {i} | Steps: {steps/1e6:.1f}M | FPS: {steps/elapsed:,.0f} | Win: {wins/games:.1%} | Loss: {loss:.4f}", flush=True)

if __name__ == "__main__":
    train()

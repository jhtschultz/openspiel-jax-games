"""PPO Training for Gin Rummy (V2) - Enhanced Observation & Full Game Play.

Improvements:
1. Enhanced Observation Space:
   - Added 52-dim vector for Discard Pile.
   - Added 52-dim vector for Opponent's Known Cards.
2. Full Game Play:
   - Removed early game termination hack.
   - Agent plays through Knock and Layoff phases.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from typing import NamedTuple
import time

# Use bfloat16 for faster computation on A100
DTYPE = jnp.bfloat16

# Use gin_rummy_jax for training (has discard_pile, full game logic)
import gin_rummy_jax as gin

# =============================================================================
# Constants
# =============================================================================
NUM_ACTIONS = 241

# Observation mode: "standard" (167 dims) or "alphazero" (728 dims, 14 planes)
OBS_MODE = "standard"  # Toggle this for experiments

# Oracle mode: if True, agent sees opponent's actual hand instead of known cards
ORACLE_MODE = False  # Set via --oracle flag

if OBS_MODE == "alphazero":
    # AlphaZero-style observation: 14 planes of 4x13
    # Plane 0: Hand (binary)
    # Plane 1: Discard Pile (binary)
    # Plane 2: Opponent Known Cards (binary)
    # Planes 3-10: Phase one-hot (8 planes)
    # Plane 11: Upcard (one-hot position)
    # Plane 12: Deadwood (fractional fill)
    # Plane 13: Deck Count (fractional fill)
    NUM_PLANES = 14
    OBS_DIM = NUM_PLANES * 4 * 13  # 728
else:
    # Standard observation: 167 dims
    # 52 (hand) + 52 (discard) + 52 (known) + 8 (phase) + 1 (upcard) + 1 (deck) + 1 (deadwood)
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
    """Update belief state about which cards each player holds.
    
    known_cards: (2, 52) bool array.
    """
    player = state['current_player']
    phase = state['phase']
    
    # Action masks
    is_draw_upcard = (((phase == gin.PHASE_FIRST_UPCARD) | (phase == gin.PHASE_DRAW)) &
                      (action == gin.ACTION_DRAW_UPCARD))
    is_discard = (phase == gin.PHASE_DISCARD) & (action < 52)
    
    # Logic:
    # 1. If player draws upcard, we know they have that card.
    # 2. If player discards a card, we know they no longer have it (in hand).
    
    # Helper for conditional update
    def do_update(k):
        # Handle Draw Upcard
        card_drawn = state['upcard'] # Valid card index if is_draw_upcard
        # Only update if is_draw_upcard is True
        k = jax.lax.cond(
            is_draw_upcard,
            lambda x: x.at[player, card_drawn].set(True),
            lambda x: x,
            k
        )
        
        # Handle Discard
        card_discarded = action # Valid if is_discard
        k = jax.lax.cond(
            is_discard,
            lambda x: x.at[player, card_discarded].set(False),
            lambda x: x,
            k
        )
        return k

    # Skip update for chance nodes or other phases (optimization)
    should_update = is_draw_upcard | is_discard
    return jax.lax.cond(should_update, do_update, lambda x: x, known_cards)


# =============================================================================
# Environment Wrapper
# =============================================================================

def _make_observation_standard(game_state, known_cards, agent_player, oracle=False):
    """Standard observation: 167 dims.

    If oracle=True, uses opponent's actual hand instead of known cards.
    """
    hand = jnp.where(
        agent_player == 0,
        game_state['player0_hand'],
        game_state['player1_hand'])
    discard_pile = game_state['discard_pile']

    # Oracle mode: see opponent's actual hand; otherwise just known cards
    opponent = 1 - agent_player
    if oracle:
        opp_info = jnp.where(
            agent_player == 0,
            game_state['player1_hand'],
            game_state['player0_hand'])
    else:
        opp_info = known_cards[opponent]

    phase_onehot = jax.nn.one_hot(game_state['phase'], 8)
    upcard_norm = (game_state['upcard'] + 1) / 53.0
    deck_count = jnp.sum(game_state['deck']) / 52.0
    deadwood = gin.calculate_deadwood_lut(hand) / 100.0

    return jnp.concatenate([
        hand.astype(DTYPE),               # 52
        discard_pile.astype(DTYPE),       # 52
        opp_info.astype(DTYPE),           # 52: opponent's hand (oracle) or known cards
        phase_onehot.astype(DTYPE),       # 8
        jnp.array([upcard_norm], dtype=DTYPE),   # 1
        jnp.array([deck_count], dtype=DTYPE),    # 1
        jnp.array([deadwood], dtype=DTYPE),      # 1
    ])  # 167


def _make_observation_alphazero(game_state, known_cards, agent_player):
    """AlphaZero-style observation: 14 planes of 4x13 = 728 dims."""
    hand = jnp.where(
        agent_player == 0,
        game_state['player0_hand'],
        game_state['player1_hand'])
    discard_pile = game_state['discard_pile']
    opponent = 1 - agent_player
    opp_known = known_cards[opponent]
    phase = game_state['phase']
    upcard = game_state['upcard']
    deck_count = jnp.sum(game_state['deck']) / 52.0
    deadwood = gin.calculate_deadwood_lut(hand) / 100.0

    # Reshape card vectors to 4x13 planes
    hand_plane = hand.reshape(4, 13).astype(DTYPE)
    discard_plane = discard_pile.reshape(4, 13).astype(DTYPE)
    opp_known_plane = opp_known.reshape(4, 13).astype(DTYPE)

    # Phase: 8 one-hot planes
    phase_planes = jnp.zeros((8, 4, 13), dtype=DTYPE)
    phase_planes = phase_planes.at[phase].set(1.0)

    # Upcard: one-hot plane
    upcard_plane = jnp.zeros((4, 13), dtype=DTYPE)
    upcard_valid = upcard >= 0
    upcard_suit = upcard // 13
    upcard_rank = upcard % 13
    upcard_plane = jnp.where(
        upcard_valid,
        upcard_plane.at[upcard_suit, upcard_rank].set(1.0),
        upcard_plane
    )

    # Fractional fill planes
    deadwood_plane = jnp.full((4, 13), deadwood, dtype=DTYPE)
    deck_plane = jnp.full((4, 13), deck_count, dtype=DTYPE)

    obs = jnp.concatenate([
        hand_plane[None], discard_plane[None], opp_known_plane[None],
        phase_planes, upcard_plane[None], deadwood_plane[None], deck_plane[None],
    ], axis=0)
    return obs.reshape(-1)  # 728


# Select observation function based on mode
# Note: ORACLE_MODE is checked at module load time, so restart to change
if OBS_MODE == "alphazero":
    @jax.jit
    def make_observation(game_state, known_cards, agent_player):
        return _make_observation_alphazero(game_state, known_cards, agent_player)
else:
    @jax.jit
    def make_observation(game_state, known_cards, agent_player):
        return _make_observation_standard(game_state, known_cards, agent_player, oracle=ORACLE_MODE)


@jax.jit
def is_chance_node(state):
    """Check if current state is a chance node (deal or waiting for stock draw)."""
    in_deal = state['phase'] == gin.PHASE_DEAL
    waiting_stock = state['waiting_stock_draw']
    return in_deal | waiting_stock


@jax.jit
def get_legal_mask(game_state, agent_player):
    """Get legal action mask for agent."""
    is_agent_turn = game_state['current_player'] == agent_player
    # Use standard legal_actions_mask (includes correct deadwood checks)
    mask = gin.legal_actions_mask(game_state) 
    return jnp.where(is_agent_turn, mask, jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_))


@jax.jit
def compute_game_score(game_state, agent_player):
    """Get score from agent's perspective at game end."""
    p0_score = game_state['p0_score']
    agent_score = jnp.where(agent_player == 0, p0_score, -p0_score)
    return agent_score.astype(jnp.float32)


def handle_chance_and_opponent(game_state, known_cards, agent_player, key):
    """Handle chance nodes, opponent moves, AND agent's endgame phases.

    Agent only makes decisions for FIRST_UPCARD, DRAW, and DISCARD phases.
    KNOCK, LAYOFF, and WALL phases use optimal bot strategy automatically.

    Tracks known cards during opponent's moves.
    """
    MAX_ITERS = 100

    def is_agent_strategic_turn(state):
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

    def should_continue(state):
        """Continue if not terminal and not agent's strategic decision point."""
        is_terminal = state['done']
        is_chance = is_chance_node(state)
        is_strategic = is_agent_strategic_turn(state)
        # Continue if: not done AND (chance node OR not agent's strategic turn)
        return ~is_terminal & (is_chance | ~is_strategic)

    def body_fn(i, carry):
        state, k_cards, key = carry
        key, subkey1 = jax.random.split(key)

        should_act = should_continue(state)
        is_chance = is_chance_node(state)

        # Chance action: sample from deck
        deck = state['deck'].astype(jnp.float32)
        deck_probs = deck / jnp.maximum(deck.sum(), 1.0)
        chance_action = jax.random.choice(subkey1, 52, p=deck_probs)

        # Non-chance action: use optimal bot (for opponent OR agent's endgame)
        bot_action = gin.simple_bot_action_opt(state)

        # Select action
        action = jnp.where(is_chance, chance_action, bot_action)

        # Update known cards (only matters for opponent's observable actions)
        is_opponent = state['current_player'] != agent_player
        should_update_known = should_act & ~is_chance & is_opponent
        updated_k_cards = update_known_cards(state, action, k_cards)

        # Step env
        new_state = gin.step(state, action)

        # Conditionally update state and known cards
        state = jax.tree.map(
            lambda new, old: jnp.where(should_act, new, old),
            new_state, state
        )
        k_cards = jax.tree.map(
            lambda new, old: jnp.where(should_update_known, new, old),
            updated_k_cards, k_cards
        )

        return (state, k_cards, key)

    final_state, final_k_cards, key = jax.lax.fori_loop(
        0, MAX_ITERS, body_fn, (game_state, known_cards, key)
    )
    return final_state, final_k_cards, key


@partial(jax.jit, static_argnums=())
def env_init(key):
    """Initialize environment."""
    key, player_key, init_key = jax.random.split(key, 3)

    agent_player = jax.random.randint(player_key, (), 0, 2, dtype=jnp.int8)

    game_state = gin.init_state()
    # No manual p0_score needed, gin_rummy_core handles it
    
    # Initialize known cards (2 players x 52 cards)
    known_cards = jnp.zeros((2, 52), dtype=jnp.bool_)

    # Handle dealing and opponent's first move
    game_state, known_cards, key = handle_chance_and_opponent(
        game_state, known_cards, agent_player, init_key
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
def env_step(env_state, action):
    """Take agent action, handle opponent, return next state and reward."""
    game_state = env_state['game_state']
    known_cards = env_state['known_cards']
    agent_player = env_state['agent_player']
    key = env_state['key']

    # 1. Update known cards based on Agent's action
    known_cards = update_known_cards(game_state, action, known_cards)

    # 2. Apply agent's action
    new_game_state = gin.step(game_state, action)

    # 3. Handle chance nodes and opponent moves (until next agent turn)
    key, step_key = jax.random.split(key)
    new_game_state, known_cards, key = handle_chance_and_opponent(
        new_game_state, known_cards, agent_player, step_key
    )

    # 4. Compute reward (only at end of game)
    agent_score = compute_game_score(new_game_state, agent_player)
    reward = jnp.where(
        new_game_state['done'],
        agent_score / 100.0,
        jnp.float32(0.0)
    )

    obs = make_observation(new_game_state, known_cards, agent_player)
    legal_mask = get_legal_mask(new_game_state, agent_player)

    return {
        'game_state': new_game_state,
        'known_cards': known_cards,
        'agent_player': agent_player,
        'obs': obs,
        'legal_mask': legal_mask,
        'done': new_game_state['done'],
        'key': key,
    }, reward


@jax.jit
def env_reset_if_done(env_state):
    """Reset environment if done."""
    key = env_state['key']
    key, reset_key = jax.random.split(key)

    fresh_state = env_init(reset_key)

    return jax.tree.map(
        lambda fresh, current: jnp.where(env_state['done'], fresh, current),
        fresh_state,
        env_state
    )


# =============================================================================
# Actor-Critic Networks
# =============================================================================

class ActorCriticCNN(nn.Module):
    """CNN-based Actor-Critic for standard 167-dim observation.

    Cards (156 dims) reshaped to 4x13 for spatial processing.
    Non-card features (11 dims) processed separately.
    Uses bfloat16 for faster computation on A100.
    """
    action_dim: int = NUM_ACTIONS
    dtype: jnp.dtype = DTYPE

    @nn.compact
    def __call__(self, obs):
        is_batched = obs.ndim == 2
        if not is_batched:
            obs = obs[None, :]
        batch_size = obs.shape[0]

        # Ensure input is bf16
        obs = obs.astype(self.dtype)

        # Split: 156 card dims + 11 other dims
        cards_flat = obs[:, :156]
        other = obs[:, 156:]

        # Reshape cards to (B, 3, 4, 13) -> (B, 4, 13, 3)
        cards = cards_flat.reshape(batch_size, 3, 4, 13)
        x = jnp.transpose(cards, (0, 2, 3, 1))

        # CNN for cards (all layers use bf16)
        x = nn.Conv(features=32, kernel_size=(1, 3), padding='SAME', dtype=self.dtype)(x)  # Runs
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 1), padding='SAME', dtype=self.dtype)(x)  # Sets
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)
        x = x.reshape(batch_size, -1)
        card_features = nn.Dense(128, dtype=self.dtype)(x)
        card_features = nn.relu(card_features)

        # Dense for non-card features
        other_features = nn.Dense(32, dtype=self.dtype)(other)
        other_features = nn.relu(other_features)

        # Combine
        combined = jnp.concatenate([card_features, other_features], axis=-1)

        # Separate heads
        actor_hidden = nn.Dense(128, dtype=self.dtype)(combined)
        actor_hidden = nn.relu(actor_hidden)
        logits = nn.Dense(self.action_dim, dtype=self.dtype)(actor_hidden)

        critic_hidden = nn.Dense(128, dtype=self.dtype)(combined)
        critic_hidden = nn.relu(critic_hidden)
        value = nn.Dense(1, dtype=self.dtype)(critic_hidden)

        value = jnp.squeeze(value, axis=-1)
        if not is_batched:
            logits = logits[0]
            value = value[0]
        return logits, value


class ActorCriticAlphaZero(nn.Module):
    """AlphaZero-style CNN for 728-dim (14 planes) observation."""
    action_dim: int = NUM_ACTIONS
    dtype: jnp.dtype = DTYPE

    @nn.compact
    def __call__(self, obs):
        is_batched = obs.ndim == 2
        if not is_batched:
            obs = obs[None, :]
        batch_size = obs.shape[0]

        # Ensure input is bf16
        obs = obs.astype(self.dtype)

        # Reshape to (B, 14, 4, 13) -> (B, 4, 13, 14)
        x = obs.reshape(batch_size, 14, 4, 13)
        x = jnp.transpose(x, (0, 2, 3, 1))

        # 3-layer CNN (all layers use bf16)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)

        x = x.reshape(batch_size, -1)
        x = nn.Dense(256, dtype=self.dtype)(x)
        x = nn.relu(x)

        # Separate heads
        actor_hidden = nn.Dense(128, dtype=self.dtype)(x)
        actor_hidden = nn.relu(actor_hidden)
        logits = nn.Dense(self.action_dim, dtype=self.dtype)(actor_hidden)

        critic_hidden = nn.Dense(128, dtype=self.dtype)(x)
        critic_hidden = nn.relu(critic_hidden)
        value = nn.Dense(1, dtype=self.dtype)(critic_hidden)

        value = jnp.squeeze(value, axis=-1)
        if not is_batched:
            logits = logits[0]
            value = value[0]
        return logits, value


# Select network based on mode
if OBS_MODE == "alphazero":
    ActorCritic = ActorCriticAlphaZero
else:
    ActorCritic = ActorCriticCNN


# =============================================================================
# PPO Functions (Standard)
# =============================================================================

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
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
    # Unpack batch - now includes old_values for value clipping
    obs, actions, old_log_probs, advantages, returns, legal_masks, old_values = batch

    logits, values = apply_fn(params, obs)
    logits = jnp.where(legal_masks, logits, -1e9)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)

    # Policy loss with clipping
    ratio = jnp.exp(action_log_probs - old_log_probs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss with clipping (prevents critic from changing too fast)
    v_loss_unclipped = (values - returns) ** 2
    v_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    v_loss_clipped = (v_clipped - returns) ** 2
    value_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

    # Entropy bonus
    probs = jax.nn.softmax(logits)
    log_probs_safe = jnp.where(legal_masks, log_probs, 0.0)
    entropy = -jnp.sum(probs * log_probs_safe, axis=-1).mean()

    total_loss = pg_loss + vf_coef * value_loss - ent_coef * entropy
    return total_loss, {'pg_loss': pg_loss, 'value_loss': value_loss, 'entropy': entropy}


# =============================================================================
# Training Loop
# =============================================================================

def train(
    num_envs=4096,
    num_steps=128,
    total_timesteps=2_000_000, # Increased for V2 as it's harder
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
    oracle=False,
):
    print(f"Training PPO (V2) on Gin Rummy", flush=True)
    oracle_str = " [ORACLE]" if oracle else ""
    print(f"  Obs mode: {OBS_MODE} ({OBS_DIM} dims){oracle_str}", flush=True)
    print(f"  Network: {'AlphaZero CNN' if OBS_MODE == 'alphazero' else 'Standard CNN'} + Separate heads", flush=True)
    print(f"  Dtype: {DTYPE}", flush=True)
    if oracle:
        print(f"  Oracle: Agent sees opponent's actual hand (upper bound test)", flush=True)
    print(f"  num_envs={num_envs}, num_steps={num_steps}", flush=True)
    print(f"  total_timesteps={total_timesteps}", flush=True)
    print(f"  Devices: {jax.devices()}", flush=True)

    key = jax.random.PRNGKey(seed)

    network = ActorCritic()
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, OBS_DIM), dtype=DTYPE)
    params = network.init(init_key, dummy_obs)

    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5)
    )
    opt_state = tx.init(params)

    print("Initializing environments...", flush=True)
    key, *env_keys = jax.random.split(key, num_envs + 1)
    env_keys = jnp.stack(env_keys)
    env_states = jax.vmap(env_init)(env_keys)
    print(f"  Environments initialized. Obs shape: {env_states['obs'].shape}", flush=True)

    apply_fn = jax.jit(network.apply)

    @jax.jit
    def get_action(params, obs, legal_mask, key):
        logits, value = apply_fn(params, obs)
        logits = jnp.where(legal_mask, logits, -1e9)
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]
        return action, value, log_prob

    @jax.jit
    def collect_rollout(params, env_states, key):
        def step_fn(carry, _):
            env_states, key = carry
            key, *action_keys = jax.random.split(key, num_envs + 1)
            action_keys = jnp.stack(action_keys)

            obs = env_states['obs']
            legal_masks = env_states['legal_mask']

            actions, values, log_probs = jax.vmap(
                lambda o, m, k: get_action(params, o, m, k)
            )(obs, legal_masks, action_keys)

            env_states, rewards = jax.vmap(env_step)(env_states, actions)
            dones = env_states['done']
            env_states = jax.vmap(env_reset_if_done)(env_states)

            transition = Transition(
                obs=obs,
                action=actions,
                reward=rewards,
                done=dones,
                value=values,
                log_prob=log_probs,
                legal_mask=legal_masks,
            )
            return (env_states, key), transition

        (env_states, key), trajectory = jax.lax.scan(
            step_fn, (env_states, key), None, num_steps
        )
        return env_states, trajectory, key

    @jax.jit
    def update_step(params, opt_state, trajectory):
        advantages, returns = compute_gae(
            trajectory.reward, trajectory.value, trajectory.done,
            gamma, gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = num_steps * num_envs
        minibatch_size = batch_size // num_minibatches

        obs = trajectory.obs.reshape(batch_size, -1)
        actions = trajectory.action.reshape(batch_size)
        old_log_probs = trajectory.log_prob.reshape(batch_size)
        advs = advantages.reshape(batch_size)
        rets = returns.reshape(batch_size)
        legal_masks = trajectory.legal_mask.reshape(batch_size, NUM_ACTIONS)
        old_values = trajectory.value.reshape(batch_size)  # For value clipping

        def epoch_step(carry, _):
            params, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, batch_size)

            def minibatch_step(carry, start_idx):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))
                batch = (obs[idx], actions[idx], old_log_probs[idx], advs[idx], rets[idx], legal_masks[idx], old_values[idx])
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

        key = jax.random.PRNGKey(0)
        (params, opt_state, _), (losses, auxs) = jax.lax.scan(
            epoch_step, (params, opt_state, key), None, update_epochs
        )
        return params, opt_state, losses.mean(), {k: v.mean() for k, v in auxs.items()}

    num_updates = total_timesteps // (num_envs * num_steps)
    print(f"\nStarting training: {num_updates} updates", flush=True)

    total_games = 0
    total_wins = 0
    total_reward = 0.0
    start_time = time.time()

    for update in range(num_updates):
        key, rollout_key = jax.random.split(key)
        env_states, trajectory, _ = collect_rollout(params, env_states, rollout_key)

        games_this_update = trajectory.done.sum()
        wins_this_update = (trajectory.reward > 0).sum()
        reward_this_update = trajectory.reward.sum()
        total_games += int(games_this_update)
        total_wins += int(wins_this_update)
        total_reward += float(reward_this_update)

        params, opt_state, loss, aux = update_step(params, opt_state, trajectory)

        if update % 1 == 0:  # Print every update
            elapsed = time.time() - start_time
            steps_done = (update + 1) * num_envs * num_steps
            steps_per_sec = steps_done / elapsed
            win_rate = total_wins / max(total_games, 1)
            avg_return = total_reward / max(total_games, 1)

            print(f"{elapsed:5.0f}s | "
                  f"Update {update:3d} | "
                  f"Steps: {steps_done/1e6:.1f}M | "
                  f"FPS: {steps_per_sec:,.0f} | "
                  f"Win: {win_rate:.1%} | "
                  f"AvgRet: {avg_return:+.3f} | "
                  f"Loss: {float(loss):.4f}", flush=True)

    elapsed = time.time() - start_time
    total_steps = num_updates * num_envs * num_steps
    print(f"\nTraining complete!", flush=True)
    print(f"  Total time: {elapsed:.1f}s", flush=True)
    print(f"  Total steps: {total_steps:,}", flush=True)
    print(f"  FPS: {total_steps / elapsed:,.0f}", flush=True)
    print(f"  Final win rate: {total_wins / max(total_games, 1):.2%}", flush=True)

    return params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle mode: agent sees opponent's actual hand")
    args = parser.parse_args()

    # Set oracle mode (must redefine make_observation before JIT compilation)
    if args.oracle:
        # Override the module-level make_observation with oracle version
        # Must use globals() to actually replace the module-level function
        def _make_obs_oracle(game_state, known_cards, agent_player):
            return _make_observation_standard(game_state, known_cards, agent_player, oracle=True)
        globals()['make_observation'] = jax.jit(_make_obs_oracle)

    train(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        seed=args.seed,
        ent_coef=args.ent_coef,
        num_minibatches=args.num_minibatches,
        oracle=args.oracle,
    )

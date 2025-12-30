"""Debug PPO - check loss progression and game lengths."""

import jax
import jax.numpy as jnp
import gin_rummy_jax as gin
import ppo_gin_rummy as ppo
import time

print("=" * 60)
print("Comparing game lengths: gin_rummy_core vs gin_rummy_jax")
print("=" * 60)

# Test game lengths with gin_rummy_jax
key = jax.random.PRNGKey(42)
total_steps = 0
games = 0

for g in range(100):
    state = gin.init_state()
    steps = 0
    for step in range(500):
        if state['done']:
            break
        action = gin.simple_bot_action_opt(state)
        state = gin.step(state, action.item())
        steps += 1
    total_steps += steps
    games += 1

print(f"gin_rummy_jax: {total_steps / games:.1f} avg steps per game")

# Count agent actions specifically
print("\n" + "=" * 60)
print("Checking agent action frequency")
print("=" * 60)

key = jax.random.PRNGKey(123)
env_state = ppo.env_init(key)

agent_actions = 0
total_steps = 0
games = 0

for i in range(5000):
    if env_state['done']:
        games += 1
        env_state = ppo.env_reset_if_done(env_state)

    legal_mask = env_state['legal_mask']
    if jnp.sum(legal_mask) == 0:
        print(f"No legal actions at step {i}!")
        break

    # Random action
    key, subkey = jax.random.split(key)
    probs = legal_mask.astype(jnp.float32) / jnp.sum(legal_mask)
    action = jax.random.choice(subkey, 241, p=probs)

    env_state, reward = ppo.env_step(env_state, action)
    agent_actions += 1

print(f"5000 agent actions -> {games} games completed")
print(f"Average agent actions per game: {5000 / max(games, 1):.1f}")

# Quick training to see loss progression
print("\n" + "=" * 60)
print("Quick training to check loss progression")
print("=" * 60)

import flax.linen as nn
import optax

# Small training run
NUM_ENVS = 256
NUM_STEPS = 64
UPDATES = 20

class ActorCritic(nn.Module):
    action_dim: int = 241

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(256)(obs)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)
        return logits, jnp.squeeze(value, axis=-1)

net = ActorCritic()
key = jax.random.PRNGKey(0)
params = net.init(key, jnp.zeros((1, 63)))
tx = optax.adam(3e-4)
opt_state = tx.init(params)

# Init envs
key, *env_keys = jax.random.split(key, NUM_ENVS + 1)
env_states = jax.vmap(ppo.env_init)(jnp.stack(env_keys))

def get_action(params, obs, legal_mask, key):
    logits, value = net.apply(params, obs[None, :])
    logits = jnp.where(legal_mask, logits[0], -1e9)
    probs = jax.nn.softmax(logits)
    action = jax.random.categorical(key, logits)
    log_prob = jnp.log(probs[action] + 1e-8)
    return action, value[0], log_prob

@jax.jit
def collect_and_update(params, opt_state, env_states, key):
    # Collect rollout
    def step_fn(carry, _):
        env_states, key = carry
        key, *keys = jax.random.split(key, NUM_ENVS + 1)
        keys = jnp.stack(keys)

        obs = env_states['obs']
        masks = env_states['legal_mask']

        actions, values, log_probs = jax.vmap(
            lambda o, m, k: get_action(params, o, m, k)
        )(obs, masks, keys)

        env_states, rewards = jax.vmap(ppo.env_step)(env_states, actions)
        dones = env_states['done']
        env_states = jax.vmap(ppo.env_reset_if_done)(env_states)

        return (env_states, key), (obs, actions, rewards, dones, values, log_probs, masks)

    (env_states, key), trajectory = jax.lax.scan(
        step_fn, (env_states, key), None, NUM_STEPS
    )

    obs, actions, rewards, dones, values, log_probs, masks = trajectory

    # Compute simple returns
    returns = jnp.zeros_like(rewards)
    running_return = jnp.zeros(NUM_ENVS)
    for t in range(NUM_STEPS - 1, -1, -1):
        running_return = rewards[t] + 0.99 * running_return * (1 - dones[t])
        returns = returns.at[t].set(running_return)

    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten
    obs_flat = obs.reshape(-1, 63)
    actions_flat = actions.flatten()
    advantages_flat = advantages.flatten()
    returns_flat = returns.flatten()
    log_probs_flat = log_probs.flatten()
    masks_flat = masks.reshape(-1, 241)

    def loss_fn(params):
        logits, vals = jax.vmap(lambda o: net.apply(params, o[None, :]))(obs_flat)
        logits = logits.squeeze(1)
        vals = vals.squeeze()

        logits = jnp.where(masks_flat, logits, -1e9)
        new_log_probs = jax.nn.log_softmax(logits)
        action_log_probs = jnp.take_along_axis(
            new_log_probs, actions_flat[:, None], axis=-1
        ).squeeze(-1)

        ratio = jnp.exp(action_log_probs - log_probs_flat)
        pg_loss = -jnp.mean(jnp.minimum(
            ratio * advantages_flat,
            jnp.clip(ratio, 0.8, 1.2) * advantages_flat
        ))
        value_loss = 0.5 * jnp.mean((vals - returns_flat) ** 2)

        return pg_loss + 0.5 * value_loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    wins = (rewards > 0).sum()
    games = dones.sum()

    return params, opt_state, env_states, key, loss, wins, games

print("\nUpdate | Loss | Wins/Games | Win%")
print("-" * 40)

total_wins = 0
total_games = 0

for i in range(UPDATES):
    key, subkey = jax.random.split(key)
    params, opt_state, env_states, key, loss, wins, games = collect_and_update(
        params, opt_state, env_states, subkey
    )
    total_wins += int(wins)
    total_games += int(games)
    win_rate = total_wins / max(total_games, 1)
    print(f"{i:6d} | {float(loss):.4f} | {total_wins}/{total_games} | {win_rate:.2%}")

print("\nIf loss is decreasing and win rate improving, learning is working.")
print("If loss is stuck or win rate not improving, there's a problem.")

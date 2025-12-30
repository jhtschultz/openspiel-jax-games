"""Debug PPO - check observations and gradients."""

import jax
import jax.numpy as jnp
import gin_rummy_jax as gin
import ppo_gin_rummy as ppo

print("=" * 50)
print("Test: Observation quality")
print("=" * 50)

key = jax.random.PRNGKey(42)
env_state = ppo.env_init(key)

obs = env_state['obs']
print(f"Observation shape: {obs.shape}")
print(f"Observation min: {obs.min():.3f}, max: {obs.max():.3f}")
print(f"Observation sum: {obs.sum():.3f}")

# Break down observation components
print(f"\nObservation breakdown:")
print(f"  Hand (0:52): {obs[:52].sum():.0f} cards, nonzero: {(obs[:52] > 0).sum()}")
print(f"  Phase one-hot (52:60): {obs[52:60]}")
print(f"  Upcard (60): {obs[60]:.3f}")
print(f"  Deck size (61): {obs[61]:.3f}")
print(f"  Deadwood (62): {obs[62]:.3f}")

# Check game state
gs = env_state['game_state']
print(f"\nGame state:")
print(f"  Phase: {gs['phase']}")
print(f"  Current player: {gs['current_player']}")
print(f"  Agent player: {env_state['agent_player']}")
print(f"  Hand size: {jnp.sum(ppo.gin.get_hand(gs, env_state['agent_player']))}")

# Check legal mask
print(f"\nLegal mask:")
print(f"  Total legal: {jnp.sum(env_state['legal_mask'])}")
print(f"  Legal actions: {jnp.where(env_state['legal_mask'])[0]}")

# Play a few steps and check obs changes
print("\n" + "=" * 50)
print("Test: Observation changes during play")
print("=" * 50)

for step in range(5):
    legal_mask = env_state['legal_mask']
    if jnp.sum(legal_mask) == 0:
        print(f"No legal actions at step {step}")
        break

    # Take first legal action
    action = jnp.where(legal_mask)[0][0]

    old_obs = env_state['obs']
    env_state, reward = ppo.env_step(env_state, action)
    new_obs = env_state['obs']

    diff = jnp.abs(new_obs - old_obs).sum()
    print(f"Step {step}: action={action}, obs_diff={diff:.3f}, reward={reward:.3f}, done={env_state['done']}")

    if env_state['done']:
        print("  Game ended!")
        break

# Check if network can learn something simple
print("\n" + "=" * 50)
print("Test: Network forward pass")
print("=" * 50)

import flax.linen as nn
import optax

# Create network
class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        actor = nn.Dense(241)(x)
        critic = nn.Dense(1)(x)
        return actor, critic

net = ActorCritic()
key = jax.random.PRNGKey(0)
params = net.init(key, jnp.zeros((1, 63)))

# Test forward pass
env_state = ppo.env_init(jax.random.PRNGKey(1))
obs = env_state['obs'][None, :]  # Add batch dim

logits, value = net.apply(params, obs)
print(f"Logits shape: {logits.shape}, range: [{logits.min():.3f}, {logits.max():.3f}]")
print(f"Value: {value[0, 0]:.3f}")

# Check action probabilities with masking
legal_mask = env_state['legal_mask'][None, :]
masked_logits = jnp.where(legal_mask, logits, -1e9)
probs = jax.nn.softmax(masked_logits, axis=-1)
print(f"Max prob: {probs.max():.3f}")
print(f"Entropy: {-(probs * jnp.log(probs + 1e-8)).sum():.3f}")

# Test gradient flow
print("\n" + "=" * 50)
print("Test: Gradient flow")
print("=" * 50)

def loss_fn(params, obs, action, advantage):
    logits, value = net.apply(params, obs)
    log_probs = jax.nn.log_softmax(logits)
    action_log_prob = log_probs[0, action]
    return -action_log_prob * advantage

grad_fn = jax.grad(loss_fn)
grads = grad_fn(params, obs, 54, 1.0)  # ACTION_PASS with advantage 1.0

# Check gradient magnitudes
total_grad_norm = 0
for key, val in jax.tree_util.tree_leaves_with_path(grads):
    norm = jnp.linalg.norm(val)
    total_grad_norm += norm ** 2
total_grad_norm = jnp.sqrt(total_grad_norm)
print(f"Total gradient norm: {total_grad_norm:.6f}")

if total_grad_norm < 1e-6:
    print("WARNING: Gradients are near zero!")
elif total_grad_norm > 100:
    print("WARNING: Gradients are very large!")
else:
    print("Gradient flow looks OK")

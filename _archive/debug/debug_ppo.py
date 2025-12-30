"""Debug PPO training to find why win rate is 0.26%."""

import jax
import jax.numpy as jnp
import gin_rummy_jax as gin

# Test 1: Do games complete with proper scores?
print("=" * 50)
print("Test 1: Game completion and scoring")
print("=" * 50)

key = jax.random.PRNGKey(42)
wins_p0 = 0
wins_p1 = 0
draws = 0
total_p0_score = 0

for g in range(100):
    state = gin.init_state()

    for step in range(300):
        if state['done']:
            break
        action = gin.simple_bot_action_opt(state)
        state = gin.step(state, action.item())

    if state['done']:
        p0_score = state['p0_score'].item()
        total_p0_score += p0_score
        if p0_score > 0:
            wins_p0 += 1
        elif p0_score < 0:
            wins_p1 += 1
        else:
            draws += 1
    else:
        print(f"Game {g} did not complete!")

print(f"P0 wins: {wins_p0}, P1 wins: {wins_p1}, Draws: {draws}")
print(f"Average P0 score: {total_p0_score / 100:.1f}")
print(f"P0 win rate: {wins_p0}%")

# Test 2: Check env wrapper behavior
print("\n" + "=" * 50)
print("Test 2: Environment wrapper behavior")
print("=" * 50)

# Import PPO env functions
import ppo_gin_rummy as ppo

key = jax.random.PRNGKey(123)
env_state = ppo.env_init(key)

print(f"Initial agent_player: {env_state['agent_player']}")
print(f"Initial game phase: {env_state['game_state']['phase']}")
print(f"Initial current_player: {env_state['game_state']['current_player']}")
print(f"Initial done: {env_state['done']}")
print(f"Legal actions: {jnp.sum(env_state['legal_mask'])} available")

# Take a few steps
total_reward = 0
games_completed = 0
agent_actions = 0

for i in range(1000):
    if env_state['done']:
        games_completed += 1
        env_state = ppo.env_reset_if_done(env_state)

    # Get legal action
    legal_mask = env_state['legal_mask']
    if jnp.sum(legal_mask) == 0:
        print(f"Step {i}: No legal actions! Phase: {env_state['game_state']['phase']}")
        break

    # Random legal action
    key, subkey = jax.random.split(key)
    probs = legal_mask.astype(jnp.float32) / jnp.sum(legal_mask)
    action = jax.random.choice(subkey, 241, p=probs)

    env_state, reward = ppo.env_step(env_state, action)
    total_reward += reward
    agent_actions += 1

    if reward != 0:
        print(f"Step {i}: Got reward {reward:.3f}, game done: {env_state['done']}")

print(f"\nAfter 1000 agent actions:")
print(f"  Games completed: {games_completed}")
print(f"  Total reward: {total_reward:.3f}")
print(f"  Avg reward per game: {total_reward / max(games_completed, 1):.3f}")

# Test 3: Check if agent is player 0 or 1
print("\n" + "=" * 50)
print("Test 3: Agent player distribution")
print("=" * 50)

p0_count = 0
p1_count = 0
for i in range(100):
    key, subkey = jax.random.split(key)
    env_state = ppo.env_init(subkey)
    if env_state['agent_player'] == 0:
        p0_count += 1
    else:
        p1_count += 1

print(f"Agent as P0: {p0_count}, Agent as P1: {p1_count}")

# Test 4: Check reward computation
print("\n" + "=" * 50)
print("Test 4: Reward computation")
print("=" * 50)

# Create a completed game state manually
state = gin.init_state()
for step in range(300):
    if state['done']:
        break
    action = gin.simple_bot_action_opt(state)
    state = gin.step(state, action.item())

print(f"Game done: {state['done']}")
print(f"p0_score: {state['p0_score']}")
print(f"p1_score: {state['p1_score']}")
print(f"winner: {state['winner']}")
print(f"knocker: {state['knocker']}")

# Test reward from both perspectives
for agent_player in [0, 1]:
    reward = ppo.compute_game_score(state, agent_player)
    print(f"Agent as P{agent_player}: score = {reward}")

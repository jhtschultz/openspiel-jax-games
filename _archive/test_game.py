"""Test that games complete properly."""
import gin_rummy_core as gin
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
state = gin.init_state()
print("Initial state phase:", state['phase'])

for i in range(300):
    if state['done']:
        print(f"Game done at step {i}, winner: {state['winner']}")
        print(f"Knocker: {state['knocker']}, knocker_dw: {state['knocker_deadwood']}")
        break

    # Chance or player action
    is_chance = (state['phase'] == gin.PHASE_DEAL) or state['waiting_stock_draw']
    if is_chance:
        # Chance: pick random card from deck
        deck = state['deck'].astype(jnp.float32)
        deck_probs = deck / jnp.maximum(deck.sum(), 1.0)
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, 52, p=deck_probs)
    else:
        action = gin.simple_bot_action_opt(state)

    state = gin.step(state, action)

    if i % 50 == 0:
        print(f"Step {i}: phase={state['phase']}, player={state['current_player']}, done={state['done']}")

if not state['done']:
    print("Game not done after 300 steps")
    print(f"Final phase: {state['phase']}, knocker: {state['knocker']}")

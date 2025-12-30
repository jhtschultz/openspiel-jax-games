"""Debug gin_rummy_core game flow."""
import gin_rummy_core as gin
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
state = gin.init_state()
print("Initial phase:", int(state["phase"]))

for i in range(150):
    if state["done"]:
        print(f"Game done at step {i}")
        break

    is_chance = (state["phase"] == gin.PHASE_DEAL) or state["waiting_stock_draw"]
    if is_chance:
        deck = state["deck"].astype(jnp.float32)
        deck_sum = deck.sum()
        if deck_sum > 0:
            key, subkey = jax.random.split(key)
            action = jax.random.choice(subkey, 52, p=deck / deck_sum)
        else:
            print(f"Deck empty at step {i}")
            break
    else:
        action = gin.simple_bot_action_opt(state)

    phase = int(state["phase"])
    player = int(state["current_player"])
    act = int(action)

    # Only print every 10 steps to reduce output
    if i % 10 == 0 or phase in [4, 5, 6, 7]:  # Always print knock/layoff/wall/gameover
        deck_count = int(state["deck"].sum())
        print(f"Step {i}: phase={phase}, player={player}, action={act}, deck={deck_count}")

    state = gin.step(state, action)

phase_final = int(state["phase"])
done_final = bool(state["done"])
print(f"Final: phase={phase_final}, done={done_final}")

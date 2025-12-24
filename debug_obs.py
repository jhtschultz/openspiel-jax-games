"""Debug observation tensor format."""

import numpy as np
import pyspiel
import connect_four_jax

orig_game = pyspiel.load_game("connect_four")
jax_game = pyspiel.load_game("python_connect_four_jax")

orig_state = orig_game.new_initial_state()
jax_state = jax_game.new_initial_state()

# Play a few moves
for action in [3, 4, 3, 4, 3, 4]:  # Player 0 building vertically
    orig_state.apply_action(action)
    jax_state.apply_action(action)

print("Board state (not terminal):")
print(f"Terminal: {orig_state.is_terminal()}")
print(f"Current player: {orig_state.current_player()}")
print(f"\nOrig:\n{orig_state}")
print(f"\nJax:\n{jax_state}")

print("\n=== Observation from orig (player 0) ===")
orig_obs = np.array(orig_state.observation_tensor(0)).reshape(3, 6, 7)
print(f"Plane 0:\n{orig_obs[0]}")
print(f"Plane 1:\n{orig_obs[1]}")
print(f"Plane 2:\n{orig_obs[2]}")

print("\n=== Observation from jax (player 0) ===")
jax_obs = np.array(jax_state.observation_tensor(0)).reshape(3, 6, 7)
print(f"Plane 0:\n{jax_obs[0]}")
print(f"Plane 1:\n{jax_obs[1]}")
print(f"Plane 2:\n{jax_obs[2]}")

# Now make it terminal (player 0 wins with vertical 4)
orig_state.apply_action(3)
jax_state.apply_action(3)

print("\n\n=== After win (terminal) ===")
print(f"Terminal: {orig_state.is_terminal()}")
print(f"\nOrig:\n{orig_state}")

print("\n=== Terminal observation from orig (player 0) ===")
orig_obs = np.array(orig_state.observation_tensor(0)).reshape(3, 6, 7)
print(f"Plane 0:\n{orig_obs[0]}")
print(f"Plane 1:\n{orig_obs[1]}")
print(f"Plane 2:\n{orig_obs[2]}")

"""Check existing connect_four game interface."""

import pyspiel
import numpy as np

# Load the game
game = pyspiel.load_game("connect_four")

print("=== Game Info ===")
print(f"Name: {game.get_type().short_name}")
print(f"Num players: {game.num_players()}")
print(f"Num distinct actions: {game.num_distinct_actions()}")
print(f"Max game length: {game.max_game_length()}")

# Check observation/information state tensor specs
print(f"\nObservation tensor shape: {game.observation_tensor_shape()}")

# Create initial state
state = game.new_initial_state()
print(f"\n=== Initial State ===")
print(f"Current player: {state.current_player()}")
print(f"Legal actions: {state.legal_actions()}")
print(f"Is terminal: {state.is_terminal()}")

# Check observation tensor
obs = state.observation_tensor()
print(f"Observation tensor length: {len(obs)}")
print(f"Observation tensor: {obs[:20]}... (first 20)")

# Play a few moves and check representation
print(f"\n=== After some moves ===")
state.apply_action(3)  # Player 0 drops in column 3
print(f"After action 3:")
print(state)
print(f"Current player: {state.current_player()}")
print(f"Legal actions: {state.legal_actions()}")

state.apply_action(3)  # Player 1 drops in column 3
print(f"\nAfter action 3 again:")
print(state)

# Check how the board is represented
print(f"\n=== Observation tensor structure ===")
obs = state.observation_tensor()
obs_array = np.array(obs)
shape = game.observation_tensor_shape()
print(f"Shape: {shape}")
reshaped = obs_array.reshape(shape)
print(f"Reshaped observation:\n{reshaped}")

# Play until terminal to see returns
print(f"\n=== Playing random game ===")
state = game.new_initial_state()
while not state.is_terminal():
    action = np.random.choice(state.legal_actions())
    state.apply_action(action)
print(state)
print(f"Returns: {state.returns()}")

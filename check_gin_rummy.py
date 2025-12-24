"""Analyze gin_rummy OpenSpiel interface."""

import pyspiel
import numpy as np

# Load the game
game = pyspiel.load_game("gin_rummy")

print("=== Game Info ===")
print(f"Name: {game.get_type().short_name}")
print(f"Num players: {game.num_players()}")
print(f"Num distinct actions: {game.num_distinct_actions()}")
print(f"Max game length: {game.max_game_length()}")
print(f"Game type: {game.get_type()}")

# Check tensor specs
print(f"\nObservation tensor shape: {game.observation_tensor_shape()}")
try:
    print(f"Information state tensor shape: {game.information_state_tensor_shape()}")
except:
    print("Information state tensor: not implemented")

# Create initial state
state = game.new_initial_state()
print(f"\n=== Initial State ===")
print(f"Current player: {state.current_player()}")
print(f"Is chance node: {state.is_chance_node()}")
print(f"Is terminal: {state.is_terminal()}")

# Handle chance nodes (dealing cards)
print(f"\n=== Dealing cards (chance nodes) ===")
move_count = 0
while state.is_chance_node():
    outcomes = state.chance_outcomes()
    # Pick first outcome for determinism
    action, prob = outcomes[0]
    state.apply_action(action)
    move_count += 1
    if move_count <= 5:
        print(f"  Dealt card (action {action})")
    elif move_count == 6:
        print(f"  ... (dealing more cards)")

print(f"  Total chance actions for deal: {move_count}")

print(f"\n=== After deal ===")
print(f"Current player: {state.current_player()}")
print(f"Is chance node: {state.is_chance_node()}")
print(f"Legal actions: {state.legal_actions()}")
print(f"Num legal actions: {len(state.legal_actions())}")

# Print state
print(f"\nState string:\n{state}")

# Check observation
print(f"\n=== Observation ===")
obs = state.observation_tensor()
print(f"Observation tensor length: {len(obs)}")
obs_shape = game.observation_tensor_shape()
print(f"Observation tensor shape: {obs_shape}")

# Play a few moves
print(f"\n=== Playing a few moves ===")
for i in range(5):
    if state.is_terminal():
        break
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action, _ = outcomes[0]
        print(f"  Chance: action {action}")
    else:
        legal = state.legal_actions()
        action = legal[0]
        player = state.current_player()
        print(f"  Player {player}: action {action} (of {len(legal)} legal)")
    state.apply_action(action)

print(f"\nState after moves:\n{state}")

# Check action meanings
print(f"\n=== Action space analysis ===")
print(f"Total actions: {game.num_distinct_actions()}")

# Try to understand action encoding
state2 = game.new_initial_state()
while state2.is_chance_node():
    outcomes = state2.chance_outcomes()
    action, _ = outcomes[0]
    state2.apply_action(action)

legal = state2.legal_actions()
print(f"Legal actions at start of game: {legal}")
print(f"Number of legal actions: {len(legal)}")

# Action string representations
print(f"\nAction meanings (first 10 legal):")
for a in legal[:10]:
    try:
        meaning = state2.action_to_string(state2.current_player(), a)
        print(f"  Action {a}: {meaning}")
    except:
        print(f"  Action {a}: (no string repr)")

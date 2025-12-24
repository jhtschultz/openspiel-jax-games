"""Map out complete gin_rummy action space and observation encoding."""

import pyspiel
import numpy as np

game = pyspiel.load_game("gin_rummy")

print("=== Complete Action Space ===")
# Based on observation: 241 actions total
# 0-51: Cards (for discard/meld selection)
# 52: Draw upcard
# 53: Draw stock
# 54: Pass
# 55: Knock
# 56+: Need to discover

# The remaining actions are likely for:
# - Gin declaration
# - Layoff during knock
# - Meld arrangements

# Let's try to trigger all action types by playing many games
def explore_actions(num_games=100):
    action_meanings = {}

    for game_idx in range(num_games):
        state = game.new_initial_state()
        np.random.seed(game_idx)

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action, _ = outcomes[np.random.randint(len(outcomes))]
            else:
                legal = state.legal_actions()
                for a in legal:
                    if a not in action_meanings:
                        try:
                            action_meanings[a] = state.action_to_string(state.current_player(), a)
                        except:
                            action_meanings[a] = f"action_{a}"
                action = np.random.choice(legal)
            state.apply_action(action)

    return action_meanings

print("Exploring actions across 100 games...")
action_meanings = explore_actions(100)
print(f"Found {len(action_meanings)} unique actions\n")

# Categorize
card_actions = {a: m for a, m in action_meanings.items() if a < 52}
special_actions = {a: m for a, m in action_meanings.items() if a >= 52}

print(f"Card actions (0-51): {len(card_actions)} found")
print(f"Special actions (52+): {len(special_actions)} found")

print("\nSpecial actions:")
for a in sorted(special_actions.keys()):
    print(f"  {a}: {special_actions[a]}")

# Now understand observation tensor structure
print("\n\n=== Observation Tensor Structure ===")

state = game.new_initial_state()
while state.is_chance_node():
    outcomes = state.chance_outcomes()
    state.apply_action(outcomes[0][0])

obs = np.array(state.observation_tensor())
print(f"Total length: {len(obs)}")

# Based on gin_rummy.cc, observation likely includes:
# - Current player's hand (52)
# - Cards seen in discard (52)
# - Current upcard (52)
# - Stock size encoding
# - Phase encoding
# - Deadwood values
# - Etc.

# Let's segment and analyze
segments = [
    (0, 52, "Segment 0-51 (hand?)"),
    (52, 104, "Segment 52-103"),
    (104, 156, "Segment 104-155"),
    (156, 208, "Segment 156-207"),
    (208, 260, "Segment 208-259"),
    (260, 312, "Segment 260-311"),
    (312, 364, "Segment 312-363"),
    (364, 416, "Segment 364-415"),
    (416, 468, "Segment 416-467"),
    (468, 520, "Segment 468-519"),
    (520, 572, "Segment 520-571"),
    (572, 624, "Segment 572-623"),
    (624, 644, "Segment 624-643 (misc)"),
]

for start, end, name in segments:
    segment = obs[start:end]
    nonzero = np.count_nonzero(segment)
    if nonzero > 0:
        print(f"{name}: {nonzero} nonzero, sum={segment.sum():.0f}")
        if nonzero <= 15:
            indices = np.where(segment > 0)[0]
            print(f"  Nonzero at: {list(indices)}")

# Game parameters
print("\n\n=== Game Parameters ===")
params = game.get_parameters()
print(f"Parameters: {params}")

# Check for Oklahoma variant and other options
print("\nDefault game info:")
print(f"  Observation tensor size: {game.observation_tensor_size()}")

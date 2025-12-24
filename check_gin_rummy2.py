"""Deep dive into gin_rummy action space and state."""

import pyspiel
import numpy as np

game = pyspiel.load_game("gin_rummy")

print("=== Full Action Space Analysis ===")
print(f"Total distinct actions: {game.num_distinct_actions()}")

# Play through different phases to see all action types
def play_to_phase(target_moves=50):
    """Play random moves and collect action info."""
    state = game.new_initial_state()
    np.random.seed(42)

    action_info = {}
    phases_seen = set()

    for _ in range(target_moves):
        if state.is_terminal():
            break

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action, _ = outcomes[0]
            state.apply_action(action)
            continue

        # Get current phase from state string
        state_str = str(state)
        for line in state_str.split('\n'):
            if 'Phase:' in line:
                phase = line.split('Phase:')[1].strip()
                phases_seen.add(phase)

        legal = state.legal_actions()
        for a in legal:
            if a not in action_info:
                try:
                    meaning = state.action_to_string(state.current_player(), a)
                    action_info[a] = meaning
                except:
                    action_info[a] = f"Unknown action {a}"

        action = np.random.choice(legal)
        state.apply_action(action)

    return action_info, phases_seen

action_info, phases = play_to_phase(200)

print(f"\nPhases seen: {phases}")
print(f"\nActions seen: {len(action_info)}")

# Group actions by type
draw_actions = []
discard_actions = []
knock_actions = []
other_actions = []

for a, meaning in sorted(action_info.items()):
    if 'Draw' in meaning:
        draw_actions.append((a, meaning))
    elif 'Discard' in meaning:
        discard_actions.append((a, meaning))
    elif 'Knock' in meaning or 'knock' in meaning:
        knock_actions.append((a, meaning))
    else:
        other_actions.append((a, meaning))

print(f"\n=== Draw actions ({len(draw_actions)}) ===")
for a, m in draw_actions[:10]:
    print(f"  {a}: {m}")

print(f"\n=== Discard actions ({len(discard_actions)}) ===")
for a, m in discard_actions[:10]:
    print(f"  {a}: {m}")
if len(discard_actions) > 10:
    print(f"  ... and {len(discard_actions) - 10} more")

print(f"\n=== Knock actions ({len(knock_actions)}) ===")
for a, m in knock_actions[:10]:
    print(f"  {a}: {m}")

print(f"\n=== Other actions ({len(other_actions)}) ===")
for a, m in other_actions:
    print(f"  {a}: {m}")

# Now let's understand the observation tensor
print(f"\n\n=== Observation Tensor Analysis ===")
state = game.new_initial_state()

# Deal cards
while state.is_chance_node():
    outcomes = state.chance_outcomes()
    state.apply_action(outcomes[0][0])

obs = np.array(state.observation_tensor())
print(f"Observation shape: {obs.shape}")
print(f"Non-zero elements: {np.count_nonzero(obs)}")
print(f"Unique values: {np.unique(obs)}")

# Try to understand structure
print(f"\nObservation segments (looking for patterns):")
print(f"  First 52 elements (hand?): sum={obs[:52].sum()}")
print(f"  Next 52 elements: sum={obs[52:104].sum()}")
print(f"  Elements 104-156: sum={obs[104:156].sum()}")

# Let's look at what changes after a move
obs1 = np.array(state.observation_tensor())
state.apply_action(state.legal_actions()[0])  # Draw
obs2 = np.array(state.observation_tensor())

diff = np.where(obs1 != obs2)[0]
print(f"\nIndices that changed after draw: {diff}")

# Play a full game to see returns
print(f"\n\n=== Playing a complete game ===")
state = game.new_initial_state()
np.random.seed(123)
move_count = 0

while not state.is_terminal():
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action, _ = outcomes[np.random.randint(len(outcomes))]
    else:
        legal = state.legal_actions()
        action = np.random.choice(legal)
        move_count += 1
    state.apply_action(action)

print(f"Game ended after {move_count} player moves")
print(f"Returns: {state.returns()}")
print(f"\nFinal state:\n{state}")

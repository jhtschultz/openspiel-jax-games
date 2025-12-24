"""Compare JAX Gin Rummy vs C++ OpenSpiel implementation."""

import numpy as np
import pyspiel
import jax
import jax.numpy as jnp
import gin_rummy_jax as jax_gin

# Mapping from JAX action to C++ action
# They should be the same: 0-51=discard, 52=draw_upcard, 53=draw_stock, 54=pass, 55=knock


def extract_hands_from_cpp(cpp_state):
    """Extract dealt hands from C++ state string."""
    state_str = str(cpp_state)

    # Parse player hands from state string
    # Format shows cards like "As 2c Kh" etc
    p0_cards = []
    p1_cards = []
    upcard = None

    lines = state_str.split('\n')
    for i, line in enumerate(lines):
        if 'Player0:' in line:
            # Next few lines have the cards
            for j in range(i+1, min(i+6, len(lines))):
                for card in parse_cards_from_line(lines[j]):
                    p0_cards.append(card)
        if 'Player1:' in line:
            for j in range(i+1, min(i+6, len(lines))):
                for card in parse_cards_from_line(lines[j]):
                    p1_cards.append(card)
        if 'Upcard:' in line:
            parts = line.split()
            for p in parts:
                if len(p) == 2 and p[0] in 'A23456789TJQK' and p[1] in 'scdh':
                    upcard = card_str_to_idx(p)

    return p0_cards, p1_cards, upcard


def parse_cards_from_line(line):
    """Parse card strings from a display line."""
    cards = []
    ranks = 'A23456789TJQK'
    suits = 'scdh'

    i = 0
    while i < len(line) - 1:
        if line[i] in ranks and line[i+1] in suits:
            cards.append(card_str_to_idx(line[i:i+2]))
            i += 2
        else:
            i += 1
    return cards


def card_str_to_idx(s):
    """Convert card string like 'As' to index 0-51."""
    ranks = 'A23456789TJQK'
    suits = 'scdh'
    rank = ranks.index(s[0])
    suit = suits.index(s[1])
    return rank * 4 + suit


def create_jax_state_from_hands(jax_game, p0_cards, p1_cards, upcard, stock_order):
    """Create a JAX state with specific hands."""
    # Create initial state
    jax_state = jax_game.new_initial_state()

    # Manually set the internal JAX state
    p0_mask = np.zeros(52, dtype=np.int8)
    p1_mask = np.zeros(52, dtype=np.int8)
    for c in p0_cards:
        p0_mask[c] = 1
    for c in p1_cards:
        p1_mask[c] = 1

    jax_state._jax_state = {
        'player0_hand': jnp.array(p0_mask),
        'player1_hand': jnp.array(p1_mask),
        'stock_mask': jnp.zeros(52, dtype=jnp.int8),  # Will set below
        'discard_mask': jnp.zeros(52, dtype=jnp.int8),
        'upcard': jnp.int8(upcard),
        'stock_top_idx': jnp.int8(0),
        'stock_order': jnp.array(stock_order, dtype=jnp.int32),
        'current_player': jnp.int8(0),
        'phase': jnp.int8(jax_gin.PHASE_FIRST_UPCARD),
        'done': jnp.bool_(False),
        'winner': jnp.int8(-1),
        'p0_score': jnp.int32(0),
        'p1_score': jnp.int32(0),
        'drawn_from_discard': jnp.int8(-1),
        'repeated_move': jnp.bool_(False),
        'pass_count': jnp.int8(0),
        # Knock/layoff state
        'knocked': jnp.zeros(2, dtype=jnp.bool_),
        'knocker': jnp.int8(-1),
        'layed_melds': jnp.zeros((2, jax_gin.NUM_MELDS), dtype=jnp.bool_),
        'layoffs_mask': jnp.zeros(52, dtype=jnp.bool_),
        'finished_layoffs': jnp.bool_(False),
        'deadwood': jnp.zeros(2, dtype=jnp.int32),
    }

    # Set stock mask for remaining cards
    stock_mask = np.zeros(52, dtype=np.int8)
    for c in stock_order:
        stock_mask[c] = 1
    jax_state._jax_state['stock_mask'] = jnp.array(stock_mask)
    jax_state._cached_legal_actions = None

    return jax_state


def compare_basic_game_flow(num_games=100, verbose=False):
    """Compare basic game flow (before knock phase)."""

    cpp_game = pyspiel.load_game("gin_rummy")
    jax_game = pyspiel.load_game("python_gin_rummy_jax")

    disagreements = []

    for game_idx in range(num_games):
        rng = np.random.RandomState(game_idx)

        cpp_state = cpp_game.new_initial_state()

        # Deal cards in C++ (chance nodes) and track what was dealt
        # C++ deals: cards 0-9 to p0, 10-19 to p1, 20 is upcard, 21-51 is stock
        dealt_cards = []
        while cpp_state.is_chance_node():
            outcomes = cpp_state.chance_outcomes()
            action, _ = outcomes[rng.randint(len(outcomes))]
            dealt_cards.append(action)
            cpp_state.apply_action(action)

        # Extract deal order (only 21 cards dealt: 10+10+1)
        p0_cards = dealt_cards[0:10]
        p1_cards = dealt_cards[10:20]
        upcard = dealt_cards[20]

        # Stock is the remaining cards - we don't know order yet (revealed via chance nodes)
        used = set(p0_cards) | set(p1_cards) | {upcard}
        remaining = [c for c in range(52) if c not in used]

        # Create JAX state with same hands (stock will be synced via chance nodes)
        jax_state = create_jax_state_from_hands(jax_game, p0_cards, p1_cards, upcard, remaining)

        move_count = 0
        while move_count < 200:
            # Handle C++ chance nodes (stock draws)
            if cpp_state.is_chance_node():
                outcomes = cpp_state.chance_outcomes()
                drawn_card, _ = outcomes[rng.randint(len(outcomes))]
                cpp_state.apply_action(drawn_card)

                # Sync the drawn card to JAX - add it to current player's hand
                player = int(jax_state._jax_state['current_player'])
                if player == 0:
                    jax_state._jax_state['player0_hand'] = jax_state._jax_state['player0_hand'].at[drawn_card].set(1)
                else:
                    jax_state._jax_state['player1_hand'] = jax_state._jax_state['player1_hand'].at[drawn_card].set(1)
                jax_state._jax_state['stock_mask'] = jax_state._jax_state['stock_mask'].at[drawn_card].set(0)
                # Move to discard phase
                jax_state._jax_state['phase'] = jnp.int8(jax_gin.PHASE_DISCARD)
                jax_state._jax_state['drawn_from_discard'] = jnp.int8(-1)
                jax_state._cached_legal_actions = None
                continue

            # Both should be non-terminal at same time
            cpp_term = cpp_state.is_terminal()
            jax_term = jax_state.is_terminal()

            if cpp_term or jax_term:
                if cpp_term != jax_term:
                    jax_s = jax_state._jax_state
                    disagreements.append({
                        'game': game_idx,
                        'move': move_count,
                        'type': 'terminal_mismatch',
                        'cpp_terminal': cpp_term,
                        'jax_terminal': jax_term,
                        'jax_phase': int(jax_s['phase']),
                        'jax_upcard': int(jax_s['upcard']),
                        'jax_pass_count': int(jax_s['pass_count']),
                        'jax_drawn_from_discard': int(jax_s['drawn_from_discard']),
                        'jax_done': bool(jax_s['done']),
                        'jax_winner': int(jax_s['winner']),
                        'cpp_state_str': str(cpp_state)[:400],
                    })
                break

            cpp_player = cpp_state.current_player()
            jax_player = jax_state.current_player()

            if cpp_player != jax_player:
                disagreements.append({
                    'game': game_idx,
                    'move': move_count,
                    'type': 'player_mismatch',
                    'cpp_player': cpp_player,
                    'jax_player': jax_player,
                    'cpp_state': str(cpp_state)[:200],
                    'jax_state': str(jax_state)[:200],
                })
                break

            cpp_legal = set(cpp_state.legal_actions())
            jax_legal = set(jax_state.legal_actions())

            # For now, skip knock-related actions since JAX doesn't handle them fully yet
            cpp_legal_basic = set(a for a in cpp_legal if a <= 55)  # Include knock
            jax_legal_basic = set(a for a in jax_legal if a <= 55)

            # Check if basic actions agree (ignoring knock for now)
            cpp_no_knock = cpp_legal_basic - {55}
            jax_no_knock = jax_legal_basic - {55}

            if cpp_no_knock != jax_no_knock:
                jax_s = jax_state._jax_state
                disagreements.append({
                    'game': game_idx,
                    'move': move_count,
                    'type': 'legal_actions_mismatch',
                    'cpp_legal': sorted(cpp_no_knock),
                    'jax_legal': sorted(jax_no_knock),
                    'cpp_only': sorted(cpp_no_knock - jax_no_knock),
                    'jax_only': sorted(jax_no_knock - cpp_no_knock),
                    'jax_phase': int(jax_s['phase']),
                    'jax_upcard': int(jax_s['upcard']),
                    'jax_pass_count': int(jax_s['pass_count']),
                    'jax_drawn_from_discard': int(jax_s['drawn_from_discard']),
                    'cpp_state_str': str(cpp_state)[:300],
                })
                break

            # Pick a random action that both agree on (avoid knock for now)
            common = cpp_no_knock & jax_no_knock
            if not common:
                disagreements.append({
                    'game': game_idx,
                    'move': move_count,
                    'type': 'no_common_actions',
                    'cpp_legal': sorted(cpp_no_knock),
                    'jax_legal': sorted(jax_no_knock),
                })
                break

            action = rng.choice(list(common))

            # Special handling for draw stock - C++ goes to chance node, JAX uses fixed order
            if action == 53:  # Draw stock
                cpp_state.apply_action(action)
                # Don't apply to JAX yet - will handle in chance node above
            else:
                cpp_state.apply_action(action)
                jax_state.apply_action(action)

            move_count += 1

        if verbose and game_idx % 10 == 0:
            print(f"Game {game_idx}: {move_count} moves, {len(disagreements)} disagreements so far")

    return disagreements


def analyze_phase_distribution(num_games=100):
    """Analyze what phases we're hitting in random play."""

    cpp_game = pyspiel.load_game("gin_rummy")

    phase_counts = {
        'FirstUpcard': 0,
        'Draw': 0,
        'Discard': 0,
        'Knock': 0,
        'Layoff': 0,
        'Wall': 0,
    }

    knock_count = 0
    wall_count = 0

    for game_idx in range(num_games):
        rng = np.random.RandomState(game_idx + 1000)
        state = cpp_game.new_initial_state()

        # Deal cards
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            action, _ = outcomes[rng.randint(len(outcomes))]
            state.apply_action(action)

        move_count = 0
        while not state.is_terminal() and move_count < 300:
            state_str = str(state)

            # Extract phase
            for phase in phase_counts:
                if f"Phase: {phase}" in state_str:
                    phase_counts[phase] += 1
                    break

            # Check for knock opportunity
            legal = state.legal_actions()
            if 55 in legal:
                knock_count += 1
                # Randomly decide to knock
                if rng.random() < 0.3:  # 30% knock rate
                    action = 55
                else:
                    legal.remove(55)
                    action = rng.choice(legal) if legal else 55
            else:
                action = rng.choice(legal)

            state.apply_action(action)
            move_count += 1

        if "Wall" in str(state):
            wall_count += 1

    print(f"Phase distribution over {num_games} games:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count}")
    print(f"\nKnock opportunities: {knock_count}")
    print(f"Games reaching wall: {wall_count}")


def test_knock_game():
    """Test a complete game with knock."""
    import pyspiel

    print("Testing knock game flow...")

    jax_game = pyspiel.load_game("python_gin_rummy_jax")
    state = jax_game.new_initial_state()

    print(f"Initial state:\n{state}\n")

    move_count = 0
    knock_tested = False

    while not state.is_terminal() and move_count < 300:
        legal = state.legal_actions()
        phase = int(state._jax_state['phase'])
        player = state.current_player()

        phase_names = ['Deal', 'FirstUpcard', 'Draw', 'Discard', 'Knock', 'Layoff', 'GameOver']

        if phase == jax_gin.PHASE_KNOCK or phase == jax_gin.PHASE_LAYOFF:
            print(f"Move {move_count}: Phase={phase_names[phase]}, Player={player}, Legal={legal}")

        # If knock is available, use it
        if 55 in legal and not knock_tested:  # ACTION_KNOCK
            print(f"Move {move_count}: KNOCK available! Taking it...")
            state.apply_action(55)
            knock_tested = True
        elif phase == jax_gin.PHASE_KNOCK:
            # In knock phase, prefer pass if available (to finish laying melds)
            if 54 in legal:  # ACTION_PASS
                state.apply_action(54)
            elif legal:
                # Otherwise take first meld action or discard
                state.apply_action(legal[0])
        elif phase == jax_gin.PHASE_LAYOFF:
            # In layoff phase, prefer pass
            if 54 in legal:
                state.apply_action(54)
            elif legal:
                state.apply_action(legal[0])
        else:
            # Random action
            action = np.random.choice(legal)
            state.apply_action(action)

        move_count += 1

    print(f"\nFinal state:\n{state}")
    print(f"Game ended after {move_count} moves")
    print(f"Returns: {state.returns()}")
    print(f"Knocked: {state._jax_state['knocked']}")
    print(f"Deadwood: {state._jax_state['deadwood']}")

    return knock_tested


if __name__ == "__main__":
    print("=== Comparing JAX vs C++ Gin Rummy ===\n")

    print("1. Analyzing phase distribution in C++:")
    analyze_phase_distribution(20)

    print("\n2. Running comparison (basic game flow):")
    disagreements = compare_basic_game_flow(100, verbose=True)

    print(f"\n=== Results ===")
    print(f"Total disagreements: {len(disagreements)}")

    if disagreements:
        print("\nFirst 3 disagreements (detailed):")
        for d in disagreements[:3]:
            print(f"\n  Game {d['game']}, Move {d['move']}: {d['type']}")
            if d['type'] == 'terminal_mismatch':
                print(f"    cpp_terminal: {d.get('cpp_terminal')}, jax_terminal: {d.get('jax_terminal')}")
                print(f"    jax_done: {d.get('jax_done')}, jax_winner: {d.get('jax_winner')}")
            else:
                print(f"    cpp_legal: {d.get('cpp_legal')}")
                print(f"    jax_legal: {d.get('jax_legal')}")
                print(f"    cpp_only: {d.get('cpp_only')}, jax_only: {d.get('jax_only')}")
            print(f"    jax_phase: {d.get('jax_phase')}, jax_upcard: {d.get('jax_upcard')}")
            print(f"    jax_pass_count: {d.get('jax_pass_count')}, drawn_from_discard: {d.get('jax_drawn_from_discard')}")
            print(f"    cpp_state:\n{d.get('cpp_state_str', '')}")
    else:
        print("No disagreements found in basic game flow!")

    print("\n3. Testing knock game flow:")
    for i in range(5):
        print(f"\n--- Knock Test Game {i+1} ---")
        np.random.seed(i * 1000 + 42)
        knocked = test_knock_game()
        if knocked:
            print("Successfully tested knock!")
        else:
            print("No knock opportunity arose in this game.")

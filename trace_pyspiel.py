"""Trace a game through pyspiel only to see exact C++ behavior."""

import numpy as np
import pyspiel

def card_str(idx):
    if idx < 0:
        return "None"
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['s', 'c', 'd', 'h']
    return ranks[idx % 13] + suits[idx // 13]

def action_str(action):
    if action < 52:
        return f"discard_{card_str(action)}"
    elif action == 52:
        return "draw_upcard"
    elif action == 53:
        return "draw_stock"
    elif action == 54:
        return "pass"
    elif action == 55:
        return "knock"
    else:
        return f"meld_{action - 56}"

def trace_game(seed, stop_at_gin=True):
    """Trace a game using pyspiel only."""
    game = pyspiel.load_game("gin_rummy")
    bot_p0 = pyspiel.make_simple_gin_rummy_bot(game.get_parameters(), 0)
    bot_p1 = pyspiel.make_simple_gin_rummy_bot(game.get_parameters(), 1)
    rng = np.random.RandomState(seed)
    state = game.new_initial_state()

    step = 0
    in_layoff = False
    knocker = None

    while not state.is_terminal():
        player = state.current_player()

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = rng.choice(actions, p=probs)
            if step >= 21:  # After deal
                print(f"Step {step}: CHANCE action={action} ({card_str(action) if action < 52 else action})")
        else:
            bot = bot_p0 if player == 0 else bot_p1
            action = bot.step(state)

            legal = state.legal_actions()

            # Detect knock
            if action == 55:  # knock
                knocker = player
                print(f"\n{'='*60}")
                print(f"Step {step}: P{player} KNOCKS")
                print(f"  Legal actions: {[action_str(a) for a in legal]}")
                print(f"{'='*60}")

            # Detect layoff phase
            if knocker is not None and action != 55:
                if not in_layoff:
                    in_layoff = True
                    print(f"\n--- LAYOFF PHASE (knocker=P{knocker}) ---")

                print(f"Step {step}: P{player} action={action} ({action_str(action)})")
                print(f"  Legal actions: {[action_str(a) for a in legal]}")

                # Show hand
                # Note: We can't easily get hand from pyspiel state, but we can show legal melds
                meld_actions = [a for a in legal if a >= 56]
                if meld_actions:
                    print(f"  Legal melds: {[action_str(a) for a in meld_actions]}")

        state.apply_action(action)
        step += 1

    print(f"\n{'='*60}")
    print(f"GAME OVER")
    print(f"Returns: {state.returns()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Game 325 had a GIN with score mismatch
    print("Tracing game 325 (known GIN with score mismatch)...")
    trace_game(325)

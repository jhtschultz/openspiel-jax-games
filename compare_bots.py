"""Compare JAX simple bot vs C++ pyspiel SimpleGinRummyBot."""

import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import numpy as np
import jax.numpy as jnp
import pyspiel
import gin_rummy_jax as gin

def card_str(idx):
    if idx < 0:
        return "None"
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['s', 'c', 'd', 'h']  # spades=0, clubs=1, diamonds=2, hearts=3 (C++ encoding)
    return ranks[idx % 13] + suits[idx // 13]

def meld_to_cards(meld_id):
    """Convert meld_id to list of card strings."""
    if meld_id < 0 or meld_id >= 185:
        return "INVALID"
    meld = gin.ALL_MELDS[meld_id]
    return [card_str(c) for c in meld]

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
        meld_id = action - 56
        cards = meld_to_cards(meld_id)
        return f"meld_{meld_id} ({','.join(cards)})"

def compare_games(n_games=10, verbose=True):
    """Run games comparing JAX bot vs C++ bot."""

    cpp_game = pyspiel.load_game("gin_rummy")

    total_decisions = 0
    disagreements = 0
    disagreement_details = []
    score_mismatches = 0

    for game_idx in range(n_games):
        # Create fresh bots for each game (they may have internal state)
        cpp_bot_p0 = pyspiel.make_simple_gin_rummy_bot(cpp_game.get_parameters(), 0)
        cpp_bot_p1 = pyspiel.make_simple_gin_rummy_bot(cpp_game.get_parameters(), 1)
        rng = np.random.RandomState(game_idx)
        cpp_state = cpp_game.new_initial_state()
        jax_state = gin.init_state()

        step = 0
        meld_count = {'cpp': 0, 'jax': 0}  # Count meld actions

        while not cpp_state.is_terminal():
            cpp_player = cpp_state.current_player()

            if cpp_state.is_chance_node():
                # Use same random chance action for both
                outcomes = cpp_state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = rng.choice(actions, p=probs)
            else:
                # Get bot actions from both implementations
                cpp_bot = cpp_bot_p0 if cpp_player == 0 else cpp_bot_p1
                cpp_action = cpp_bot.step(cpp_state)
                jax_action = int(gin.simple_bot_action_opt(jax_state))

                # Count meld actions
                if cpp_action >= 56:
                    meld_count['cpp'] += 1
                if jax_action >= 56:
                    meld_count['jax'] += 1

                total_decisions += 1

                if cpp_action != jax_action:
                    disagreements += 1

                    # Get details
                    phase = int(jax_state['phase'])
                    phase_names = ['DEAL', 'FIRST_UPCARD', 'DRAW', 'DISCARD', 'KNOCK', 'LAYOFF', 'WALL', 'GAME_OVER']
                    phase_name = phase_names[phase] if phase < len(phase_names) else f"PHASE_{phase}"

                    hand = gin.get_hand(jax_state, jax_state['current_player'])
                    hand_cards = [card_str(i) for i in range(52) if hand[i] > 0]
                    upcard = int(jax_state['upcard'])

                    detail = {
                        'game': game_idx,
                        'step': step,
                        'phase': phase_name,
                        'player': cpp_player,
                        'cpp_action': cpp_action,
                        'jax_action': jax_action,
                        'hand': hand_cards,
                        'upcard': card_str(upcard),
                    }
                    disagreement_details.append(detail)

                    if verbose:
                        print(f"\nDISAGREEMENT Game {game_idx} Step {step}:")
                        print(f"  Phase: {phase_name}, Player: {cpp_player}")
                        print(f"  C++ bot: {cpp_action} = {action_str(cpp_action)}")
                        print(f"  JAX bot: {jax_action} = {action_str(jax_action)}")
                        print(f"  Hand ({len(hand_cards)} cards): {', '.join(hand_cards)}")
                        print(f"  Upcard: {card_str(upcard)}")

                        # Debug JAX bot decision
                        current_dw = int(gin.calculate_deadwood_lut(hand))
                        hand_with_upcard = hand.at[upcard].set(1) if upcard >= 0 else hand
                        _, up_dws = gin.calculate_deadwood_compressed(hand_with_upcard)
                        min_dw_with_upcard = int(jnp.min(up_dws))
                        print(f"  JAX current_dw: {current_dw}, min_dw_with_upcard: {min_dw_with_upcard}")

                        # For DISCARD/KNOCK phase, show all optimal discards
                        if phase_name in ['DISCARD', 'KNOCK']:
                            hand_indices, hand_dws = gin.calculate_deadwood_compressed(hand)
                            min_dw = int(jnp.min(hand_dws))
                            print(f"  Min deadwood after discard: {min_dw}")
                            print("  Optimal discards:")
                            for i in range(11):
                                idx = int(hand_indices[i])
                                dw = int(hand_dws[i])
                                if dw == min_dw and dw < 999:
                                    val = int(gin.CARD_VALUES[idx])
                                    print(f"    {card_str(idx)} (idx={idx}, val={val}, dw={dw})")

                action = cpp_action  # Use C++ action to keep states in sync

            cpp_state.apply_action(action)
            jax_state = gin.step(jax_state, action)
            step += 1

        # Check final scores match
        cpp_returns = cpp_state.returns()
        jax_p0_score = int(jax_state['p0_score'])
        jax_p1_score = int(jax_state['p1_score'])
        jax_returns = [jax_p0_score, jax_p1_score]

        scores_match = (cpp_returns[0] == jax_returns[0] and cpp_returns[1] == jax_returns[1])
        if not scores_match:
            score_mismatches += 1
            print(f"\n{'='*60}")
            print(f"SCORE MISMATCH Game {game_idx}:")
            print(f"  C++ returns: {cpp_returns}")
            print(f"  JAX returns: {jax_returns}")
            print(f"  Difference: {cpp_returns[0] - jax_returns[0]}")
            print(f"  Total steps: {step}")
            print(f"  Meld counts - C++ would lay: {meld_count['cpp']}, JAX would lay: {meld_count['jax']}")
            print(f"  JAX knocker_deadwood: {int(jax_state['knocker_deadwood'])}")
            print(f"  JAX knocker: {int(jax_state['knocker'])}")
            print(f"  JAX p0_score: {int(jax_state['p0_score'])}, p1_score: {int(jax_state['p1_score'])}")
            # Show final hands
            p0_hand = gin.get_hand(jax_state, 0)
            p1_hand = gin.get_hand(jax_state, 1)
            p0_cards = [card_str(i) for i in range(52) if p0_hand[i] > 0]
            p1_cards = [card_str(i) for i in range(52) if p1_hand[i] > 0]
            print(f"  JAX P0 final hand ({len(p0_cards)}): {', '.join(p0_cards) if p0_cards else 'empty'}")
            print(f"  JAX P1 final hand ({len(p1_cards)}): {', '.join(p1_cards) if p1_cards else 'empty'}")
            print(f"{'='*60}")

        if verbose and game_idx % 1 == 0:
            print(f"Game {game_idx}: {step} steps, {disagreements} total disagreements so far, scores_match={scores_match}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_games} games, {total_decisions} bot decisions")
    print(f"Disagreements: {disagreements} ({100*disagreements/total_decisions:.2f}%)")
    print(f"Score mismatches: {score_mismatches} ({100*score_mismatches/n_games:.2f}%)")

    if score_mismatches == 0:
        print("All game scores matched!")

    if disagreements == 0:
        print("PASSED! JAX bot matches C++ bot exactly.")
    else:
        print(f"FAILED! {disagreements} disagreements found.")

    return disagreements, disagreement_details


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    compare_games(args.n_games, verbose=not args.quiet)

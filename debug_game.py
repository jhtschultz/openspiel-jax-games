"""Debug script to trace a game step by step."""

import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import jax
import jax.numpy as jnp
import gin_rummy_jax as gin

def card_str(idx):
    """Convert card index to string."""
    if idx < 0:
        return "None"
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['s', 'c', 'd', 'h']  # spades=0, clubs=1, diamonds=2, hearts=3 (C++ encoding)
    return ranks[idx % 13] + suits[idx // 13]

def action_str(action):
    """Convert action to string."""
    if action < 52:
        return f"discard {card_str(action)}"
    elif action == 52:
        return "draw_upcard"  # 52 = draw upcard (from pyspiel)
    elif action == 53:
        return "draw_stock"   # 53 = draw stock (from pyspiel)
    elif action == 54:
        return "pass"
    elif action == 55:
        return "knock"
    else:
        return f"meld_{action - 56}"

def hand_str(hand):
    """Convert hand to string."""
    cards = [card_str(i) for i in range(52) if hand[i] > 0]
    return ", ".join(cards)

def replay_game(history, stop_at=None):
    """Replay a game and print each step."""
    state = gin.init_state()

    phase_names = ['DEAL', 'FIRST_UPCARD', 'DRAW', 'DISCARD', 'KNOCK', 'LAYOFF', 'WALL', 'GAME_OVER']

    for step, action in enumerate(history):
        if stop_at is not None and step >= stop_at:
            break

        phase = int(state['phase'])
        player = int(state['current_player'])
        is_chance = (phase == gin.PHASE_DEAL) or bool(state['waiting_stock_draw'])
        upcard = int(state['upcard'])

        phase_name = phase_names[phase] if phase < len(phase_names) else f"PHASE_{phase}"

        # Get hands
        p0_hand = state['player0_hand']
        p1_hand = state['player1_hand']

        # For non-chance, non-deal actions, show more detail
        if not is_chance and phase != gin.PHASE_DEAL:
            hand = p0_hand if player == 0 else p1_hand

            # Check what the bot would do
            bot_action = int(gin.simple_bot_action_opt(state))

            # Calculate deadwood info
            current_dw = int(gin.calculate_deadwood_lut(hand))

            # If draw phase, check upcard logic
            if phase == gin.PHASE_DRAW or phase == gin.PHASE_FIRST_UPCARD:
                hand_with_upcard = hand.at[upcard].set(1) if upcard >= 0 else hand
                _, up_dws = gin.calculate_deadwood_compressed(hand_with_upcard)
                min_dw_with_upcard = int(jnp.min(up_dws))

                # Also check what the bot is actually computing internally
                # Replicate the bot's logic exactly
                hand_indices, hand_dws = gin.calculate_deadwood_compressed(hand)
                bot_min_dw_hand = int(jnp.min(hand_dws))
                bot_current_dw = int(gin.calculate_deadwood_lut(hand))

                upcard_enables_knock = min_dw_with_upcard <= 10
                upcard_reduces_dw = min_dw_with_upcard < bot_current_dw
                take_upcard = upcard_enables_knock or upcard_reduces_dw

                print(f"\n{'='*60}")
                print(f"Step {step}: {phase_name} P{player}")
                print(f"  Action taken: {action} = {action_str(action)}")
                print(f"  Bot would do: {bot_action} = {action_str(bot_action)}")
                print(f"  Hand ({int(jnp.sum(hand))} cards): {hand_str(hand)}")
                print(f"  Upcard: {card_str(upcard)}")
                print(f"  Bot's current_deadwood (LUT): {bot_current_dw}")
                print(f"  Bot's min_dw_with_upcard: {min_dw_with_upcard}")
                print(f"  upcard_enables_knock? {upcard_enables_knock}")
                print(f"  upcard_reduces_dw? {upcard_reduces_dw}")
                print(f"  take_upcard? {take_upcard}")

                if action == 52:  # draw_upcard
                    print(f"  >>> TAKING UPCARD: {card_str(upcard)}")
            elif phase == gin.PHASE_DISCARD:
                print(f"\n{'='*60}")
                print(f"Step {step}: {phase_name} P{player}")
                print(f"  Action taken: {action} = {action_str(action)}")
                print(f"  Bot would do: {bot_action} = {action_str(bot_action)}")
                print(f"  Hand ({int(jnp.sum(hand))} cards): {hand_str(hand)}")
                print(f"  Current deadwood: {current_dw}")
        else:
            # Just show brief info for chance/deal
            if step < 21:
                pass  # Skip deal
            else:
                print(f"Step {step}: {phase_name} P{player} chance={is_chance} action={action}")

        state = gin.step(state, action)

    print(f"\n{'='*60}")
    print("GAME OVER")
    print(f"P0 score: {int(state['p0_score'])}")
    print(f"P1 score: {int(state['p1_score'])}")


if __name__ == "__main__":
    # The problematic game history
    history = [14, 1, 6, 48, 23, 31, 45, 27, 22, 37, 41, 24, 2, 51, 35, 26, 44, 7, 29, 30, 12, 54, 52, 7, 52, 22, 53, 3, 12, 53, 13, 23, 53, 17, 51, 53, 11, 11, 53, 36, 24, 53, 4, 37, 53, 15, 35, 53, 49, 48, 53, 21, 36, 53, 0, 49, 53, 42, 21, 53, 34, 34, 53, 38, 38, 53, 25, 25, 53, 19, 19, 52, 7, 53, 8, 8, 53, 20, 20, 53, 16, 44, 53, 5, 5, 53, 18, 55, 26, 72, 68, 135, 54, 54, 88, 64, 54]

    print("Replaying problematic game...")
    print("Looking for 8s (card index 46) being picked up...")
    print()

    # Find where 8s might be picked up
    # 8s = 7 (rank 8) + 39 (spades) = 46
    # But user said "8s" - let me check if they mean something else
    # Actually in the history I see action 53 (draw_upcard) multiple times

    replay_game(history)

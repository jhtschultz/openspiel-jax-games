"""Debug the simple_bot_action_opt function directly."""

import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import jax
import jax.numpy as jnp
import gin_rummy_jax as gin

def card_str(idx):
    if idx < 0:
        return "None"
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['c', 'd', 'h', 's']
    return ranks[idx % 13] + suits[idx // 13]

def debug_bot_action(state):
    """Copy of simple_bot_action_opt with debug prints."""
    phase = state['phase']
    player = state['current_player']
    hand = gin.get_hand(state, player)
    upcard = state['upcard']
    deck = state['deck']

    print(f"  Phase: {int(phase)}, Player: {int(player)}")
    print(f"  Upcard: {int(upcard)} = {card_str(int(upcard))}")
    print(f"  Hand size: {int(jnp.sum(hand))}")

    # 1. Compressed deadwood for current hand
    hand_indices, hand_dws = gin.calculate_deadwood_compressed(hand)
    min_dw_hand = jnp.min(hand_dws)
    print(f"  min_dw_hand (10-card after discard): {int(min_dw_hand)}")

    # 2. Compressed deadwood with upcard
    hand_with_upcard = jnp.where(upcard >= 0, hand.at[upcard].set(1), hand)
    _, up_dws = gin.calculate_deadwood_compressed(hand_with_upcard)
    min_dw_with_upcard = jnp.min(up_dws)
    print(f"  min_dw_with_upcard: {int(min_dw_with_upcard)}")

    # Current hand deadwood (10 cards)
    current_deadwood = gin.calculate_deadwood_lut(hand)
    print(f"  current_deadwood (LUT): {int(current_deadwood)}")

    # Draw phase evaluation
    upcard_enables_knock = min_dw_with_upcard <= gin.KNOCK_THRESHOLD
    upcard_reduces_deadwood = min_dw_with_upcard < current_deadwood
    take_upcard = upcard_enables_knock | upcard_reduces_deadwood

    print(f"  upcard_enables_knock: {bool(upcard_enables_knock)}")
    print(f"  upcard_reduces_deadwood: {bool(upcard_reduces_deadwood)}")
    print(f"  take_upcard: {bool(take_upcard)}")

    # Compute draw_action manually
    is_wall = jnp.sum(deck) <= gin.WALL_STOCK_SIZE

    print(f"  is_wall: {bool(is_wall)}")
    print(f"  take_upcard type: {type(take_upcard)}, value: {take_upcard}")
    print(f"  upcard type: {type(upcard)}, value: {upcard}")
    print(f"  upcard >= 0: {upcard >= 0}")
    print(f"  take_upcard & (upcard >= 0): {take_upcard & (upcard >= 0)}")

    condition = take_upcard & (upcard >= 0)
    print(f"  condition type: {type(condition)}, value: {condition}")

    draw_action = jnp.where(condition, gin.ACTION_DRAW_UPCARD, gin.ACTION_DRAW_STOCK)
    print(f"  draw_action after first where: {int(draw_action)}")

    draw_action = jnp.where(is_wall & ~take_upcard, gin.ACTION_PASS, draw_action)
    print(f"  Computed draw_action: {int(draw_action)}")

    # What the bot actually returns
    actual_action = gin.simple_bot_action_opt(state)
    print(f"  Actual bot action: {int(actual_action)}")

    # Clear cache and try again?
    print(f"\n  --- Reimporting module ---")
    import importlib
    importlib.reload(gin)
    actual_action2 = gin.simple_bot_action_opt(state)
    print(f"  After reload bot action: {int(actual_action2)}")

    return int(actual_action)


# Replay to step 26
history = [14, 1, 6, 48, 23, 31, 45, 27, 22, 37, 41, 24, 2, 51, 35, 26, 44, 7, 29, 30, 12, 54, 52, 7, 52, 22, 53, 3, 12, 53, 13, 23, 53, 17, 51, 53, 11, 11, 53, 36, 24, 53, 4, 37, 53, 15, 35, 53, 49, 48, 53, 21, 36, 53, 0, 49, 53, 42, 21, 53, 34, 34, 53, 38, 38, 53, 25, 25, 53, 19, 19, 52, 7, 53, 8, 8, 53, 20, 20, 53, 16, 44, 53, 5, 5, 53, 18, 55, 26, 72, 68, 135, 54, 54, 88, 64, 54]

state = gin.init_state()
for step, action in enumerate(history[:26]):
    state = gin.step(state, action)

print(f"\n=== At step 26 (before action) ===")
debug_bot_action(state)

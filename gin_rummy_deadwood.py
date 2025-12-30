"""Deadwood calculation functions for Gin Rummy.

This module contains the core deadwood calculation functions shared between
gin_rummy_core.py and gin_rummy_jax.py.

Deadwood is the total point value of cards not in any meld. The goal in
Gin Rummy is to minimize deadwood to enable knocking or going gin.

Key functions:
- hand_to_4x13: Convert 52-card mask to 4×13 suit×rank matrix
- calculate_deadwood_lut: Calculate deadwood for a 10-card hand
- calculate_deadwood_all_discards_exact: Calculate deadwood for each possible discard
"""

import jax
import jax.numpy as jnp

from gin_rummy_luts import (
    RANK_VALUES, RANK_VALUES_16,
    POWERS_OF_2, POWERS_OF_2_16,
    SUIT_POWERS, SUIT_POWERS_8,
    NIBBLE_POPCOUNTS, NIBBLE_POPCOUNTS_8,
    NIBBLE_TO_SUITS, NIBBLE_TO_SUITS_8,
    RUN_SCORE_LUT,
    SUBSET_TABLE,
    RUN_MEMBER_LUT, RUN_DECOMP_LUT,
    VALID_SET_MODES_8, VALID_SET_MODE_COUNTS_8,
    COMBO_IDX_0_8, COMBO_IDX_1_8,
    SCENARIO_IDX_A, SCENARIO_IDX_B, SCENARIO_IDX_C,
)


# =============================================================================
# Hand Representation
# =============================================================================

@jax.jit
def hand_to_4x13(hand):
    """Convert 52-card binary mask to (4, 13) suit×rank matrix.

    Args:
        hand: (52,) int8 binary mask of cards held

    Returns:
        (4, 13) int8 matrix where [suit, rank] = 1 if card held
    """
    return hand.reshape(4, 13)


# =============================================================================
# Core Deadwood Calculation (216-scenario exact algorithm)
# =============================================================================

@jax.jit
def _compute_optimal_meld_info(hand_4x13):
    """Compute optimal meld decomposition for a hand.

    Uses 216-scenario enumeration (3 sets × 6 options each) to find
    the optimal meld configuration that minimizes deadwood.

    Args:
        hand_4x13: (4, 13) suit×rank matrix

    Returns:
        max_meld_points: Total points from optimal melds
        optimal_meld_mask: (185,) boolean mask of melds in optimal decomposition
    """
    rank_nibbles = jnp.dot(SUIT_POWERS, hand_4x13.astype(jnp.int32))  # (13,)
    rank_counts = NIBBLE_POPCOUNTS[rank_nibbles]

    is_candidate = rank_counts >= 3
    set_indices = jnp.argsort(~is_candidate)[:3]  # Top 3 ranks
    set_nibbles = rank_nibbles[set_indices]  # (3,) held cards per candidate

    # Get options per candidate (6 options each: skip, 4 3-card subsets, 4-card)
    options = SUBSET_TABLE[set_nibbles]  # (3, 6)

    # Build 216 scenarios using precomputed indices
    mask_A = options[0, SCENARIO_IDX_A]  # (216,)
    mask_B = options[1, SCENARIO_IDX_B]  # (216,)
    mask_C = options[2, SCENARIO_IDX_C]  # (216,)

    # Set points per scenario
    pts_A = NIBBLE_POPCOUNTS[mask_A].astype(jnp.int16) * RANK_VALUES_16[set_indices[0]]
    pts_B = NIBBLE_POPCOUNTS[mask_B].astype(jnp.int16) * RANK_VALUES_16[set_indices[1]]
    pts_C = NIBBLE_POPCOUNTS[mask_C].astype(jnp.int16) * RANK_VALUES_16[set_indices[2]]
    scenario_set_pts = pts_A + pts_B + pts_C  # (216,)

    # Build removal masks for runs
    def to_suit_removal(mask, rank_idx):
        bits = jnp.stack([((mask >> i) & 1) for i in range(4)], axis=-1).astype(jnp.int16)
        return bits * (jnp.int16(1) << rank_idx)

    rem_A = to_suit_removal(mask_A, set_indices[0])  # (216, 4)
    rem_B = to_suit_removal(mask_B, set_indices[1])
    rem_C = to_suit_removal(mask_C, set_indices[2])
    total_removal = rem_A | rem_B | rem_C  # (216, 4)

    # Base suit bitmasks
    base_suits = jnp.dot(hand_4x13.astype(jnp.int16), POWERS_OF_2_16)  # (4,)

    # Apply removal to get scenario suit configs
    scenario_suits = base_suits[None, :] & (~total_removal)  # (216, 4)

    # Lookup run scores for each scenario
    run_scores = RUN_SCORE_LUT[scenario_suits]  # (216, 4)
    run_total = jnp.sum(run_scores, axis=1)  # (216,)

    # Total meld points per scenario
    total_meld_pts = scenario_set_pts + run_total  # (216,)

    # Find best scenario with tie-breaker: prefer sets over runs (C++ behavior)
    set_card_count = (NIBBLE_POPCOUNTS[mask_A] + NIBBLE_POPCOUNTS[mask_B] +
                      NIBBLE_POPCOUNTS[mask_C]).astype(jnp.int32)
    score_with_tiebreak = total_meld_pts.astype(jnp.int32) * 100 + set_card_count
    best_idx = jnp.argmax(score_with_tiebreak)
    max_meld_points = total_meld_pts[best_idx]

    # Extract specific meld IDs for best scenario
    best_mask_A = mask_A[best_idx]
    best_mask_B = mask_B[best_idx]
    best_mask_C = mask_C[best_idx]

    def get_set_meld_id(nibble_mask, rank):
        popcount = NIBBLE_POPCOUNTS[nibble_mask]
        missing_suit = jnp.where(nibble_mask & 1,
                                 jnp.where(nibble_mask & 2,
                                          jnp.where(nibble_mask & 4, 3, 2), 1), 0)
        meld_id_3 = rank * 5 + missing_suit
        meld_id_4 = rank * 5 + 4
        return jnp.where(popcount >= 4, meld_id_4,
                        jnp.where(popcount >= 3, meld_id_3, jnp.int32(-1)))

    set_meld_A = get_set_meld_id(best_mask_A, set_indices[0])
    set_meld_B = get_set_meld_id(best_mask_B, set_indices[1])
    set_meld_C = get_set_meld_id(best_mask_C, set_indices[2])

    # Build optimal meld mask
    optimal_meld_mask = jnp.zeros(185, dtype=jnp.bool_)
    optimal_meld_mask = jnp.where(set_meld_A >= 0,
                                   optimal_meld_mask.at[set_meld_A].set(True),
                                   optimal_meld_mask)
    optimal_meld_mask = jnp.where(set_meld_B >= 0,
                                   optimal_meld_mask.at[set_meld_B].set(True),
                                   optimal_meld_mask)
    optimal_meld_mask = jnp.where(set_meld_C >= 0,
                                   optimal_meld_mask.at[set_meld_C].set(True),
                                   optimal_meld_mask)

    # Mark run melds from RUN_DECOMP_LUT
    best_scenario_suits = scenario_suits[best_idx]
    for suit in range(4):
        suit_mask = best_scenario_suits[suit]
        run_meld_ids = RUN_DECOMP_LUT[suit_mask]
        for j in range(5):
            base_meld_id = run_meld_ids[j]
            is_3card = (base_meld_id >= 65) & (base_meld_id < 109)
            is_4card = (base_meld_id >= 109) & (base_meld_id < 149)
            is_5card = (base_meld_id >= 149) & (base_meld_id < 185)
            suit_offset = jnp.where(is_3card, suit * 11,
                                   jnp.where(is_4card, suit * 10,
                                            jnp.where(is_5card, suit * 9, 0)))
            meld_id = base_meld_id + suit_offset
            is_valid = base_meld_id >= 0
            optimal_meld_mask = jnp.where(is_valid,
                                          optimal_meld_mask.at[meld_id].set(True),
                                          optimal_meld_mask)

    return max_meld_points, optimal_meld_mask


@jax.jit
def calculate_deadwood_lut(hand):
    """Calculate exact deadwood using LUT + 3-set enumeration.

    This is the primary deadwood calculation function, using the 216-scenario
    algorithm that handles up to 3 sets with 4-card splitting.

    Args:
        hand: (52,) int8 binary mask of cards held (10 cards)

    Returns:
        Deadwood value (total points of unmelded cards)
    """
    hand_4x13 = hand_to_4x13(hand)
    max_meld_points, _ = _compute_optimal_meld_info(hand_4x13)
    total_hand_value = jnp.sum(hand_4x13.astype(jnp.int32) * RANK_VALUES[None, :])
    return total_hand_value - max_meld_points


@jax.jit
def calculate_deadwood_with_melds(hand):
    """Calculate deadwood and return optimal meld mask.

    Args:
        hand: (52,) int8 binary mask of cards held

    Returns:
        deadwood: Deadwood value
        optimal_meld_mask: (185,) boolean mask of melds in optimal decomposition
    """
    hand_4x13 = hand_to_4x13(hand)
    max_meld_points, optimal_meld_mask = _compute_optimal_meld_info(hand_4x13)
    total_hand_value = jnp.sum(hand_4x13.astype(jnp.int32) * RANK_VALUES[None, :])
    return total_hand_value - max_meld_points, optimal_meld_mask


# =============================================================================
# All-Discards Calculation (for 11-card hands)
# =============================================================================

@jax.jit
def calculate_deadwood_all_discards_exact(hand):
    """Calculate deadwood for each possible discard from an 11-card hand.

    Uses full 216-scenario enumeration for each discard to ensure
    100% correctness.

    Args:
        hand: (52,) int8 binary mask of 11 cards

    Returns:
        (52,) int16 array of deadwood values for discarding each card.
        Cards not in hand get value 999.
    """
    hand_4x13 = hand_to_4x13(hand)
    total_hand_value = jnp.sum(hand_4x13.astype(jnp.int32) * RANK_VALUES[None, :])

    def deadwood_after_discard(card_idx):
        suit = card_idx // 13
        rank = card_idx % 13
        card_present = hand[card_idx]

        # Create hand with card removed
        new_hand = hand.at[card_idx].set(0)
        new_hand_4x13 = hand_to_4x13(new_hand)

        # Compute optimal meld points for new hand
        max_meld_points, _ = _compute_optimal_meld_info(new_hand_4x13)

        new_hand_value = total_hand_value - RANK_VALUES[rank] * card_present
        dw = new_hand_value - max_meld_points

        return jnp.where(card_present > 0, dw.astype(jnp.int16), jnp.int16(999))

    return jax.vmap(deadwood_after_discard)(jnp.arange(52))


@jax.jit
def calculate_deadwood_compressed(hand):
    """Calculate deadwood for all discards with compressed output.

    Returns the held card indices and their corresponding deadwood values,
    sorted by deadwood (ascending).

    Args:
        hand: (52,) int8 binary mask of 11 cards

    Returns:
        hand_indices: (11,) int32 indices of held cards, sorted by deadwood
        hand_deadwoods: (11,) int16 deadwood values, sorted ascending
    """
    all_dw = calculate_deadwood_all_discards_exact(hand)

    # Get indices of held cards
    held_mask = hand > 0
    held_indices = jnp.where(held_mask, jnp.arange(52), 999)

    # Sort by deadwood, keeping track of indices
    sort_order = jnp.argsort(all_dw)
    sorted_indices = sort_order[:11]
    sorted_dw = all_dw[sorted_indices]

    return sorted_indices.astype(jnp.int32), sorted_dw.astype(jnp.int16)

"""JAX Gin Rummy core - standalone version without pyspiel dependency."""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

# =============================================================================
# Constants (matching C++)
# =============================================================================
NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4
HAND_SIZE = 10
KNOCK_THRESHOLD = 10
GIN_BONUS = 25
UNDERCUT_BONUS = 25

CARD_POINTS = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=jnp.int8)

PHASE_DEAL = 0
PHASE_FIRST_UPCARD = 1
PHASE_DRAW = 2
PHASE_DISCARD = 3
PHASE_KNOCK = 4
PHASE_LAYOFF = 5
PHASE_WALL = 6
PHASE_GAME_OVER = 7

MAX_NUM_DRAW_UPCARD_ACTIONS = 50
WALL_STOCK_SIZE = 2

ACTION_DRAW_UPCARD = 52
ACTION_DRAW_STOCK = 53
ACTION_PASS = 54
ACTION_KNOCK = 55
ACTION_MELD_BASE = 56

NUM_MELDS = 185
NUM_ACTIONS = 241

CHANCE_PLAYER = -1
TERMINAL_PLAYER = -4

# =============================================================================
# LUT-based deadwood calculation (exact & fast)
# =============================================================================

def _build_run_lookup_table():
    """Build lookup table: suit bitmask (13 bits) -> max points from runs.

    Uses DP: for each position, either skip or start a run of length 3/4/5/...
    """
    card_values = [min(i + 1, 10) for i in range(13)]  # A=1, 2-9, 10/J/Q/K=10
    lut = []

    for bits in range(8192):  # 2^13 configurations
        held = [(bits >> i) & 1 for i in range(13)]

        # DP: dp[i] = max points using cards from position i onwards
        # For each position, we can skip it or start a run of length 3, 4, 5...
        dp = [0] * 14  # dp[13] = 0 (base case)

        for i in range(12, -1, -1):  # Process right to left
            # Option 1: skip this position
            best = dp[i + 1]

            # Option 2: start a run at position i (if possible)
            for length in range(3, 14 - i):  # runs of length 3, 4, 5...
                # Check if all cards in run are held
                if all(held[i + k] for k in range(length)):
                    run_points = sum(card_values[i + k] for k in range(length))
                    # After using this run, next available position is i + length
                    best = max(best, run_points + dp[i + length])
                else:
                    break  # Can't extend further if a card is missing

            dp[i] = best

        lut.append(dp[0])

    return jnp.array(lut, dtype=jnp.int16)

# Precomputed tables
RUN_SCORE_LUT = _build_run_lookup_table()
RANK_VALUES = jnp.array([min(i + 1, 10) for i in range(13)], dtype=jnp.int32)
POWERS_OF_2 = 2 ** jnp.arange(13, dtype=jnp.int32)
SUIT_POWERS = jnp.array([1, 2, 4, 8], dtype=jnp.int32)

# All 16 possible 4-bit masks for set subsets
ALL_NIBBLES = jnp.arange(16, dtype=jnp.int32)
NIBBLE_POPCOUNTS = jnp.array([bin(i).count('1') for i in range(16)], dtype=jnp.int32)

# Precompute meshgrid for 256 set combinations
_idx_grid = jnp.arange(16)
_grid_A, _grid_B = jnp.meshgrid(_idx_grid, _idx_grid)
GRID_A = _grid_A.flatten()  # (256,)
GRID_B = _grid_B.flatten()  # (256,)
MASK_A_ALL = ALL_NIBBLES[GRID_A]  # (256,)
MASK_B_ALL = ALL_NIBBLES[GRID_B]  # (256,)
POPCNT_A = NIBBLE_POPCOUNTS[GRID_A]  # (256,)
POPCNT_B = NIBBLE_POPCOUNTS[GRID_B]  # (256,)

# Precompute suit expansion for all nibble values (16, 4)
NIBBLE_TO_SUITS = jnp.stack([((jnp.arange(16) >> i) & 1) for i in range(4)], axis=-1)

# =============================================================================
# Valid-only set combinations (reduces 256 → 36 max)
# =============================================================================
# For each nibble (0-15), enumerate valid set modes: 0, or subset of size 3+
def _build_valid_set_modes():
    """For each nibble, list valid set subsets (size 0, 3, or 4)."""
    # valid_modes[nibble] = list of valid subset nibbles
    # Max 6 modes per nibble: 0 + C(4,3)=4 three-subsets + 1 four-subset
    MAX_MODES = 6
    modes = np.zeros((16, MAX_MODES), dtype=np.int32)
    mode_counts = np.zeros(16, dtype=np.int32)

    for nibble in range(16):
        valid = [0]  # Always can skip
        popcount = bin(nibble).count('1')
        if popcount >= 3:
            # Add all 3-card subsets
            for mask in range(16):
                if bin(mask).count('1') == 3 and (nibble & mask) == mask:
                    valid.append(mask)
            # Add 4-card subset if exactly 4
            if popcount == 4:
                valid.append(nibble)
        modes[nibble, :len(valid)] = valid
        mode_counts[nibble] = len(valid)

    return jnp.array(modes), jnp.array(mode_counts)

VALID_SET_MODES, VALID_SET_MODE_COUNTS = _build_valid_set_modes()  # (16, 6), (16,)
VALID_SET_MODES_8 = VALID_SET_MODES.astype(jnp.int8)
VALID_SET_MODE_COUNTS_8 = VALID_SET_MODE_COUNTS.astype(jnp.int8)

# Precompute all 36 valid combo pairs (max 6 modes × 6 modes)
# We'll mask out invalid combos at runtime based on actual held cards
_idx6 = np.arange(6)
_g0, _g1 = np.meshgrid(_idx6, _idx6)
COMBO_IDX_0 = jnp.array(_g0.flatten())  # (36,)
COMBO_IDX_1 = jnp.array(_g1.flatten())  # (36,)
COMBO_IDX_0_8 = COMBO_IDX_0.astype(jnp.int8)
COMBO_IDX_1_8 = COMBO_IDX_1.astype(jnp.int8)

# Int8/int16 versions of constants for memory-optimized path
RANK_VALUES_16 = RANK_VALUES.astype(jnp.int16)
POWERS_OF_2_16 = POWERS_OF_2.astype(jnp.int16)
SUIT_POWERS_8 = SUIT_POWERS.astype(jnp.int8)
NIBBLE_POPCOUNTS_8 = NIBBLE_POPCOUNTS.astype(jnp.int8)
NIBBLE_TO_SUITS_8 = NIBBLE_TO_SUITS.astype(jnp.int8)

@jax.jit
def hand_to_4x13(hand):
    """Convert 52-card binary mask to (4, 13) suit×rank matrix."""
    return hand.reshape(4, 13)

@jax.jit
def _compute_max_meld_points(rank_nibbles, base_suits):
    """Core meld computation shared by single and all-discards versions."""
    rank_counts = NIBBLE_POPCOUNTS[rank_nibbles]
    is_set_candidate = rank_counts >= 3

    set_indices = jnp.argsort(~is_set_candidate)[:2]
    held_masks = rank_nibbles[set_indices]

    # Valid set modes: (2, 16)
    subset_check = (held_masks[:, None] & ALL_NIBBLES[None, :]) == ALL_NIBBLES[None, :]
    valid_size = (NIBBLE_POPCOUNTS[None, :] >= 3) | (ALL_NIBBLES[None, :] == 0)
    valid_set_modes = subset_check & valid_size

    # Check combo validity using precomputed grids
    combo_valid = valid_set_modes[0, GRID_A] & valid_set_modes[1, GRID_B]

    # Points from sets
    val_A = RANK_VALUES[set_indices[0]]
    val_B = RANK_VALUES[set_indices[1]]
    points_sets = (POPCNT_A * val_A) + (POPCNT_B * val_B)

    # Build removal masks using precomputed suit expansions
    shift_A = set_indices[0]
    shift_B = set_indices[1]
    suits_A = NIBBLE_TO_SUITS[GRID_A]  # (256, 4)
    suits_B = NIBBLE_TO_SUITS[GRID_B]  # (256, 4)
    remove_masks = (suits_A << shift_A) | (suits_B << shift_B)

    # Apply removal and lookup
    final_suits = base_suits[None, :] & (~remove_masks)
    run_scores = RUN_SCORE_LUT[final_suits]
    total_run_points = jnp.sum(run_scores, axis=1)

    # Total meld points
    total_meld_points = points_sets + total_run_points
    valid_meld_points = jnp.where(combo_valid, total_meld_points, -1)
    return jnp.max(valid_meld_points)

@jax.jit
def calculate_deadwood_lut(hand):
    """Calculate exact deadwood using LUT + set enumeration."""
    hand_4x13 = hand_to_4x13(hand)
    rank_nibbles = jnp.dot(SUIT_POWERS, hand_4x13.astype(jnp.int32))
    base_suits = jnp.dot(hand_4x13.astype(jnp.int32), POWERS_OF_2)

    max_meld_points = _compute_max_meld_points(rank_nibbles, base_suits)
    total_hand_value = jnp.sum(hand_4x13.astype(jnp.int32) * RANK_VALUES[None, :])
    return total_hand_value - max_meld_points

@jax.jit
def calculate_deadwood_all_discards_lut(hand):
    """Calculate deadwood for each possible discard using LUT approach (exact).

    Memory-optimized: freezes set_indices from 11-card hand to avoid resorting per discard.
    """
    hand_4x13 = hand_to_4x13(hand)
    total_hand_value = jnp.sum(hand_4x13.astype(jnp.int32) * RANK_VALUES[None, :])
    base_suits = jnp.dot(hand_4x13.astype(jnp.int32), POWERS_OF_2)

    # Freeze set candidates from original hand (removing 1 card rarely changes top-2)
    rank_nibbles_orig = jnp.dot(SUIT_POWERS, hand_4x13.astype(jnp.int32))
    rank_counts_orig = NIBBLE_POPCOUNTS[rank_nibbles_orig]
    set_indices = jnp.argsort(~(rank_counts_orig >= 3))[:2]  # Top 2 set candidate ranks

    def deadwood_after_discard(card_idx):
        suit = card_idx // 13
        rank = card_idx % 13
        card_present = hand[card_idx]

        # Update suit bits for this discard
        suit_mask = (jnp.arange(4) == suit).astype(jnp.int32) * (1 << rank) * card_present
        new_suit_bits = base_suits ^ suit_mask

        # Recompute nibbles only for the 2 set candidate ranks (not all 13)
        def get_nibble_for_rank(r):
            return ((new_suit_bits >> r) & 1).dot(SUIT_POWERS)

        nib_0 = get_nibble_for_rank(set_indices[0])
        nib_1 = get_nibble_for_rank(set_indices[1])
        held_masks = jnp.array([nib_0, nib_1])

        # Check valid set modes
        subset_check = (held_masks[:, None] & ALL_NIBBLES[None, :]) == ALL_NIBBLES[None, :]
        valid_size = (NIBBLE_POPCOUNTS[None, :] >= 3) | (ALL_NIBBLES[None, :] == 0)
        valid_set_modes = subset_check & valid_size

        combo_valid = valid_set_modes[0, GRID_A] & valid_set_modes[1, GRID_B]

        # Points from sets
        val_A = RANK_VALUES[set_indices[0]]
        val_B = RANK_VALUES[set_indices[1]]
        points_sets = (POPCNT_A * val_A) + (POPCNT_B * val_B)

        # Build removal masks
        suits_A = NIBBLE_TO_SUITS[GRID_A]
        suits_B = NIBBLE_TO_SUITS[GRID_B]
        remove_masks = (suits_A << set_indices[0]) | (suits_B << set_indices[1])

        # Apply removal and lookup
        final_suits = new_suit_bits[None, :] & (~remove_masks)
        run_scores = RUN_SCORE_LUT[final_suits]
        total_run_points = jnp.sum(run_scores, axis=1)

        total_meld_points = points_sets + total_run_points
        valid_meld_points = jnp.where(combo_valid, total_meld_points, -1)
        max_meld_points = jnp.max(valid_meld_points)

        new_hand_value = total_hand_value - RANK_VALUES[rank] * card_present
        dw = new_hand_value - max_meld_points

        return jnp.where(card_present > 0, dw, jnp.int32(999))

    return jax.vmap(deadwood_after_discard)(jnp.arange(52))

@jax.jit
def calculate_deadwood_all_discards_opt(hand):
    """Optimized: uses 36 valid combos instead of 256, with int16 intermediates.

    Memory reduction: 256 → 36 = 7x, int32 → int16 = 2x → ~14x total.
    """
    hand_4x13 = hand_to_4x13(hand)
    total_hand_value = jnp.sum(hand_4x13.astype(jnp.int16) * RANK_VALUES_16[None, :])
    base_suits = jnp.dot(hand_4x13.astype(jnp.int16), POWERS_OF_2_16)

    # Freeze set candidates from original hand
    rank_nibbles_orig = jnp.dot(SUIT_POWERS_8, hand_4x13.astype(jnp.int8))
    rank_counts_orig = NIBBLE_POPCOUNTS_8[rank_nibbles_orig]
    set_indices = jnp.argsort(~(rank_counts_orig >= 3))[:2]

    def deadwood_after_discard(card_idx):
        suit = card_idx // 13
        rank = card_idx % 13
        card_present = hand[card_idx]

        # Update suit bits for this discard (int16)
        suit_mask = (jnp.arange(4) == suit).astype(jnp.int16) * (1 << rank) * card_present
        new_suit_bits = base_suits ^ suit_mask

        # Recompute nibbles for the 2 set candidate ranks (int8)
        def get_nibble_for_rank(r):
            return ((new_suit_bits >> r) & 1).astype(jnp.int8).dot(SUIT_POWERS_8)

        nib_0 = get_nibble_for_rank(set_indices[0])
        nib_1 = get_nibble_for_rank(set_indices[1])

        # Get valid modes for each nibble (max 6 each)
        modes_0 = VALID_SET_MODES_8[nib_0]  # (6,) int8
        modes_1 = VALID_SET_MODES_8[nib_1]  # (6,) int8
        count_0 = VALID_SET_MODE_COUNTS_8[nib_0]
        count_1 = VALID_SET_MODE_COUNTS_8[nib_1]

        # Create 36 combinations from the valid modes
        mask_A = modes_0[COMBO_IDX_0_8]  # (36,) int8
        mask_B = modes_1[COMBO_IDX_1_8]  # (36,) int8

        # Validity check
        combo_valid = (COMBO_IDX_0_8 < count_0) & (COMBO_IDX_1_8 < count_1)

        # Points from sets (int16)
        val_A = RANK_VALUES_16[set_indices[0]]
        val_B = RANK_VALUES_16[set_indices[1]]
        points_sets = (NIBBLE_POPCOUNTS_8[mask_A].astype(jnp.int16) * val_A +
                       NIBBLE_POPCOUNTS_8[mask_B].astype(jnp.int16) * val_B)

        # Build removal masks (int16 for bit shifts)
        suits_A = NIBBLE_TO_SUITS_8[mask_A]  # (36, 4) int8
        suits_B = NIBBLE_TO_SUITS_8[mask_B]  # (36, 4) int8
        remove_masks = (suits_A.astype(jnp.int16) << set_indices[0]) | (suits_B.astype(jnp.int16) << set_indices[1])

        # Apply removal and lookup
        final_suits = new_suit_bits[None, :] & (~remove_masks)  # (36, 4) int16
        run_scores = RUN_SCORE_LUT[final_suits]  # (36, 4) int16
        total_run_points = jnp.sum(run_scores, axis=1)  # (36,) int16

        total_meld_points = points_sets + total_run_points
        valid_meld_points = jnp.where(combo_valid, total_meld_points, jnp.int16(-1))
        max_meld_points = jnp.max(valid_meld_points)

        new_hand_value = total_hand_value - RANK_VALUES_16[rank] * card_present
        dw = new_hand_value - max_meld_points

        return jnp.where(card_present > 0, dw, jnp.int16(999))

    return jax.vmap(deadwood_after_discard)(jnp.arange(52))

# =============================================================================
# Full 216-scenario exact deadwood (handles 3 sets + 4-card splitting)
# =============================================================================

def _build_subset_table():
    """Build (16, 6) table mapping held nibble -> valid subset options.

    For each nibble (0-15), list valid subsets:
    - 0 (don't use as set)
    - Any subset with popcount >= 3 that fits within held cards

    Examples:
    - nibble=7 (0111, 3 cards): options = [0, 7]
    - nibble=15 (1111, 4 cards): options = [0, 7, 11, 13, 14, 15]
    """
    import numpy as np
    table = np.zeros((16, 6), dtype=np.int32)

    for nibble in range(16):
        options = [0]  # Always can skip
        popcount = bin(nibble).count('1')

        if popcount >= 3:
            # Find all valid subsets (size 3 or 4)
            for subset in range(1, 16):
                subset_pop = bin(subset).count('1')
                # Subset must fit in nibble and have 3+ cards
                if (nibble & subset) == subset and subset_pop >= 3:
                    options.append(subset)

        # Pad to length 6
        while len(options) < 6:
            options.append(0)
        table[nibble] = options[:6]

    return jnp.array(table, dtype=jnp.int32)

SUBSET_TABLE = _build_subset_table()  # (16, 6)

def _build_run_member_table():
    """Build (8192,) table: which ranks are part of a run for each suit config.

    Value is a 13-bit mask indicating which ranks participate in runs.
    """
    import numpy as np
    lut = []
    for bits in range(8192):
        held = [(bits >> i) & 1 for i in range(13)]
        is_member = [0] * 13

        # Check for runs of 3+
        for i in range(11):  # Start positions 0..10
            if held[i] and held[i+1] and held[i+2]:
                is_member[i] = 1
                is_member[i+1] = 1
                is_member[i+2] = 1
                # Extend for lengths 4, 5, ...
                k = 3
                while i + k < 13 and held[i + k]:
                    is_member[i + k] = 1
                    k += 1

        mask = 0
        for i, m in enumerate(is_member):
            if m:
                mask |= (1 << i)
        lut.append(mask)

    return jnp.array(lut, dtype=jnp.int16)

RUN_MEMBER_LUT = _build_run_member_table()  # (8192,)

# Int8 version of RUN_SCORE_LUT for memory bandwidth
RUN_SCORE_LUT_8 = RUN_SCORE_LUT.astype(jnp.int8)

# Precompute the 216 scenario indices
_idx6 = jnp.arange(6)
_grid_A, _grid_B, _grid_C = jnp.meshgrid(_idx6, _idx6, _idx6, indexing='ij')
SCENARIO_IDX_A = _grid_A.flatten()  # (216,)
SCENARIO_IDX_B = _grid_B.flatten()  # (216,)
SCENARIO_IDX_C = _grid_C.flatten()  # (216,)

@jax.jit
def calculate_deadwood_all_discards_exact(hand):
    """Exact deadwood with full 216-scenario enumeration.

    Handles:
    - Up to 3 set candidates
    - 4-card set splitting (use 3, leave 1 for runs)
    - Proper validity checking for discards

    Complexity: ~3,744 ops per hand (216 scenarios × ~17 ops)
    """
    hand_4x13 = hand_to_4x13(hand)

    # --- 1. IDENTIFY TOP 3 SET CANDIDATES ---
    rank_nibbles = jnp.dot(SUIT_POWERS, hand_4x13.astype(jnp.int32))  # (13,)
    rank_counts = NIBBLE_POPCOUNTS[rank_nibbles]

    # Sort by "is valid set candidate" (3+ cards), then by count
    is_candidate = rank_counts >= 3
    set_indices = jnp.argsort(~is_candidate)[:3]  # Top 3 ranks
    set_nibbles = rank_nibbles[set_indices]  # (3,) held cards per candidate

    # --- 2. GET OPTIONS PER CANDIDATE ---
    # options[i] = valid subset masks for candidate i
    options = SUBSET_TABLE[set_nibbles]  # (3, 6)

    # --- 3. BUILD 216 SCENARIOS ---
    # Look up actual masks for each scenario
    mask_A = options[0, SCENARIO_IDX_A]  # (216,)
    mask_B = options[1, SCENARIO_IDX_B]  # (216,)
    mask_C = options[2, SCENARIO_IDX_C]  # (216,)

    # Set points per scenario (int16)
    pts_A = NIBBLE_POPCOUNTS[mask_A].astype(jnp.int16) * RANK_VALUES_16[set_indices[0]]
    pts_B = NIBBLE_POPCOUNTS[mask_B].astype(jnp.int16) * RANK_VALUES_16[set_indices[1]]
    pts_C = NIBBLE_POPCOUNTS[mask_C].astype(jnp.int16) * RANK_VALUES_16[set_indices[2]]
    scenario_set_pts = pts_A + pts_B + pts_C  # (216,) int16

    # --- 4. BUILD REMOVAL MASKS FOR RUNS (using int16 for memory) ---
    def to_suit_removal(mask, rank_idx):
        """Convert nibble mask to per-suit bit removal. Returns (216, 4)."""
        bits = jnp.stack([((mask >> i) & 1) for i in range(4)], axis=-1).astype(jnp.int16)
        return bits * (jnp.int16(1) << rank_idx)

    rem_A = to_suit_removal(mask_A, set_indices[0])  # (216, 4) int16
    rem_B = to_suit_removal(mask_B, set_indices[1])
    rem_C = to_suit_removal(mask_C, set_indices[2])
    total_removal = rem_A | rem_B | rem_C  # (216, 4) int16

    # Base suit bitmasks (int16 is enough for 13-bit masks)
    base_suits = jnp.dot(hand_4x13.astype(jnp.int16), POWERS_OF_2_16)  # (4,)

    # Apply removal to get scenario suit configs
    scenario_suits = base_suits[None, :] & (~total_removal)  # (216, 4)

    # Lookup run scores for each scenario
    base_run_scores = RUN_SCORE_LUT[scenario_suits]  # (216, 4)
    base_run_total = jnp.sum(base_run_scores, axis=1)  # (216,)

    # Total meld points per scenario (before discard)
    base_total = scenario_set_pts + base_run_total  # (216,)

    # --- 5. PROCESS EACH DISCARD ---
    total_hand_value = jnp.sum(hand_4x13.astype(jnp.int16) * RANK_VALUES_16[None, :])

    def process_discard(discard_idx):
        d_suit = discard_idx // 13
        d_rank = discard_idx % 13
        held = hand[discard_idx]

        # A. VALIDITY CHECK: Did we discard a card needed by the scenario?
        fail_A = (d_rank == set_indices[0]) & ((mask_A >> d_suit) & 1).astype(jnp.bool_)
        fail_B = (d_rank == set_indices[1]) & ((mask_B >> d_suit) & 1).astype(jnp.bool_)
        fail_C = (d_rank == set_indices[2]) & ((mask_C >> d_suit) & 1).astype(jnp.bool_)
        is_invalid = fail_A | fail_B | fail_C  # (216,)

        # B. RUN UPDATE: Only the discarded suit changes
        new_suit_bits = scenario_suits[:, d_suit] & (~(jnp.int16(1) << d_rank))  # (216,)
        new_run_score = RUN_SCORE_LUT[new_suit_bits]  # (216,) int16
        old_run_score = base_run_scores[:, d_suit]  # (216,) int16

        # Final meld points = base - old_run + new_run
        final_meld_pts = base_total - old_run_score + new_run_score  # (216,)

        # Filter invalid scenarios
        valid_meld_pts = jnp.where(is_invalid, jnp.int16(-1), final_meld_pts)
        max_meld_pts = jnp.max(valid_meld_pts)

        # Deadwood = hand_value - discard_value - meld_points
        new_hand_value = total_hand_value - RANK_VALUES_16[d_rank]
        dw = new_hand_value - max_meld_pts

        return jnp.where(held > 0, dw, jnp.int16(999))

    return jax.vmap(process_discard)(jnp.arange(52))

@jax.jit
def calculate_deadwood_all_discards_v3(hand):
    """Ultra-optimized: ~224 LUT lookups instead of 7488.

    Uses "Delta Lookups" approach:
    - 4 set scenarios: None, SetA, SetB, Both
    - Precompute base run scores (4 suits × 4 scenarios = 16 lookups)
    - Precompute discard run scores (52 cards × 4 scenarios = 208 lookups)
    - For each discard: O(1) arithmetic to get final answer
    """
    hand_4x13 = hand_to_4x13(hand)
    total_hand_value = jnp.sum(hand_4x13.astype(jnp.int16) * RANK_VALUES_16[None, :])

    # Base suit bits (4,)
    base_suits = jnp.dot(hand_4x13.astype(jnp.int16), POWERS_OF_2_16)

    # Identify set candidates (top 2 ranks with 3+ cards)
    rank_nibbles = jnp.dot(SUIT_POWERS_8, hand_4x13.astype(jnp.int8))
    rank_counts = NIBBLE_POPCOUNTS_8[rank_nibbles]
    is_set_candidate = rank_counts >= 3
    set_indices = jnp.argsort(~is_set_candidate)[:2]
    rank_A = set_indices[0]
    rank_B = set_indices[1]

    # Check if these ranks actually have 3+ cards
    has_set_A = rank_counts[rank_A] >= 3
    has_set_B = rank_counts[rank_B] >= 3

    # Set point values (use all available cards of the rank)
    count_A = jnp.where(has_set_A, rank_counts[rank_A], jnp.int8(0))
    count_B = jnp.where(has_set_B, rank_counts[rank_B], jnp.int8(0))
    set_points_A = count_A.astype(jnp.int16) * RANK_VALUES_16[rank_A]
    set_points_B = count_B.astype(jnp.int16) * RANK_VALUES_16[rank_B]

    set_points = jnp.array([
        jnp.int16(0),                    # Scenario 0: No sets
        set_points_A,                     # Scenario 1: Set A only
        set_points_B,                     # Scenario 2: Set B only
        set_points_A + set_points_B,      # Scenario 3: Both sets
    ])

    # Build suit removal masks for each scenario
    # Clearing bit rank_X removes that rank from runs (no-op if suit doesn't have it)
    mask_A = (jnp.int16(1) << rank_A) * has_set_A.astype(jnp.int16)
    mask_B = (jnp.int16(1) << rank_B) * has_set_B.astype(jnp.int16)

    remove_masks = jnp.array([
        jnp.int16(0),          # Scenario 0: no removal
        mask_A,                # Scenario 1: remove rank A
        mask_B,                # Scenario 2: remove rank B
        mask_A | mask_B,       # Scenario 3: remove both
    ])

    # Step 1: Base_Scores[4 suits, 4 scenarios] = 16 lookups
    base_with_removals = base_suits[:, None] & ~remove_masks[None, :]  # (4, 4)
    base_run_scores = RUN_SCORE_LUT[base_with_removals]  # (4, 4)
    total_base = jnp.sum(base_run_scores, axis=0)  # (4,) sum over suits per scenario

    # Step 2: Discard_Scores[52 cards, 4 scenarios] = 208 lookups
    def discard_run_scores(card_idx):
        suit = card_idx // 13
        rank = card_idx % 13
        card_bit = jnp.int16(1 << rank)
        card_present = hand[card_idx]

        # New suit bits after discarding this card
        new_suit = base_suits[suit] ^ (card_bit * card_present)

        # Run scores under each removal scenario
        new_with_removals = new_suit & ~remove_masks  # (4,)
        return RUN_SCORE_LUT[new_with_removals]  # (4,)

    all_discard_scores = jax.vmap(discard_run_scores)(jnp.arange(52))  # (52, 4)

    # Step 3: For each discard, compute deadwood via O(1) arithmetic
    def deadwood_for_discard(card_idx):
        suit = card_idx // 13
        rank = card_idx % 13
        card_present = hand[card_idx]

        # New_Total = Total_Base - Base_Score[suit] + Discard_Score[card]
        new_run_totals = total_base - base_run_scores[suit] + all_discard_scores[card_idx]  # (4,)

        # Total meld points = run points + set points
        total_meld_points = new_run_totals + set_points  # (4,)

        # Max over all 4 scenarios
        max_meld_points = jnp.max(total_meld_points)

        # Deadwood = hand_value - discarded_card_value - meld_points
        new_hand_value = total_hand_value - RANK_VALUES_16[rank] * card_present
        dw = new_hand_value - max_meld_points

        return jnp.where(card_present > 0, dw, jnp.int16(999))

    return jax.vmap(deadwood_for_discard)(jnp.arange(52))

# Alias for backwards compatibility - use exact 216-scenario version
calculate_deadwood_all_discards = calculate_deadwood_all_discards_exact

# =============================================================================
# Compressed deadwood solver (5x faster - only processes held cards)
# =============================================================================

@jax.jit
def calculate_deadwood_compressed(hand):
    """Calculate deadwood ONLY for cards actually in hand (11 vs 52 iterations).

    Returns:
        held_indices: (11,) int32 - indices of cards in hand (padded with 0)
        deadwoods: (11,) int16 - deadwood if that card is discarded
    """
    # 1. Compress: Find indices of held cards (max 11 in Gin)
    held_indices = jnp.argsort(~(hand > 0))[:11]  # Non-zero first
    held_mask = jnp.take(hand, held_indices) > 0  # (11,) bool

    # 2. Setup (same as exact version)
    hand_4x13 = hand_to_4x13(hand)
    rank_nibbles = jnp.dot(SUIT_POWERS, hand_4x13.astype(jnp.int32))
    rank_counts = NIBBLE_POPCOUNTS[rank_nibbles]

    is_candidate = rank_counts >= 3
    set_indices = jnp.argsort(~is_candidate)[:3]
    set_nibbles = rank_nibbles[set_indices]

    options = SUBSET_TABLE[set_nibbles]
    mask_A = options[0, SCENARIO_IDX_A]
    mask_B = options[1, SCENARIO_IDX_B]
    mask_C = options[2, SCENARIO_IDX_C]

    pts_A = NIBBLE_POPCOUNTS[mask_A].astype(jnp.int16) * RANK_VALUES_16[set_indices[0]]
    pts_B = NIBBLE_POPCOUNTS[mask_B].astype(jnp.int16) * RANK_VALUES_16[set_indices[1]]
    pts_C = NIBBLE_POPCOUNTS[mask_C].astype(jnp.int16) * RANK_VALUES_16[set_indices[2]]
    scenario_set_pts = pts_A + pts_B + pts_C

    def _get_rem(mask, r_idx):
        bits = jnp.stack([((mask >> i) & 1) for i in range(4)], axis=-1).astype(jnp.int16)
        return bits * (jnp.int16(1) << r_idx)

    total_removal = (_get_rem(mask_A, set_indices[0]) |
                     _get_rem(mask_B, set_indices[1]) |
                     _get_rem(mask_C, set_indices[2]))

    base_suits = jnp.dot(hand_4x13.astype(jnp.int16), POWERS_OF_2_16)
    scenario_suits = base_suits[None, :] & (~total_removal)
    base_run_scores = RUN_SCORE_LUT_8[scenario_suits].astype(jnp.int16)
    base_run_total = jnp.sum(base_run_scores, axis=1)
    base_total = scenario_set_pts + base_run_total

    total_hand_value = jnp.sum(hand_4x13.astype(jnp.int16) * RANK_VALUES_16[None, :])

    # 3. Process only the 11 held cards (not 52!)
    def process_held_discard(discard_idx):
        d_suit = discard_idx // 13
        d_rank = discard_idx % 13

        fail_A = (d_rank == set_indices[0]) & ((mask_A >> d_suit) & 1).astype(jnp.bool_)
        fail_B = (d_rank == set_indices[1]) & ((mask_B >> d_suit) & 1).astype(jnp.bool_)
        fail_C = (d_rank == set_indices[2]) & ((mask_C >> d_suit) & 1).astype(jnp.bool_)
        is_invalid = fail_A | fail_B | fail_C

        new_suit_bits = scenario_suits[:, d_suit] & (~(jnp.int16(1) << d_rank))
        new_run_score = RUN_SCORE_LUT_8[new_suit_bits].astype(jnp.int16)
        old_run_score = base_run_scores[:, d_suit]

        final_meld_pts = base_total - old_run_score + new_run_score
        valid_meld_pts = jnp.where(is_invalid, jnp.int16(-1), final_meld_pts)
        max_meld_pts = jnp.max(valid_meld_pts)

        new_hand_value = total_hand_value - RANK_VALUES_16[d_rank]
        return new_hand_value - max_meld_pts

    deadwoods = jax.vmap(process_held_discard)(held_indices)
    deadwoods = jnp.where(held_mask, deadwoods, jnp.int16(999))

    return held_indices, deadwoods


@jax.jit
def card_would_be_in_meld_fast(hand, card):
    """O(1) check if adding card creates/joins a meld using LUT."""
    rank = card % 13
    suit = card // 13

    # 1. Set check: Do we have >= 2 of this rank already?
    rank_count = hand[rank] + hand[rank + 13] + hand[rank + 26] + hand[rank + 39]
    is_set = rank_count >= 2

    # 2. Run check: Use RUN_MEMBER_LUT
    suit_start = suit * 13
    suit_cards = jax.lax.dynamic_slice(hand, (suit_start,), (13,))
    suit_mask = jnp.dot(suit_cards.astype(jnp.int16), POWERS_OF_2_16)
    new_mask = suit_mask | (jnp.int16(1) << rank)
    run_members = RUN_MEMBER_LUT[new_mask]
    is_run = ((run_members >> rank) & 1).astype(jnp.bool_)

    return is_set | is_run


@jax.jit
def simple_bot_action_opt(state):
    """Optimized simple bot using compressed deadwood solver."""
    phase = state['phase']
    player = state['current_player']
    hand = get_hand(state, player)
    upcard = state['upcard']
    deck = state['deck']

    # 1. Compressed deadwood for current hand
    hand_indices, hand_dws = calculate_deadwood_compressed(hand)
    min_dw_hand = jnp.min(hand_dws)

    # Find best discard (minimize deadwood, break ties by highest value)
    cand_values = CARD_VALUES[hand_indices]
    is_optimal = (hand_dws == min_dw_hand) & (hand_dws < 999)
    discard_score = jnp.where(is_optimal, cand_values.astype(jnp.int16) + 1000, jnp.int16(-1))
    best_idx_local = jnp.argmax(discard_score)
    best_discard_card = hand_indices[best_idx_local]

    # 2. Compressed deadwood with upcard (for draw decisions)
    hand_with_upcard = jnp.where(upcard >= 0, hand.at[upcard].set(1), hand)
    _, up_dws = calculate_deadwood_compressed(hand_with_upcard)
    min_dw_with_upcard = jnp.min(up_dws)

    # === DECISION LOGIC ===

    # Draw phase evaluation
    upcard_enables_knock = min_dw_with_upcard <= KNOCK_THRESHOLD
    upcard_in_meld = jnp.where(upcard >= 0, card_would_be_in_meld_fast(hand, upcard), False)
    take_upcard = upcard_enables_knock | upcard_in_meld

    # Chance action
    is_chance = (phase == PHASE_DEAL) | state['waiting_stock_draw']
    chance_action = jnp.argmax(deck)

    # First upcard action
    first_upcard_action = jnp.where(take_upcard, ACTION_DRAW_UPCARD, ACTION_PASS)

    # Draw action
    is_wall = jnp.sum(deck) <= WALL_STOCK_SIZE
    draw_action = jnp.where(take_upcard & (upcard >= 0), ACTION_DRAW_UPCARD, ACTION_DRAW_STOCK)
    draw_action = jnp.where(is_wall & ~take_upcard, ACTION_PASS, draw_action)

    # Discard action
    can_knock = min_dw_hand <= KNOCK_THRESHOLD
    discard_action = jnp.where(can_knock, ACTION_KNOCK, best_discard_card)

    # Knock action
    hand_size = jnp.sum(hand)
    knock_action = jnp.where(hand_size >= 11, best_discard_card, ACTION_PASS)

    # Wall action
    wall_can_knock = min_dw_with_upcard <= KNOCK_THRESHOLD
    wall_action = jnp.where(wall_can_knock & (upcard >= 0), ACTION_KNOCK, ACTION_PASS)

    # Multiplexer
    action = jnp.where(is_chance, chance_action, jnp.int32(0))
    action = jnp.where(phase == PHASE_FIRST_UPCARD, first_upcard_action, action)
    action = jnp.where(phase == PHASE_DRAW, draw_action, action)
    action = jnp.where(phase == PHASE_DISCARD, discard_action, action)
    action = jnp.where(phase == PHASE_KNOCK, knock_action, action)
    action = jnp.where(phase == PHASE_WALL, wall_action, action)

    return action.astype(jnp.int32)

# =============================================================================
# Meld encoding
# =============================================================================

def card_rank(card):
    return card % NUM_RANKS

def card_suit(card):
    return card // NUM_RANKS

def make_card(suit, rank):
    return suit * NUM_RANKS + rank

def generate_all_melds():
    melds = [None] * 185
    for rank in range(NUM_RANKS):
        for missing_suit in range(NUM_SUITS):
            meld_id = rank * 5 + missing_suit
            cards = tuple(make_card(s, rank) for s in range(NUM_SUITS) if s != missing_suit)
            melds[meld_id] = cards
        meld_id = rank * 5 + 4
        cards = tuple(make_card(s, rank) for s in range(NUM_SUITS))
        melds[meld_id] = cards

    offset = 65
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 2):
            meld_id = offset + suit * 11 + start_rank
            cards = tuple(make_card(suit, r) for r in range(start_rank, start_rank + 3))
            melds[meld_id] = cards

    offset = 109
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 3):
            meld_id = offset + suit * 10 + start_rank
            cards = tuple(make_card(suit, r) for r in range(start_rank, start_rank + 4))
            melds[meld_id] = cards

    offset = 149
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 4):
            meld_id = offset + suit * 9 + start_rank
            cards = tuple(make_card(suit, r) for r in range(start_rank, start_rank + 5))
            melds[meld_id] = cards

    return melds

ALL_MELDS = generate_all_melds()

def meld_to_mask(meld):
    mask = np.zeros(NUM_CARDS, dtype=np.int8)
    for card in meld:
        mask[card] = 1
    return mask

MELD_MASKS = jnp.array([meld_to_mask(m) for m in ALL_MELDS], dtype=jnp.int8)

def _compute_overlap_matrix():
    overlap = np.zeros((NUM_MELDS, NUM_MELDS), dtype=np.bool_)
    for i in range(NUM_MELDS):
        for j in range(NUM_MELDS):
            if i == j:
                overlap[i, j] = True
            else:
                set_i = set(ALL_MELDS[i])
                set_j = set(ALL_MELDS[j])
                if set_i & set_j:
                    overlap[i, j] = True
    return jnp.array(overlap)

MELD_OVERLAP = _compute_overlap_matrix()

def _compute_meld_points():
    points = np.zeros(NUM_MELDS, dtype=np.int8)
    for i, meld in enumerate(ALL_MELDS):
        for card in meld:
            rank = card % NUM_RANKS
            points[i] += int(CARD_POINTS[rank])
    return jnp.array(points)

MELD_POINTS = _compute_meld_points()

# Precompute card values (point value of each card 0-51) - used by simple_bot for tie-breaking
CARD_VALUES = jnp.array([CARD_POINTS[i % NUM_RANKS] for i in range(NUM_CARDS)], dtype=jnp.int8)

# Precompute meld sizes for valid_melds_mask
MELD_SIZES = jnp.sum(MELD_MASKS, axis=1)

@jax.jit
def hand_total_points(hand):
    """Calculate total point value of cards in hand."""
    return jnp.sum(hand * CARD_VALUES, dtype=jnp.int16)

@jax.jit
def valid_melds_mask(hand):
    """Check which of the 185 melds are valid for this hand."""
    hand_expanded = hand[None, :]
    meld_card_counts = jnp.sum(MELD_MASKS * hand_expanded, axis=1)
    return meld_card_counts == MELD_SIZES

# =============================================================================
# Game state and step function (simplified for benchmarking)
# =============================================================================

def init_state():
    return {
        'player0_hand': jnp.zeros(NUM_CARDS, dtype=jnp.int8),
        'player1_hand': jnp.zeros(NUM_CARDS, dtype=jnp.int8),
        'deck': jnp.ones(NUM_CARDS, dtype=jnp.int8),
        'upcard': jnp.int8(-1),
        'current_player': jnp.int8(-1),
        'phase': jnp.int8(PHASE_DEAL),
        'done': jnp.bool_(False),
        'winner': jnp.int8(-1),
        'cards_dealt': jnp.int16(0),
        'pass_count': jnp.int8(0),
        'num_draw_upcard': jnp.int8(0),
        'knocker': jnp.int8(-1),
        'knocker_deadwood': jnp.int8(0),
        'layed_melds': jnp.zeros((2, NUM_MELDS), dtype=jnp.bool_),
        'layoffs_mask': jnp.zeros(NUM_CARDS, dtype=jnp.bool_),
        'finished_layoffs': jnp.bool_(False),
        'waiting_stock_draw': jnp.bool_(False),
    }

@jax.jit
def get_hand(state, player):
    return jnp.where(player == 0, state['player0_hand'], state['player1_hand'])

@jax.jit
def legal_actions_mask_fast(state):
    """Fast legal actions - skips knock eligibility (always allows knock in discard phase)."""
    phase = state['phase']
    player = state['current_player']
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)
    hand = get_hand(state, player)
    card_in_hand = hand > 0

    is_chance = (phase == PHASE_DEAL) | state['waiting_stock_draw']
    deck_cards = state['deck'] > 0
    mask = jnp.where(is_chance, mask.at[:NUM_CARDS].set(deck_cards), mask)

    first_upcard_phase = (phase == PHASE_FIRST_UPCARD) & ~is_chance
    has_upcard = state['upcard'] >= 0
    mask = jnp.where(first_upcard_phase, mask.at[ACTION_PASS].set(True), mask)
    mask = jnp.where(first_upcard_phase & has_upcard, mask.at[ACTION_DRAW_UPCARD].set(True), mask)

    draw_phase = (phase == PHASE_DRAW) & ~is_chance
    stock_count = jnp.sum(state['deck'])
    mask = jnp.where(draw_phase & has_upcard, mask.at[ACTION_DRAW_UPCARD].set(True), mask)
    mask = jnp.where(draw_phase & (stock_count > 0), mask.at[ACTION_DRAW_STOCK].set(True), mask)

    # FAST: Just allow discarding any card, always allow knock (step will validate)
    discard_phase = (phase == PHASE_DISCARD) & ~is_chance
    mask = jnp.where(discard_phase, mask.at[:NUM_CARDS].set(card_in_hand), mask)
    mask = jnp.where(discard_phase, mask.at[ACTION_KNOCK].set(True), mask)

    # Skip knock phase handling for speed - just pass
    knock_phase = (phase == PHASE_KNOCK) & ~is_chance
    mask = jnp.where(knock_phase, mask.at[ACTION_PASS].set(True), mask)

    wall_phase = (phase == PHASE_WALL) & ~is_chance
    mask = jnp.where(wall_phase, mask.at[ACTION_PASS].set(True), mask)
    mask = jnp.where(wall_phase, mask.at[ACTION_KNOCK].set(True), mask)

    return mask

@jax.jit
def legal_actions_mask(state):
    phase = state['phase']
    player = state['current_player']
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)
    hand = get_hand(state, player)
    card_in_hand = hand > 0
    valid_melds = valid_melds_mask(hand)

    is_chance = (phase == PHASE_DEAL) | state['waiting_stock_draw']
    deck_cards = state['deck'] > 0
    mask = jnp.where(is_chance, mask.at[:NUM_CARDS].set(deck_cards), mask)

    first_upcard_phase = (phase == PHASE_FIRST_UPCARD) & ~is_chance
    has_upcard = state['upcard'] >= 0
    mask = jnp.where(first_upcard_phase, mask.at[ACTION_PASS].set(True), mask)
    mask = jnp.where(first_upcard_phase & has_upcard, mask.at[ACTION_DRAW_UPCARD].set(True), mask)

    draw_phase = (phase == PHASE_DRAW) & ~is_chance
    stock_count = jnp.sum(state['deck'])
    mask = jnp.where(draw_phase & has_upcard, mask.at[ACTION_DRAW_UPCARD].set(True), mask)
    mask = jnp.where(draw_phase & (stock_count > 0), mask.at[ACTION_DRAW_STOCK].set(True), mask)

    # Only compute deadwood for phases that need it (discard, knock)
    needs_deadwood = (phase == PHASE_DISCARD) | (phase == PHASE_KNOCK)
    def compute_deadwood():
        return calculate_deadwood_all_discards(hand)
    def skip_deadwood():
        return jnp.full(52, 999, dtype=jnp.int16)
    all_deadwoods = jax.lax.cond(needs_deadwood & ~is_chance, compute_deadwood, skip_deadwood)
    min_dw = jnp.min(all_deadwoods)

    discard_phase = (phase == PHASE_DISCARD) & ~is_chance
    can_knock_discard = discard_phase & (min_dw <= KNOCK_THRESHOLD)
    mask = jnp.where(discard_phase, mask.at[:NUM_CARDS].set(card_in_hand), mask)
    mask = jnp.where(can_knock_discard, mask.at[ACTION_KNOCK].set(True), mask)

    knock_phase = (phase == PHASE_KNOCK) & ~is_chance
    hand_count = jnp.sum(hand)
    has_11_cards = hand_count == 11
    knock_discard_mask = (hand > 0) & (all_deadwoods <= KNOCK_THRESHOLD)
    mask = jnp.where(knock_phase & has_11_cards, mask.at[:NUM_CARDS].set(knock_discard_mask), mask)
    knock_10 = knock_phase & (hand_count <= 10)
    meld_mask = knock_10 & valid_melds
    remaining_points = hand_total_points(hand)
    can_pass_knock = remaining_points <= KNOCK_THRESHOLD
    mask = jnp.where(knock_10 & can_pass_knock, mask.at[ACTION_PASS].set(True), mask)
    mask = jnp.where(knock_10, mask.at[ACTION_MELD_BASE:ACTION_MELD_BASE + NUM_MELDS].set(meld_mask), mask)

    wall_phase = (phase == PHASE_WALL) & ~is_chance
    mask = jnp.where(wall_phase, mask.at[ACTION_PASS].set(True), mask)
    upcard = state['upcard']
    # Use lax.cond to skip expensive deadwood calculation when not in wall phase
    def compute_wall_knock():
        hand_with_upcard = jnp.where(upcard >= 0, hand.at[upcard].set(1), hand)
        return jnp.min(calculate_deadwood_all_discards(hand_with_upcard)) <= KNOCK_THRESHOLD
    can_knock_wall = jax.lax.cond(wall_phase, compute_wall_knock, lambda: False)
    mask = jnp.where(wall_phase & can_knock_wall, mask.at[ACTION_KNOCK].set(True), mask)

    return mask

@jax.jit
def step(state, action):
    # Simplified step - just handle basic dealing and game flow for benchmarking
    phase = state['phase']
    player = state['current_player']

    # Deal phase
    is_deal = phase == PHASE_DEAL
    cards_dealt = state['cards_dealt']

    # Update deck
    new_deck = jnp.where(is_deal, state['deck'].at[action].set(0), state['deck'])

    # Update hands during deal
    deal_to_p0 = is_deal & (cards_dealt < 10)
    deal_to_p1 = is_deal & (cards_dealt >= 10) & (cards_dealt < 20)
    deal_upcard = is_deal & (cards_dealt == 20)

    new_p0_hand = jnp.where(deal_to_p0, state['player0_hand'].at[action].set(1), state['player0_hand'])
    new_p1_hand = jnp.where(deal_to_p1, state['player1_hand'].at[action].set(1), state['player1_hand'])
    new_upcard = jnp.where(deal_upcard, jnp.int8(action), state['upcard'])

    new_cards_dealt = jnp.where(is_deal, cards_dealt + 1, cards_dealt)

    # Transition from deal to first upcard phase
    deal_done = is_deal & (new_cards_dealt == 21)
    new_phase = jnp.where(deal_done, jnp.int8(PHASE_FIRST_UPCARD), state['phase'])
    new_player = jnp.where(deal_done, jnp.int8(0), state['current_player'])

    # First upcard phase
    is_first = phase == PHASE_FIRST_UPCARD
    took_upcard = is_first & (action == ACTION_DRAW_UPCARD)
    passed = is_first & (action == ACTION_PASS)

    # If took upcard, add to hand
    new_p0_hand = jnp.where(took_upcard & (player == 0), new_p0_hand.at[state['upcard']].set(1), new_p0_hand)
    new_p1_hand = jnp.where(took_upcard & (player == 1), new_p1_hand.at[state['upcard']].set(1), new_p1_hand)
    new_upcard = jnp.where(took_upcard, jnp.int8(-1), new_upcard)
    new_phase = jnp.where(took_upcard, jnp.int8(PHASE_DISCARD), new_phase)

    # Handle pass in first upcard - both passed = go to draw
    new_pass_count = jnp.where(passed, state['pass_count'] + 1, state['pass_count'])
    both_passed = passed & (new_pass_count >= 2)
    new_phase = jnp.where(both_passed, jnp.int8(PHASE_DRAW), new_phase)
    new_pass_count = jnp.where(both_passed, jnp.int8(0), new_pass_count)
    new_player = jnp.where(passed & ~both_passed, 1 - player, new_player)

    # Draw phase
    is_draw = phase == PHASE_DRAW
    drew_upcard = is_draw & (action == ACTION_DRAW_UPCARD)
    drew_stock = is_draw & (action == ACTION_DRAW_STOCK)

    new_p0_hand = jnp.where(drew_upcard & (player == 0), new_p0_hand.at[state['upcard']].set(1), new_p0_hand)
    new_p1_hand = jnp.where(drew_upcard & (player == 1), new_p1_hand.at[state['upcard']].set(1), new_p1_hand)
    new_upcard = jnp.where(drew_upcard, jnp.int8(-1), new_upcard)
    new_phase = jnp.where(drew_upcard, jnp.int8(PHASE_DISCARD), new_phase)

    # Stock draw - goes to waiting for chance
    new_waiting = jnp.where(drew_stock, jnp.bool_(True), state['waiting_stock_draw'])

    # Waiting for stock draw (chance node)
    is_waiting = state['waiting_stock_draw']
    new_p0_hand = jnp.where(is_waiting & (player == 0), new_p0_hand.at[action].set(1), new_p0_hand)
    new_p1_hand = jnp.where(is_waiting & (player == 1), new_p1_hand.at[action].set(1), new_p1_hand)
    new_deck = jnp.where(is_waiting, new_deck.at[action].set(0), new_deck)
    new_waiting = jnp.where(is_waiting, jnp.bool_(False), new_waiting)
    new_phase = jnp.where(is_waiting, jnp.int8(PHASE_DISCARD), new_phase)

    # Discard phase
    is_discard = phase == PHASE_DISCARD
    discarded = is_discard & (action < NUM_CARDS)
    knocked = is_discard & (action == ACTION_KNOCK)

    new_p0_hand = jnp.where(discarded & (player == 0), new_p0_hand.at[action].set(0), new_p0_hand)
    new_p1_hand = jnp.where(discarded & (player == 1), new_p1_hand.at[action].set(0), new_p1_hand)
    new_upcard = jnp.where(discarded, jnp.int8(action), new_upcard)

    # Check for wall
    stock_count = jnp.sum(new_deck)
    enter_wall = discarded & (stock_count <= WALL_STOCK_SIZE)
    new_phase = jnp.where(discarded & ~enter_wall, jnp.int8(PHASE_DRAW), new_phase)
    new_phase = jnp.where(enter_wall, jnp.int8(PHASE_WALL), new_phase)
    new_player = jnp.where(discarded, 1 - player, new_player)

    new_phase = jnp.where(knocked, jnp.int8(PHASE_KNOCK), new_phase)

    # Simplified: end game after some moves for benchmarking
    move_limit = state['cards_dealt'] > 200
    new_done = jnp.where(move_limit, jnp.bool_(True), state['done'])
    new_phase = jnp.where(move_limit, jnp.int8(PHASE_GAME_OVER), new_phase)

    return {
        'player0_hand': new_p0_hand,
        'player1_hand': new_p1_hand,
        'deck': new_deck,
        'upcard': new_upcard,
        'current_player': new_player,
        'phase': new_phase,
        'done': new_done,
        'winner': state['winner'],
        'cards_dealt': new_cards_dealt,
        'pass_count': new_pass_count,
        'num_draw_upcard': state['num_draw_upcard'],
        'knocker': state['knocker'],
        'knocker_deadwood': state['knocker_deadwood'],
        'layed_melds': state['layed_melds'],
        'layoffs_mask': state['layoffs_mask'],
        'finished_layoffs': state['finished_layoffs'],
        'waiting_stock_draw': new_waiting,
    }

# =============================================================================
# Simple Gin Bot
# =============================================================================

@jax.jit
def best_discard(hand):
    """Find the best card to discard (minimizes deadwood, breaks ties by highest value).

    Returns the card index that, when discarded, results in minimum deadwood.
    Among cards with equal minimum deadwood, picks the highest-value card.
    """
    dw_after = calculate_deadwood_all_discards(hand)
    min_dw = jnp.min(dw_after)
    # Cards that achieve minimum deadwood
    is_optimal = (dw_after == min_dw) & (hand > 0)
    # Among optimal cards, pick highest value (break ties)
    # Use large negative for non-optimal so argmax picks from optimal set
    score = jnp.where(is_optimal, CARD_VALUES.astype(jnp.int16) + 100, jnp.int16(-1))
    return jnp.argmax(score)

@jax.jit
def card_would_be_in_meld(hand, card):
    """Check if adding card to hand would put it in at least one valid meld."""
    new_hand = hand.at[card].set(1)
    valid = valid_melds_mask(new_hand)
    # Check if any valid meld contains this card
    card_in_meld = MELD_MASKS[:, card] > 0
    return jnp.any(valid & card_in_meld)

@jax.jit
def simple_bot_action(state):
    """Returns the action the simple bot would take. Vmappable.

    Strategy:
    - Draw: take upcard if it enables knock or belongs to a meld, else draw stock
    - Discard: knock if able, else discard highest-value deadwood card
    - Knock: discard highest deadwood, then pass
    - Wall: knock if able, else pass

    Optimized: computes deadwood only once per hand configuration (2 calls instead of 4).
    """
    phase = state['phase']
    player = state['current_player']
    hand = get_hand(state, player)
    upcard = state['upcard']
    deck = state['deck']

    # === Precompute deadwood arrays ONCE (using exact 216-scenario) ===
    # For current hand (used in discard/knock phases)
    dw_hand = calculate_deadwood_all_discards_exact(hand)
    min_dw_hand = jnp.min(dw_hand)

    # For hand with upcard (used in draw/wall phases)
    hand_with_upcard = jnp.where(upcard >= 0, hand.at[upcard].set(1), hand)
    dw_with_upcard = calculate_deadwood_all_discards_exact(hand_with_upcard)
    min_dw_with_upcard = jnp.min(dw_with_upcard)

    # Best discard from current hand (reused in discard and knock phases)
    is_optimal = (dw_hand == min_dw_hand) & (hand > 0)
    score = jnp.where(is_optimal, CARD_VALUES.astype(jnp.int16) + 100, jnp.int16(-1))
    best_discard_card = jnp.argmax(score)

    # === Chance node: pick first available card from deck ===
    is_chance = (phase == PHASE_DEAL) | state['waiting_stock_draw']
    chance_action = jnp.argmax(deck)

    # === First upcard / Draw phase decision ===
    upcard_enables_knock = min_dw_with_upcard <= KNOCK_THRESHOLD
    upcard_in_meld = jnp.where(upcard >= 0, card_would_be_in_meld(hand, upcard), False)
    take_upcard = upcard_enables_knock | upcard_in_meld
    first_upcard_action = jnp.where(take_upcard, ACTION_DRAW_UPCARD, ACTION_PASS)

    # === Draw phase ===
    stock_count = jnp.sum(deck)
    at_wall = stock_count <= WALL_STOCK_SIZE
    draw_action = jnp.where(take_upcard & (upcard >= 0), ACTION_DRAW_UPCARD, ACTION_DRAW_STOCK)
    draw_action = jnp.where(at_wall & ~take_upcard, ACTION_PASS, draw_action)

    # === Discard phase ===
    can_knock = min_dw_hand <= KNOCK_THRESHOLD
    discard_action = jnp.where(can_knock, ACTION_KNOCK, best_discard_card)

    # === Knock phase ===
    hand_count = jnp.sum(hand)
    knock_action = jnp.where(hand_count >= 11, best_discard_card, ACTION_PASS)

    # === Wall phase ===
    wall_can_knock = min_dw_with_upcard <= KNOCK_THRESHOLD
    wall_action = jnp.where(wall_can_knock & (upcard >= 0), ACTION_KNOCK, ACTION_PASS)

    # === Select action based on phase ===
    action = jnp.where(is_chance, chance_action, 0)
    action = jnp.where(phase == PHASE_FIRST_UPCARD, first_upcard_action, action)
    action = jnp.where(phase == PHASE_DRAW, draw_action, action)
    action = jnp.where(phase == PHASE_DISCARD, discard_action, action)
    action = jnp.where(phase == PHASE_KNOCK, knock_action, action)
    action = jnp.where(phase == PHASE_WALL, wall_action, action)

    return action.astype(jnp.int32)

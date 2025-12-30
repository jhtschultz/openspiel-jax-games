"""Precomputed lookup tables for Gin Rummy deadwood calculation.

This module contains all LUT-building functions and precomputed arrays
shared between gin_rummy_core.py and gin_rummy_jax.py.

These tables are computed once at import time and enable O(1) deadwood
lookups that achieve 200-400x speedup over C++ OpenSpiel.
"""

import numpy as np
import jax.numpy as jnp

from constants import NUM_RANKS, NUM_SUITS, NUM_CARDS


# =============================================================================
# Core Value Tables
# =============================================================================

# Card point values by rank: A=1, 2-9=face, 10/J/Q/K=10
RANK_VALUES = jnp.array([min(i + 1, 10) for i in range(NUM_RANKS)], dtype=jnp.int32)
RANK_VALUES_16 = RANK_VALUES.astype(jnp.int16)
RANK_VALUES_8 = RANK_VALUES.astype(jnp.int8)

# Card values for each card index (0-51)
CARD_VALUES = jnp.array([RANK_VALUES[i % NUM_RANKS] for i in range(NUM_CARDS)], dtype=jnp.int8)

# Card points by rank (same as RANK_VALUES but named consistently with original)
CARD_POINTS = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=jnp.int32)
CARD_POINTS_8 = CARD_POINTS.astype(jnp.int8)

# Powers of 2 for bit manipulation
POWERS_OF_2 = 2 ** jnp.arange(NUM_RANKS, dtype=jnp.int32)
POWERS_OF_2_16 = POWERS_OF_2.astype(jnp.int16)

# Suit powers for nibble encoding (4 bits)
SUIT_POWERS = jnp.array([1, 2, 4, 8], dtype=jnp.int32)
SUIT_POWERS_8 = SUIT_POWERS.astype(jnp.int8)


# =============================================================================
# Nibble Tables (4-bit set representations)
# =============================================================================

# All 16 possible 4-bit masks for set subsets
ALL_NIBBLES = jnp.arange(16, dtype=jnp.int32)

# Popcount for each nibble value
NIBBLE_POPCOUNTS = jnp.array([bin(i).count('1') for i in range(16)], dtype=jnp.int32)
NIBBLE_POPCOUNTS_8 = NIBBLE_POPCOUNTS.astype(jnp.int8)

# Precompute suit expansion for all nibble values (16, 4)
NIBBLE_TO_SUITS = jnp.stack([((jnp.arange(16) >> i) & 1) for i in range(4)], axis=-1)
NIBBLE_TO_SUITS_8 = NIBBLE_TO_SUITS.astype(jnp.int8)


# =============================================================================
# Set Combination Grids (256 combos for 2-set enumeration)
# =============================================================================

_idx_grid = jnp.arange(16)
_grid_A, _grid_B = jnp.meshgrid(_idx_grid, _idx_grid)
GRID_A = _grid_A.flatten()  # (256,)
GRID_B = _grid_B.flatten()  # (256,)
MASK_A_ALL = ALL_NIBBLES[GRID_A]  # (256,)
MASK_B_ALL = ALL_NIBBLES[GRID_B]  # (256,)
POPCNT_A = NIBBLE_POPCOUNTS[GRID_A]  # (256,)
POPCNT_B = NIBBLE_POPCOUNTS[GRID_B]  # (256,)


# =============================================================================
# LUT Building Functions
# =============================================================================

def _build_run_lookup_table():
    """Build lookup table: suit bitmask (13 bits) -> max points from runs.

    Uses DP: for each position, either skip or start a run of length 3/4/5/...
    Returns (8192,) int16 array.
    """
    card_values = [min(i + 1, 10) for i in range(13)]  # A=1, 2-9, 10/J/Q/K=10
    lut = []

    for bits in range(8192):  # 2^13 configurations
        held = [(bits >> i) & 1 for i in range(13)]

        # DP: dp[i] = max points using cards from position i onwards
        dp = [0] * 14  # dp[13] = 0 (base case)

        for i in range(12, -1, -1):  # Process right to left
            best = dp[i + 1]  # Option 1: skip this position

            # Option 2: start a run at position i (if possible)
            for length in range(3, 14 - i):  # runs of length 3, 4, 5...
                if all(held[i + k] for k in range(length)):
                    run_points = sum(card_values[i + k] for k in range(length))
                    best = max(best, run_points + dp[i + length])
                else:
                    break

            dp[i] = best

        lut.append(dp[0])

    return jnp.array(lut, dtype=jnp.int16)


def _build_valid_set_modes():
    """For each nibble, list valid set subsets (size 0, 3, or 4).

    Returns:
        modes: (16, 6) array of valid subset nibbles
        mode_counts: (16,) array of counts
    """
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


def _build_subset_table():
    """Build (16, 6) table mapping held nibble -> valid subset options.

    For each nibble (0-15), list valid subsets:
    - 0 (don't use as set)
    - Any subset with popcount >= 3 that fits within held cards
    """
    table = np.zeros((16, 6), dtype=np.int32)

    for nibble in range(16):
        options = [0]  # Always can skip
        popcount = bin(nibble).count('1')

        if popcount >= 3:
            for subset in range(1, 16):
                subset_pop = bin(subset).count('1')
                if (nibble & subset) == subset and subset_pop >= 3:
                    options.append(subset)

        while len(options) < 6:
            options.append(0)
        table[nibble] = options[:6]

    return jnp.array(table, dtype=jnp.int32)


def _build_run_member_table():
    """Build (8192,) table: which ranks are part of a run for each suit config.

    Value is a 13-bit mask indicating which ranks participate in runs.
    """
    lut = []
    for bits in range(8192):
        held = [(bits >> i) & 1 for i in range(13)]
        is_member = [0] * 13

        for i in range(11):  # Start positions 0..10
            if held[i] and held[i+1] and held[i+2]:
                is_member[i] = 1
                is_member[i+1] = 1
                is_member[i+2] = 1
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


def _build_run_decomposition_table():
    """Build (8192, 5) table: optimal run meld IDs for each suit config.

    For suit 0 (spades), returns actual meld IDs. Adjust for other suits at runtime.
    Meld ID encoding:
    - 3-card runs: 65 + suit*11 + start_rank
    - 4-card runs: 109 + suit*10 + start_rank
    - 5-card runs: 149 + suit*9 + start_rank

    Uses greedy decomposition: for each maximal run, split into valid melds.
    """
    lut = []
    for bits in range(8192):
        held = [(bits >> i) & 1 for i in range(13)]

        # Find maximal runs
        runs = []  # List of (start, length)
        i = 0
        while i < 13:
            if held[i]:
                start = i
                while i < 13 and held[i]:
                    i += 1
                length = i - start
                if length >= 3:
                    runs.append((start, length))
            else:
                i += 1

        # Decompose each run into 3/4/5 card melds
        meld_ids = []
        for start, length in runs:
            pos = start
            remaining = length
            while remaining >= 3:
                if remaining == 3:
                    meld_ids.append(65 + pos)  # 3-card run for suit 0
                    pos += 3
                    remaining -= 3
                elif remaining == 4:
                    meld_ids.append(109 + pos)  # 4-card run
                    pos += 4
                    remaining -= 4
                elif remaining == 5:
                    meld_ids.append(149 + pos)  # 5-card run
                    pos += 5
                    remaining -= 5
                elif remaining == 6:
                    # Two 3-card runs
                    meld_ids.append(65 + pos)
                    meld_ids.append(65 + pos + 3)
                    remaining = 0
                elif remaining == 7:
                    # 3 + 4
                    meld_ids.append(65 + pos)
                    meld_ids.append(109 + pos + 3)
                    remaining = 0
                elif remaining == 8:
                    # 3 + 5
                    meld_ids.append(65 + pos)
                    meld_ids.append(149 + pos + 3)
                    remaining = 0
                elif remaining >= 9:
                    # Take a 3-card run, continue
                    meld_ids.append(65 + pos)
                    pos += 3
                    remaining -= 3

        # Pad to 5 entries
        while len(meld_ids) < 5:
            meld_ids.append(-1)

        lut.append(meld_ids[:5])

    return jnp.array(lut, dtype=jnp.int32)


# =============================================================================
# Precomputed Tables (built at import time)
# =============================================================================

# Run scoring LUT: suit bitmask -> max points from runs
RUN_SCORE_LUT = _build_run_lookup_table()
RUN_SCORE_LUT_8 = RUN_SCORE_LUT.astype(jnp.int8)

# Valid set modes per nibble
VALID_SET_MODES, VALID_SET_MODE_COUNTS = _build_valid_set_modes()
VALID_SET_MODES_8 = VALID_SET_MODES.astype(jnp.int8)
VALID_SET_MODE_COUNTS_8 = VALID_SET_MODE_COUNTS.astype(jnp.int8)

# Subset table for set enumeration
SUBSET_TABLE = _build_subset_table()

# Run membership LUT: suit bitmask -> which ranks are in runs
RUN_MEMBER_LUT = _build_run_member_table()

# Run decomposition LUT: suit bitmask -> optimal meld IDs
RUN_DECOMP_LUT = _build_run_decomposition_table()


# =============================================================================
# Combo Index Tables (for 36-combo and 216-scenario variants)
# =============================================================================

# 36 valid combo pairs (6 modes × 6 modes for 2-set enumeration)
_idx6 = np.arange(6)
_g0, _g1 = np.meshgrid(_idx6, _idx6)
COMBO_IDX_0 = jnp.array(_g0.flatten())  # (36,)
COMBO_IDX_1 = jnp.array(_g1.flatten())  # (36,)
COMBO_IDX_0_8 = COMBO_IDX_0.astype(jnp.int8)
COMBO_IDX_1_8 = COMBO_IDX_1.astype(jnp.int8)

# 216 scenario indices (6 × 6 × 6 for 3-set enumeration)
_idx6_jnp = jnp.arange(6)
_grid_A_216, _grid_B_216, _grid_C_216 = jnp.meshgrid(_idx6_jnp, _idx6_jnp, _idx6_jnp, indexing='ij')
SCENARIO_IDX_A = _grid_A_216.flatten()  # (216,)
SCENARIO_IDX_B = _grid_B_216.flatten()  # (216,)
SCENARIO_IDX_C = _grid_C_216.flatten()  # (216,)

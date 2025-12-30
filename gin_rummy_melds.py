"""Meld encoding and generation for Gin Rummy.

This module contains all meld-related functions and precomputed arrays
shared between gin_rummy_core.py and gin_rummy_jax.py.

Meld Encoding (matching C++ OpenSpiel):
- 185 total melds
- Indices 0-64: Rank melds (sets of 3-4 same rank)
- Indices 65-184: Suit melds (runs of 3-5 consecutive same suit)
"""

import numpy as np
import jax.numpy as jnp

from constants import (
    NUM_CARDS, NUM_RANKS, NUM_SUITS, NUM_MELDS,
    card_rank, card_suit, make_card
)
from gin_rummy_luts import CARD_POINTS


# =============================================================================
# Meld Generation
# =============================================================================

def generate_all_melds():
    """Generate all valid melds (sets and runs).

    Matches C++ encoding: 185 melds total
    Card encoding: card = suit * 13 + rank

    Returns:
        List of 185 tuples, each containing card indices in the meld.
    """
    melds = [None] * NUM_MELDS

    # Rank melds (sets) - indices 0-64
    # Same rank, different suits
    for rank in range(NUM_RANKS):
        # Size 3: missing one suit
        for missing_suit in range(NUM_SUITS):
            meld_id = rank * 5 + missing_suit
            cards = tuple(make_card(rank, s) for s in range(NUM_SUITS) if s != missing_suit)
            melds[meld_id] = cards
        # Size 4: all suits
        meld_id = rank * 5 + 4
        cards = tuple(make_card(rank, s) for s in range(NUM_SUITS))
        melds[meld_id] = cards

    # Suit melds (runs) - indices 65-184
    # Same suit, consecutive ranks
    offset = 65

    # Size 3 runs: 11 per suit (start ranks 0-10)
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 2):  # 0-10
            meld_id = offset + suit * 11 + start_rank
            cards = tuple(make_card(r, suit) for r in range(start_rank, start_rank + 3))
            melds[meld_id] = cards

    offset = 109  # 65 + 44

    # Size 4 runs: 10 per suit (start ranks 0-9)
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 3):  # 0-9
            meld_id = offset + suit * 10 + start_rank
            cards = tuple(make_card(r, suit) for r in range(start_rank, start_rank + 4))
            melds[meld_id] = cards

    offset = 149  # 109 + 40

    # Size 5 runs: 9 per suit (start ranks 0-8)
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 4):  # 0-8
            meld_id = offset + suit * 9 + start_rank
            cards = tuple(make_card(r, suit) for r in range(start_rank, start_rank + 5))
            melds[meld_id] = cards

    return melds


def meld_to_mask(meld):
    """Convert a meld (tuple of card indices) to a binary mask."""
    mask = np.zeros(NUM_CARDS, dtype=np.int8)
    for card in meld:
        mask[card] = 1
    return mask


def _compute_overlap_matrix(melds):
    """Compute which melds overlap (share cards) with each other.

    Returns:
        (185, 185) boolean array where overlap[i,j] = True if melds i,j share cards
    """
    overlap = np.zeros((NUM_MELDS, NUM_MELDS), dtype=np.bool_)
    for i in range(NUM_MELDS):
        for j in range(NUM_MELDS):
            if i == j:
                overlap[i, j] = True
            else:
                set_i = set(melds[i])
                set_j = set(melds[j])
                if set_i & set_j:
                    overlap[i, j] = True
    return jnp.array(overlap)


def _compute_meld_points(melds):
    """Compute total point value of each meld.

    Returns:
        (185,) int32 array of meld point values
    """
    points = np.zeros(NUM_MELDS, dtype=np.int32)
    for i, meld in enumerate(melds):
        for card in meld:
            rank = card_rank(card)
            points[i] += int(CARD_POINTS[rank])
    return jnp.array(points)


# =============================================================================
# Precomputed Meld Tables
# =============================================================================

# All 185 valid melds
ALL_MELDS = generate_all_melds()

# Binary masks for each meld (185, 52)
MELD_MASKS = jnp.array([meld_to_mask(m) for m in ALL_MELDS], dtype=jnp.int8)

# Overlap matrix (185, 185) - True if melds share any cards
MELD_OVERLAP = _compute_overlap_matrix(ALL_MELDS)

# Point values for each meld (185,)
MELD_POINTS = _compute_meld_points(ALL_MELDS)


# =============================================================================
# Utility Functions
# =============================================================================

def get_meld_cards(meld_id):
    """Get the card indices in a meld.

    Args:
        meld_id: Meld index (0-184)

    Returns:
        Tuple of card indices
    """
    if 0 <= meld_id < NUM_MELDS:
        return ALL_MELDS[meld_id]
    return ()


def meld_to_str(meld_id):
    """Convert meld ID to human-readable string.

    Args:
        meld_id: Meld index (0-184)

    Returns:
        String like "As,2s,3s" or "Ah,Ac,Ad"
    """
    from constants import card_str as cs
    if 0 <= meld_id < NUM_MELDS:
        cards = ALL_MELDS[meld_id]
        return ",".join(cs(c) for c in cards)
    return f"INVALID({meld_id})"


def is_set_meld(meld_id):
    """Check if meld is a set (same rank, different suits)."""
    return 0 <= meld_id < 65


def is_run_meld(meld_id):
    """Check if meld is a run (consecutive ranks, same suit)."""
    return 65 <= meld_id < NUM_MELDS


def get_meld_size(meld_id):
    """Get the number of cards in a meld."""
    if 0 <= meld_id < NUM_MELDS:
        return len(ALL_MELDS[meld_id])
    return 0

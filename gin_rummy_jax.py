"""JAX-accelerated Gin Rummy for OpenSpiel.

This implements Gin Rummy matching the C++ OpenSpiel implementation exactly,
including explicit chance nodes for dealing and stock draws.

Card encoding (matching C++): 0-51 where card = suit * 13 + rank
  Suits: spades=0, clubs=1, diamonds=2, hearts=3
  Ranks: A=0, 2=1, ..., K=12

  Cards 0-12: spades (As, 2s, ..., Ks)
  Cards 13-25: clubs (Ac, 2c, ..., Kc)
  Cards 26-38: diamonds (Ad, 2d, ..., Kd)
  Cards 39-51: hearts (Ah, 2h, ..., Kh)
"""

from typing import List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import pyspiel

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

# Card point values (A=1, 2-9=face, 10/J/Q/K=10)
CARD_POINTS = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=jnp.int32)

# Game phases (matching C++ Phase enum)
PHASE_DEAL = 0        # Chance node: dealing cards
PHASE_FIRST_UPCARD = 1
PHASE_DRAW = 2
PHASE_DISCARD = 3
PHASE_KNOCK = 4
PHASE_LAYOFF = 5
PHASE_WALL = 6        # Added: explicit wall phase
PHASE_GAME_OVER = 7

# Limits (matching C++)
MAX_NUM_DRAW_UPCARD_ACTIONS = 50
WALL_STOCK_SIZE = 2  # When stock <= this, enter wall phase

# Actions (matching C++)
# 0-51: card actions (deal in chance nodes, discard in regular play)
ACTION_DRAW_UPCARD = 52
ACTION_DRAW_STOCK = 53
ACTION_PASS = 54
ACTION_KNOCK = 55
ACTION_MELD_BASE = 56
# Actions 56-240 are meld declarations (185 melds)

NUM_MELDS = 185
NUM_ACTIONS = 241  # 0-51 cards, 52-55 special, 56-240 melds

# Special player IDs
CHANCE_PLAYER = -1
TERMINAL_PLAYER = -4

# =============================================================================
# Meld encoding (same as before - this part is correct)
# =============================================================================

def card_rank(card):
    """Get rank (0-12) from card using C++ encoding: card = suit * 13 + rank."""
    return card % NUM_RANKS

def card_suit(card):
    """Get suit (0-3) from card using C++ encoding: card = suit * 13 + rank."""
    return card // NUM_RANKS

def make_card(suit, rank):
    """Create card from suit and rank using C++ encoding."""
    return suit * NUM_RANKS + rank


def generate_all_melds():
    """Generate all valid melds (sets and runs).

    Matches C++ encoding: 185 melds total
    Card encoding: card = suit * 13 + rank
    """
    melds = [None] * 185

    # Rank melds (sets) - indices 0-64
    # Same rank, different suits
    for rank in range(NUM_RANKS):
        # Size 3: missing one suit
        for missing_suit in range(NUM_SUITS):
            meld_id = rank * 5 + missing_suit
            cards = tuple(make_card(s, rank) for s in range(NUM_SUITS) if s != missing_suit)
            melds[meld_id] = cards
        # Size 4: all suits
        meld_id = rank * 5 + 4
        cards = tuple(make_card(s, rank) for s in range(NUM_SUITS))
        melds[meld_id] = cards

    # Suit melds (runs) - indices 65-184
    # Same suit, consecutive ranks
    offset = 65
    # Size 3 runs: 11 per suit (start ranks 0-10)
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 2):  # 0-10
            meld_id = offset + suit * 11 + start_rank
            cards = tuple(make_card(suit, r) for r in range(start_rank, start_rank + 3))
            melds[meld_id] = cards

    offset = 109  # 65 + 44
    # Size 4 runs: 10 per suit (start ranks 0-9)
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 3):  # 0-9
            meld_id = offset + suit * 10 + start_rank
            cards = tuple(make_card(suit, r) for r in range(start_rank, start_rank + 4))
            melds[meld_id] = cards

    offset = 149  # 109 + 40
    # Size 5 runs: 9 per suit (start ranks 0-8)
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 4):  # 0-8
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


# =============================================================================
# Deadwood calculation (keeping the existing correct implementation)
# =============================================================================

# Precompute meld overlap matrix
def _compute_overlap_matrix():
    overlap = np.zeros((NUM_MELDS, NUM_MELDS), dtype=np.bool_)
    for i in range(NUM_MELDS):
        for j in range(NUM_MELDS):
            # A meld overlaps with itself (can't use same meld twice)
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
    points = np.zeros(NUM_MELDS, dtype=np.int32)
    for i, meld in enumerate(ALL_MELDS):
        for card in meld:
            rank = card % NUM_RANKS  # C++ encoding
            points[i] += int(CARD_POINTS[rank])
    return jnp.array(points)

MELD_POINTS = _compute_meld_points()

# =============================================================================
# Vectorized meld combination search (K=24 max melds)
# =============================================================================

MAX_MELDS_IN_HAND = 24  # Max possible melds in an 11-card hand

def _generate_selection_matrix(k):
    """Generate all subsets of size 0, 1, 2, 3 from k elements.

    Returns a binary matrix where each row is a subset selection.
    For k=24: 1 + 24 + 276 + 2024 = 2325 rows.
    """
    from itertools import combinations

    subsets = []
    # Size 0: empty set
    subsets.append([0] * k)
    # Size 1
    for i in range(k):
        row = [0] * k
        row[i] = 1
        subsets.append(row)
    # Size 2
    for i, j in combinations(range(k), 2):
        row = [0] * k
        row[i] = 1
        row[j] = 1
        subsets.append(row)
    # Size 3
    for i, j, l in combinations(range(k), 3):
        row = [0] * k
        row[i] = 1
        row[j] = 1
        row[l] = 1
        subsets.append(row)

    return np.array(subsets, dtype=np.int8)

SELECTION_MATRIX = jnp.array(_generate_selection_matrix(MAX_MELDS_IN_HAND))
NUM_COMBOS = SELECTION_MATRIX.shape[0]  # 2325 for K=24


@jax.jit
def hand_total_points(hand):
    """Calculate total point value of all cards in hand.

    Uses C++ encoding: card = suit * 13 + rank, so rank = card % 13
    """
    card_vals = jnp.zeros(NUM_CARDS, dtype=jnp.int32)
    for i in range(NUM_CARDS):
        rank = i % NUM_RANKS  # C++ encoding
        card_vals = card_vals.at[i].set(CARD_POINTS[rank])
    return jnp.sum(hand * card_vals)


@jax.jit
def valid_melds_mask(hand):
    """Return mask of which melds are valid for this hand."""
    hand_expanded = hand[None, :]
    meld_card_counts = jnp.sum(MELD_MASKS * hand_expanded, axis=1)
    meld_sizes = jnp.sum(MELD_MASKS, axis=1)
    return meld_card_counts == meld_sizes


@jax.jit
def calculate_deadwood(hand):
    """Calculate minimum deadwood for a hand using exhaustive meld search."""
    total_points = hand_total_points(hand)
    valid = valid_melds_mask(hand)
    v_points = jnp.where(valid, MELD_POINTS, 0)

    # Find valid meld indices (up to 32)
    valid_indices = jnp.where(valid, size=32, fill_value=0)[0]
    n_valid = jnp.sum(valid.astype(jnp.int32))

    # Single meld savings
    best_single = jnp.max(v_points)

    # Pairs of non-overlapping melds
    v_overlap = MELD_OVERLAP[valid_indices][:, valid_indices]
    v_pts = v_points[valid_indices]
    valid_pair = ~v_overlap
    pair_savings = jnp.where(valid_pair, v_pts[:, None] + v_pts[None, :], 0)
    best_pair = jnp.max(pair_savings)

    # Triples
    ov = v_overlap
    valid_triple = ~ov[:, None, :] & ~ov[None, :, :] & ~ov[:, :, None]
    valid_triple = valid_triple & (jnp.arange(32)[:, None, None] < jnp.arange(32)[None, :, None])
    valid_triple = valid_triple & (jnp.arange(32)[None, :, None] < jnp.arange(32)[None, None, :])

    triple_savings = jnp.where(
        valid_triple,
        v_pts[:, None, None] + v_pts[None, :, None] + v_pts[None, None, :],
        0
    )
    best_triple = jnp.max(triple_savings)

    best_savings = jnp.maximum(jnp.maximum(best_single, best_pair), best_triple)
    return total_points - best_savings


@jax.jit
def calculate_deadwood_all_discards(hand):
    """Calculate deadwood for all 52 possible discards using vectorized search.

    Uses precomputed SELECTION_MATRIX to enumerate all meld combinations
    efficiently via dense matrix operations. No branching or variable shapes.

    Returns: (52,) array of deadwood values after discarding each card.
             Cards not in hand get deadwood = 999 (invalid).
    """
    # Card point values
    card_values = jnp.array([CARD_POINTS[i % NUM_RANKS] for i in range(NUM_CARDS)])
    total_hand_points = jnp.sum(hand * card_values)

    # Step 1: Find valid melds for full 11-card hand
    is_valid = valid_melds_mask(hand)  # (185,) bool

    # Step 2: Select top K=24 melds by (validity, points)
    # Sort by validity (True first) then by points (descending)
    # Use negative points so that argsort gives highest first among valid
    sort_key = jnp.where(is_valid, -MELD_POINTS, 1000)
    top_k_indices = jnp.argsort(sort_key)[:MAX_MELDS_IN_HAND]  # (24,)

    # Get masks and properties for top K melds
    top_k_masks = MELD_MASKS[top_k_indices]  # (24, 52)
    top_k_valid = is_valid[top_k_indices]    # (24,) bool
    top_k_points = MELD_POINTS[top_k_indices]  # (24,)

    # Step 3: Compute card usage for all 2325 combinations
    # SELECTION_MATRIX: (2325, 24) binary - which melds to include
    # card_usage: (2325, 52) - count of how many times each card is used
    card_usage = jnp.dot(SELECTION_MATRIX, top_k_masks)  # (2325, 52)

    # Step 4: Check disjointness (no card used twice)
    is_disjoint = jnp.all(card_usage <= 1, axis=1)  # (2325,)

    # Step 5: Check all selected melds are valid
    # Count how many valid melds each combo selected
    selected_valid_count = jnp.dot(SELECTION_MATRIX, top_k_valid.astype(jnp.int32))  # (2325,)
    selected_total_count = jnp.sum(SELECTION_MATRIX, axis=1)  # (2325,)
    all_selected_valid = (selected_valid_count == selected_total_count)  # (2325,)

    # Combined validity for each combo
    combo_valid = is_disjoint & all_selected_valid  # (2325,)

    # Step 6: Compute meld points for each combo
    combo_points = jnp.dot(SELECTION_MATRIX, top_k_points)  # (2325,)

    # Step 7: For all 52 discards at once
    # A combo is valid for discard c if: combo_valid AND combo doesn't use card c
    card_not_used = (card_usage == 0)  # (2325, 52) bool
    valid_for_discard = combo_valid[:, None] & card_not_used  # (2325, 52)

    # Find best meld points for each discard
    # Set invalid combos to -inf so they don't win the max
    combo_points_expanded = combo_points[:, None]  # (2325, 1)
    masked_points = jnp.where(valid_for_discard, combo_points_expanded, -1)  # (2325, 52)
    best_meld_points = jnp.max(masked_points, axis=0)  # (52,)

    # Compute deadwood for each discard
    # deadwood = (total_hand_points - card_value) - best_meld_points
    deadwood = (total_hand_points - card_values) - best_meld_points  # (52,)

    # Mask out cards not in hand
    deadwood = jnp.where(hand > 0, deadwood, 999)

    return deadwood


@jax.jit
def min_deadwood_after_discard(hand):
    """Calculate minimum deadwood after optimal discard (for 11-card hands).

    This matches C++ behavior exactly:
    1. Find best melds for all 11 cards
    2. Compute which cards are deadwood (not in melds)
    3. Discard the highest-value deadwood card
    4. Return sum of remaining deadwood

    This is different from trying every discard - we only discard from deadwood.
    """
    hand_count = jnp.sum(hand)

    # For 10-card hands, just return regular deadwood
    regular_deadwood = calculate_deadwood(hand)

    # For 11-card hands: find best melds, then discard highest deadwood
    # First, find which melds are valid and compute best meld set
    valid = valid_melds_mask(hand)
    v_points = jnp.where(valid, MELD_POINTS, 0)

    # Find valid meld indices (up to 32)
    valid_indices = jnp.where(valid, size=32, fill_value=0)[0]

    # Find best non-overlapping melds (same logic as calculate_deadwood)
    v_overlap = MELD_OVERLAP[valid_indices][:, valid_indices]
    v_pts = v_points[valid_indices]

    # Best single meld
    best_single_idx = jnp.argmax(v_pts)

    # Best pair
    valid_pair = ~v_overlap
    pair_savings = jnp.where(valid_pair, v_pts[:, None] + v_pts[None, :], 0)
    best_pair_flat = jnp.argmax(pair_savings.flatten())
    best_pair_i, best_pair_j = best_pair_flat // 32, best_pair_flat % 32

    # Best triple
    ov = v_overlap
    valid_triple = ~ov[:, None, :] & ~ov[None, :, :] & ~ov[:, :, None]
    valid_triple = valid_triple & (jnp.arange(32)[:, None, None] < jnp.arange(32)[None, :, None])
    valid_triple = valid_triple & (jnp.arange(32)[None, :, None] < jnp.arange(32)[None, None, :])
    triple_savings = jnp.where(
        valid_triple,
        v_pts[:, None, None] + v_pts[None, :, None] + v_pts[None, None, :],
        0
    )
    best_triple_flat = jnp.argmax(triple_savings.flatten())
    best_triple_i = best_triple_flat // (32 * 32)
    best_triple_j = (best_triple_flat // 32) % 32
    best_triple_k = best_triple_flat % 32

    # Determine which configuration is best
    single_val = v_pts[best_single_idx]
    pair_val = jnp.max(pair_savings)
    triple_val = jnp.max(triple_savings)
    best_is_single = (single_val >= pair_val) & (single_val >= triple_val) & (single_val > 0)
    best_is_pair = (pair_val > single_val) & (pair_val >= triple_val)
    best_is_triple = (triple_val > single_val) & (triple_val > pair_val)

    # Build mask of cards in best melds
    meld_cards = jnp.zeros(NUM_CARDS, dtype=jnp.int8)
    # Single meld
    meld_cards = jnp.where(best_is_single, meld_cards | MELD_MASKS[valid_indices[best_single_idx]], meld_cards)
    # Pair melds
    meld_cards = jnp.where(best_is_pair, meld_cards | MELD_MASKS[valid_indices[best_pair_i]] | MELD_MASKS[valid_indices[best_pair_j]], meld_cards)
    # Triple melds
    meld_cards = jnp.where(best_is_triple, meld_cards | MELD_MASKS[valid_indices[best_triple_i]] | MELD_MASKS[valid_indices[best_triple_j]] | MELD_MASKS[valid_indices[best_triple_k]], meld_cards)

    # Deadwood cards = in hand but not in melds
    deadwood_cards = hand & ~meld_cards

    # Find highest value deadwood card to discard
    card_values = jnp.array([CARD_POINTS[i % NUM_RANKS] for i in range(NUM_CARDS)])
    deadwood_values = jnp.where(deadwood_cards > 0, card_values, -1)
    highest_deadwood_idx = jnp.argmax(deadwood_values)
    highest_deadwood_val = deadwood_values[highest_deadwood_idx]

    # Total deadwood = sum of all deadwood values minus the discarded card
    total_deadwood = jnp.sum(deadwood_cards * card_values)
    deadwood_after_discard = total_deadwood - jnp.maximum(highest_deadwood_val, 0)

    # Return for 11 cards; regular for 10
    return jnp.where(hand_count == 11, deadwood_after_discard, regular_deadwood)


# =============================================================================
# Meld layoff helpers
# =============================================================================

def _compute_meld_info():
    is_rank_meld = np.zeros(NUM_MELDS, dtype=np.bool_)
    is_suit_meld = np.zeros(NUM_MELDS, dtype=np.bool_)

    for meld_id, meld in enumerate(ALL_MELDS):
        if meld is None:
            continue
        # C++ encoding: rank = card % 13
        ranks = [c % NUM_RANKS for c in meld]
        if len(set(ranks)) == 1:
            is_rank_meld[meld_id] = True
        else:
            is_suit_meld[meld_id] = True

    return jnp.array(is_rank_meld), jnp.array(is_suit_meld)

IS_RANK_MELD, IS_SUIT_MELD = _compute_meld_info()


def _compute_meld_layoff_info():
    layoff_cards = np.zeros((NUM_MELDS, NUM_CARDS), dtype=np.bool_)

    for meld_id, meld in enumerate(ALL_MELDS):
        if meld is None:
            continue

        if IS_RANK_MELD[meld_id]:
            # Rank meld (set): same rank, different suits
            if len(meld) == 3:
                rank = meld[0] % NUM_RANKS  # C++ encoding
                for suit in range(NUM_SUITS):
                    card = make_card(suit, rank)
                    if card not in meld:
                        layoff_cards[meld_id, card] = True
        else:
            # Suit meld (run): same suit, consecutive ranks
            suit = meld[0] // NUM_RANKS  # C++ encoding
            ranks = sorted([c % NUM_RANKS for c in meld])
            min_rank, max_rank = ranks[0], ranks[-1]
            if min_rank > 0:
                layoff_cards[meld_id, make_card(suit, min_rank - 1)] = True
            if max_rank < NUM_RANKS - 1:
                layoff_cards[meld_id, make_card(suit, max_rank + 1)] = True

    return jnp.array(layoff_cards)

MELD_LAYOFF_CARDS = _compute_meld_layoff_info()


@jax.jit
def compute_layoff_cards(layed_melds_mask, layoffs_mask):
    """Compute valid layoff cards from the knocker's laid melds, including run extensions.

    When a card is laid off onto a run, it extends the run, creating new layoff opportunities.
    For example: if 2-3-4♠ is a meld and 5♠ is laid off, now 6♠ is also valid.
    Sets (rank melds) don't extend - once they have 4 cards, they're full.
    """
    # Base layoffs from original melds
    base_layoffs = jnp.any(MELD_LAYOFF_CARDS * layed_melds_mask[:, None], axis=0)

    # Only consider extensions for cards laid off onto RUN melds (suit melds)
    # First, find which base layoffs came from run melds (not set melds)
    run_meld_layoffs = jnp.any(MELD_LAYOFF_CARDS * (layed_melds_mask & IS_SUIT_MELD)[:, None], axis=0)

    def extend_once(extended, _):
        """One iteration of run extension."""
        # Only extend cards that were both laid off AND were layoffs for run melds
        just_extended = extended & layoffs_mask & run_meld_layoffs
        ext_2d = just_extended.reshape(NUM_SUITS, NUM_RANKS)

        # High extension: rank+1 valid if rank was extended (shift right, clear first col)
        high_ext = jnp.roll(ext_2d, 1, axis=1).at[:, 0].set(False)

        # Low extension: rank-1 valid if rank was extended (shift left, clear last col)
        low_ext = jnp.roll(ext_2d, -1, axis=1).at[:, -1].set(False)

        return extended | (high_ext | low_ext).flatten(), None

    # Iterate extension up to 10 times (max practical run extension depth)
    extended, _ = jax.lax.scan(extend_once, base_layoffs, None, length=10)

    return extended


# =============================================================================
# JAX State Representation - V2 with explicit chance nodes
# =============================================================================

@jax.jit
def init_state():
    """Initialize a new game state ready for dealing (chance nodes)."""
    return {
        # Card locations
        'player0_hand': jnp.zeros(NUM_CARDS, dtype=jnp.int8),
        'player1_hand': jnp.zeros(NUM_CARDS, dtype=jnp.int8),
        'deck': jnp.ones(NUM_CARDS, dtype=jnp.int8),  # All cards in deck initially
        'discard_pile': jnp.zeros(NUM_CARDS, dtype=jnp.int8),
        'upcard': jnp.int8(-1),  # -1 = no upcard

        # Game state
        'current_player': jnp.int8(CHANCE_PLAYER),  # Start with chance node
        'phase': jnp.int8(PHASE_DEAL),
        'deal_cards_dealt': jnp.int8(0),  # 0-20 during deal, then 21 = done
        'waiting_stock_draw': jnp.bool_(False),  # True after draw_stock action

        # Tracking
        'pass_count': jnp.int8(0),
        'drawn_from_discard': jnp.int8(-1),
        'num_draw_upcard': jnp.int32(0),
        'prev_upcard': jnp.int8(-1),  # Upcard before current player's discard
        'repeated_move': jnp.bool_(False),  # For repeated move termination

        # Knock/layoff state
        'knocked': jnp.zeros(2, dtype=jnp.bool_),
        'knocker': jnp.int8(-1),
        'layed_melds': jnp.zeros((2, NUM_MELDS), dtype=jnp.bool_),
        'layoffs_mask': jnp.zeros(NUM_CARDS, dtype=jnp.bool_),
        'finished_layoffs': jnp.bool_(False),
        'deadwood': jnp.zeros(2, dtype=jnp.int32),
        'knocker_deadwood': jnp.int32(-1),  # Knocker's deadwood when entering layoff (-1 = not set)

        # Game outcome
        'done': jnp.bool_(False),
        'winner': jnp.int8(-1),
        'p0_score': jnp.int32(0),
        'p1_score': jnp.int32(0),
    }


@jax.jit
def is_chance_node(state):
    """Return True if current state is a chance node."""
    in_deal = state['phase'] == PHASE_DEAL
    waiting_stock = state['waiting_stock_draw']
    return in_deal | waiting_stock


@jax.jit
def get_hand(state, player):
    """Get hand mask for a player."""
    return jnp.where(player == 0, state['player0_hand'], state['player1_hand'])


# =============================================================================
# Legal Actions - V2
# =============================================================================

@jax.jit
def chance_outcomes_mask(state):
    """Return mask of legal chance outcomes (cards still in deck)."""
    return state['deck'] > 0


@jax.jit
def legal_actions_mask(state):
    """Return mask of legal actions."""
    phase = state['phase']
    player = state['current_player']

    # Initialize mask
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)

    # === Chance nodes: legal actions are cards in deck ===
    is_chance = is_chance_node(state)
    deck_cards = state['deck'] > 0
    mask = jnp.where(is_chance, mask.at[:NUM_CARDS].set(deck_cards), mask)

    # For non-chance nodes:
    hand = get_hand(state, player)
    card_in_hand = hand > 0

    # Calculate deadwood for knock eligibility
    deadwood = calculate_deadwood(hand)
    valid_melds = valid_melds_mask(hand)

    # Stock info
    stock_count = jnp.sum(state['deck'])
    is_wall = stock_count <= WALL_STOCK_SIZE
    has_upcard = state['upcard'] >= 0

    # === Phase: FirstUpcard ===
    first_upcard = (phase == PHASE_FIRST_UPCARD) & ~is_chance
    mask = jnp.where(first_upcard & has_upcard, mask.at[ACTION_DRAW_UPCARD].set(True), mask)
    mask = jnp.where(first_upcard, mask.at[ACTION_PASS].set(True), mask)

    # === Phase: Draw ===
    draw_phase = (phase == PHASE_DRAW) & ~is_chance
    can_draw_upcard = draw_phase & ~is_wall & has_upcard
    can_draw_stock = draw_phase & ~is_wall & (stock_count > 0)
    mask = jnp.where(can_draw_upcard, mask.at[ACTION_DRAW_UPCARD].set(True), mask)
    mask = jnp.where(can_draw_stock, mask.at[ACTION_DRAW_STOCK].set(True), mask)

    # === Phase: Wall ===
    wall_phase = (phase == PHASE_WALL) & ~is_chance
    # In wall phase, knock automatically includes the upcard, so check deadwood with upcard added
    upcard_for_wall = state['upcard']
    hand_with_upcard = jnp.where(upcard_for_wall >= 0, hand.at[upcard_for_wall].set(1), hand)
    deadwood_with_upcard = min_deadwood_after_discard(hand_with_upcard)  # 11 cards, need to discard
    can_knock_wall = wall_phase & (deadwood_with_upcard <= KNOCK_THRESHOLD)
    mask = jnp.where(wall_phase, mask.at[ACTION_PASS].set(True), mask)
    mask = jnp.where(can_knock_wall, mask.at[ACTION_KNOCK].set(True), mask)

    # === Phase: Discard ===
    discard_phase = (phase == PHASE_DISCARD) & ~is_chance

    # For knock eligibility with 11 cards, C++ calculates deadwood AFTER optimal discard
    min_dw_after_discard = min_deadwood_after_discard(hand)
    can_knock_discard = discard_phase & (min_dw_after_discard <= KNOCK_THRESHOLD)

    # Can discard any card in hand (C++ has no restriction on drawing and discarding same card)
    mask = jnp.where(discard_phase, mask.at[:NUM_CARDS].set(card_in_hand), mask)
    mask = jnp.where(can_knock_discard, mask.at[ACTION_KNOCK].set(True), mask)

    # === Phase: Knock ===
    knock_phase = (phase == PHASE_KNOCK) & ~is_chance
    hand_count = jnp.sum(hand)
    has_11_cards = hand_count == 11

    # With 11 cards: can only discard cards that keep deadwood <= KNOCK_THRESHOLD
    # For each card, check if removing it gives valid knock deadwood
    # Optimized: compute deadwood for all discards at once (computes valid_melds ONCE)
    all_deadwoods = calculate_deadwood_all_discards(hand)
    knock_discard_mask = (hand > 0) & (all_deadwoods <= KNOCK_THRESHOLD)
    mask = jnp.where(knock_phase & has_11_cards, mask.at[:NUM_CARDS].set(knock_discard_mask), mask)

    # With 10 or fewer cards: can lay valid melds
    # Rule: can pass once remaining hand's total points <= KNOCK_THRESHOLD (10)
    # (Cards are removed from hand when melds are laid, remaining cards are all deadwood)
    knock_10 = knock_phase & (hand_count <= 10)
    meld_mask = knock_10 & valid_melds
    # Calculate raw point total of remaining cards (no meld optimization - these ARE the deadwood)
    remaining_points = hand_total_points(hand)
    can_pass_knock = remaining_points <= KNOCK_THRESHOLD
    # Allow pass if remaining points are low enough
    mask = jnp.where(knock_10 & can_pass_knock, mask.at[ACTION_PASS].set(True), mask)
    mask = jnp.where(knock_10, mask.at[ACTION_MELD_BASE:ACTION_MELD_BASE + NUM_MELDS].set(meld_mask), mask)

    # === Phase: Layoff ===
    layoff_phase = (phase == PHASE_LAYOFF) & ~is_chance
    finished = state['finished_layoffs']
    knocker = state['knocker']
    safe_knocker = jnp.where(knocker >= 0, knocker, 0)
    knocker_melds = state['layed_melds'][safe_knocker]

    # If knocker has gin (0 deadwood), opponent can't lay off - skip to meld phase
    # Use the scalar knocker_deadwood field set when entering layoff phase
    knocker_deadwood = state['knocker_deadwood']
    is_gin = knocker_deadwood == 0
    finished = finished | is_gin  # Treat gin as if layoffs are finished

    layoff_possible = compute_layoff_cards(knocker_melds, state['layoffs_mask'])

    # Before finishing layoffs: can lay off cards or pass
    pre_finish = layoff_phase & ~finished
    layoff_card_mask = card_in_hand & layoff_possible
    mask = jnp.where(pre_finish, mask.at[:NUM_CARDS].set(layoff_card_mask), mask)
    mask = jnp.where(pre_finish, mask.at[ACTION_PASS].set(True), mask)

    # After finishing layoffs: can lay own melds or pass
    post_finish = layoff_phase & finished
    mask = jnp.where(post_finish, mask.at[ACTION_PASS].set(True), mask)
    meld_mask_layoff = post_finish & valid_melds
    mask = jnp.where(post_finish, mask.at[ACTION_MELD_BASE:ACTION_MELD_BASE + NUM_MELDS].set(meld_mask_layoff), mask)

    return mask


# =============================================================================
# State Transitions - V2
# =============================================================================

@jax.jit
def step(state, action):
    """Apply action and return new state."""
    phase = state['phase']
    player = state['current_player']
    is_chance = is_chance_node(state)

    # Copy all state fields
    new_state = {k: v for k, v in state.items()}

    # === Handle chance nodes ===

    # Deal phase: action is which card to deal
    in_deal = (phase == PHASE_DEAL) & is_chance
    deal_count = state['deal_cards_dealt']

    # First 10 cards go to player 0
    to_p0 = in_deal & (deal_count < 10)
    new_p0_hand = jnp.where(to_p0, state['player0_hand'].at[action].set(1), state['player0_hand'])

    # Next 10 cards go to player 1
    to_p1 = in_deal & (deal_count >= 10) & (deal_count < 20)
    new_p1_hand = jnp.where(to_p1, state['player1_hand'].at[action].set(1), state['player1_hand'])

    # Card 21 is upcard
    to_upcard = in_deal & (deal_count == 20)
    new_upcard = jnp.where(to_upcard, jnp.int8(action), state['upcard'])

    # Remove card from deck
    new_deck = jnp.where(in_deal, state['deck'].at[action].set(0), state['deck'])

    # Update deal count
    new_deal_count = jnp.where(in_deal, deal_count + 1, deal_count)

    # Check if dealing is complete (21 cards dealt)
    deal_complete = in_deal & (new_deal_count >= 21)
    new_phase = jnp.where(deal_complete, jnp.int8(PHASE_FIRST_UPCARD), phase)
    new_player = jnp.where(deal_complete, jnp.int8(0), player)
    # Set prev_upcard to the dealt upcard when deal completes
    new_prev_upcard = jnp.where(deal_complete, jnp.int8(action), state['prev_upcard'])

    # Keep as chance player during deal
    new_player = jnp.where(in_deal & ~deal_complete, jnp.int8(CHANCE_PLAYER), new_player)

    # Stock draw chance node: action is which card player gets
    stock_draw = state['waiting_stock_draw'] & is_chance
    hand = get_hand(state, player)
    drawn_hand = hand.at[action].set(1)
    new_p0_hand = jnp.where(stock_draw & (player == 0), drawn_hand, new_p0_hand)
    new_p1_hand = jnp.where(stock_draw & (player == 1), drawn_hand, new_p1_hand)
    new_deck = jnp.where(stock_draw, new_deck.at[action].set(0), new_deck)
    new_waiting = jnp.where(stock_draw, False, state['waiting_stock_draw'])
    new_phase = jnp.where(stock_draw, jnp.int8(PHASE_DISCARD), new_phase)
    # After stock draw, drawn_from_discard = -1 (can discard anything)
    new_drawn_from_discard = jnp.where(stock_draw, jnp.int8(-1), state['drawn_from_discard'])

    # Update state with chance node changes
    new_state['player0_hand'] = new_p0_hand
    new_state['player1_hand'] = new_p1_hand
    new_state['deck'] = new_deck
    new_state['upcard'] = new_upcard
    new_state['deal_cards_dealt'] = new_deal_count
    new_state['waiting_stock_draw'] = new_waiting
    new_state['phase'] = new_phase
    new_state['current_player'] = new_player
    new_state['drawn_from_discard'] = new_drawn_from_discard
    new_state['prev_upcard'] = new_prev_upcard

    # === Handle regular (non-chance) actions ===
    not_chance = ~is_chance

    # --- FirstUpcard phase ---
    first_upcard = (phase == PHASE_FIRST_UPCARD) & not_chance

    # Pass in first upcard
    pass_first = first_upcard & (action == ACTION_PASS)
    new_pass_count = jnp.where(pass_first, state['pass_count'] + 1, state['pass_count'])
    both_passed = new_pass_count >= 2
    # After both pass, upcard goes to discard and player 0 draws
    new_phase = jnp.where(pass_first & both_passed, jnp.int8(PHASE_DRAW), new_state['phase'])
    new_player = jnp.where(pass_first & both_passed, jnp.int8(0), new_state['current_player'])
    new_upcard_after_pass = jnp.where(pass_first & both_passed, jnp.int8(-1), new_state['upcard'])
    # Add upcard to discard pile when both pass
    upcard_val = state['upcard']
    new_discard = jnp.where(
        pass_first & both_passed & (upcard_val >= 0),
        state['discard_pile'].at[upcard_val].set(1),
        state['discard_pile']
    )
    # Switch player if only first pass
    new_player = jnp.where(pass_first & ~both_passed, jnp.int8(1 - player), new_player)

    # Draw upcard in first upcard
    draw_first = first_upcard & (action == ACTION_DRAW_UPCARD)
    upcard_card = state['upcard']
    hand = get_hand(state, player)
    hand_with_upcard = hand.at[upcard_card].set(1)
    new_p0_hand_draw = jnp.where(draw_first & (player == 0), hand_with_upcard, new_state['player0_hand'])
    new_p1_hand_draw = jnp.where(draw_first & (player == 1), hand_with_upcard, new_state['player1_hand'])
    new_upcard_draw = jnp.where(draw_first, jnp.int8(-1), new_upcard_after_pass)
    new_phase_draw = jnp.where(draw_first, jnp.int8(PHASE_DISCARD), new_phase)
    new_drawn_from_discard = jnp.where(draw_first, jnp.int8(upcard_card), new_state['drawn_from_discard'])
    # Count upcard draws in first_upcard phase too
    new_num_upcard_first = jnp.where(draw_first, state['num_draw_upcard'] + 1, state['num_draw_upcard'])

    new_state['pass_count'] = new_pass_count
    new_state['phase'] = jnp.where(draw_first, new_phase_draw, new_phase)
    new_state['current_player'] = new_player
    new_state['upcard'] = jnp.where(draw_first, new_upcard_draw, new_upcard_after_pass)
    new_state['discard_pile'] = new_discard
    new_state['player0_hand'] = new_p0_hand_draw
    new_state['player1_hand'] = new_p1_hand_draw
    new_state['drawn_from_discard'] = new_drawn_from_discard
    new_state['num_draw_upcard'] = new_num_upcard_first

    # --- Draw phase ---
    draw_phase = (phase == PHASE_DRAW) & not_chance

    # Draw from stock -> becomes chance node
    draw_stock = draw_phase & (action == ACTION_DRAW_STOCK)
    new_state['waiting_stock_draw'] = jnp.where(draw_stock, True, new_state['waiting_stock_draw'])
    # current_player stays the same, will be used in stock draw chance node

    # Draw upcard in draw phase
    draw_up = draw_phase & (action == ACTION_DRAW_UPCARD)
    new_num_upcard = jnp.where(draw_up, new_state['num_draw_upcard'] + 1, new_state['num_draw_upcard'])
    too_many = new_num_upcard >= MAX_NUM_DRAW_UPCARD_ACTIONS

    # If too many upcard draws, game ends
    new_state['done'] = jnp.where(draw_up & too_many, True, new_state['done'])
    new_state['winner'] = jnp.where(draw_up & too_many, jnp.int8(2), new_state['winner'])
    new_state['phase'] = jnp.where(draw_up & too_many, jnp.int8(PHASE_GAME_OVER), new_state['phase'])

    # Otherwise draw the upcard
    draw_up_ok = draw_up & ~too_many
    upcard_d = state['upcard']
    hand_d = get_hand(state, player)
    hand_with_up = hand_d.at[upcard_d].set(1)
    new_state['player0_hand'] = jnp.where(draw_up_ok & (player == 0), hand_with_up, new_state['player0_hand'])
    new_state['player1_hand'] = jnp.where(draw_up_ok & (player == 1), hand_with_up, new_state['player1_hand'])
    new_state['upcard'] = jnp.where(draw_up_ok, jnp.int8(-1), new_state['upcard'])
    new_state['phase'] = jnp.where(draw_up_ok, jnp.int8(PHASE_DISCARD), new_state['phase'])
    new_state['drawn_from_discard'] = jnp.where(draw_up_ok, jnp.int8(upcard_d), new_state['drawn_from_discard'])
    new_state['num_draw_upcard'] = new_num_upcard

    # --- Discard phase ---
    discard_phase = (phase == PHASE_DISCARD) & not_chance
    is_discard = discard_phase & (action < NUM_CARDS)
    is_knock = discard_phase & (action == ACTION_KNOCK)

    # Discard a card
    hand_disc = get_hand(new_state, player)
    hand_after_disc = hand_disc.at[action].set(0)
    new_state['player0_hand'] = jnp.where(is_discard & (player == 0), hand_after_disc, new_state['player0_hand'])
    new_state['player1_hand'] = jnp.where(is_discard & (player == 1), hand_after_disc, new_state['player1_hand'])
    # Card becomes new upcard
    new_state['upcard'] = jnp.where(is_discard, jnp.int8(action), new_state['upcard'])
    # Old upcard goes to discard pile
    old_upcard = state['upcard']
    new_state['discard_pile'] = jnp.where(
        is_discard & (old_upcard >= 0),
        new_state['discard_pile'].at[old_upcard].set(1),
        new_state['discard_pile']
    )

    # Check for repeated move termination (discarding same card as prev_upcard)
    prev_up = state['prev_upcard']
    is_repeated = is_discard & (action == prev_up)
    was_repeated = state['repeated_move']
    # If this is the second repeated move, game ends
    repeated_game_end = is_repeated & was_repeated
    new_state['done'] = jnp.where(repeated_game_end, True, new_state['done'])
    new_state['winner'] = jnp.where(repeated_game_end, jnp.int8(2), new_state['winner'])  # Draw
    new_state['phase'] = jnp.where(repeated_game_end, jnp.int8(PHASE_GAME_OVER), new_state['phase'])
    # If this is the first repeated move, set flag
    new_state['repeated_move'] = jnp.where(is_repeated & ~was_repeated, True, new_state['repeated_move'])
    # Reset repeated_move if not a repeated move
    new_state['repeated_move'] = jnp.where(is_discard & ~is_repeated, False, new_state['repeated_move'])

    # Check stock for wall phase (only if game didn't end from repeated move)
    stock_count = jnp.sum(new_state['deck'])
    enter_wall = is_discard & (stock_count <= WALL_STOCK_SIZE) & ~repeated_game_end
    new_state['phase'] = jnp.where(is_discard & enter_wall & ~repeated_game_end, jnp.int8(PHASE_WALL), new_state['phase'])
    new_state['phase'] = jnp.where(is_discard & ~enter_wall & ~repeated_game_end, jnp.int8(PHASE_DRAW), new_state['phase'])
    new_state['current_player'] = jnp.where(is_discard & ~repeated_game_end, jnp.int8(1 - player), new_state['current_player'])
    new_state['drawn_from_discard'] = jnp.where(is_discard, jnp.int8(-1), new_state['drawn_from_discard'])
    # Set prev_upcard for next player's turn (the card we just discarded is now the upcard)
    new_state['prev_upcard'] = jnp.where(is_discard & ~repeated_game_end, jnp.int8(action), new_state['prev_upcard'])

    # Knock action -> go to knock phase
    new_state['phase'] = jnp.where(is_knock, jnp.int8(PHASE_KNOCK), new_state['phase'])
    new_state['knocker'] = jnp.where(is_knock, player, new_state['knocker'])
    new_state['knocked'] = jnp.where(is_knock, new_state['knocked'].at[player].set(True), new_state['knocked'])

    # --- Wall phase ---
    wall_phase = (phase == PHASE_WALL) & not_chance
    wall_pass = wall_phase & (action == ACTION_PASS)
    wall_knock = wall_phase & (action == ACTION_KNOCK)

    # Pass in wall -> game ends immediately (single pass ends the game in wall phase)
    new_state['done'] = jnp.where(wall_pass, True, new_state['done'])
    new_state['winner'] = jnp.where(wall_pass, jnp.int8(2), new_state['winner'])  # Draw
    new_state['phase'] = jnp.where(wall_pass, jnp.int8(PHASE_GAME_OVER), new_state['phase'])

    # Knock in wall - automatically add upcard to hand first
    wall_upcard = state['upcard']
    hand_wall = get_hand(new_state, player)
    hand_with_wall_up = hand_wall.at[wall_upcard].set(1)
    new_state['player0_hand'] = jnp.where(wall_knock & (player == 0), hand_with_wall_up, new_state['player0_hand'])
    new_state['player1_hand'] = jnp.where(wall_knock & (player == 1), hand_with_wall_up, new_state['player1_hand'])
    new_state['upcard'] = jnp.where(wall_knock, jnp.int8(-1), new_state['upcard'])
    new_state['phase'] = jnp.where(wall_knock, jnp.int8(PHASE_KNOCK), new_state['phase'])
    new_state['knocker'] = jnp.where(wall_knock, player, new_state['knocker'])
    new_state['knocked'] = jnp.where(wall_knock, new_state['knocked'].at[player].set(True), new_state['knocked'])

    # NOTE: Knocker's deadwood is NOT computed at knock time.
    # It's computed when entering layoff phase (see knock_pass handling below).
    # At that point, the knocker has already discarded and laid their melds,
    # so we can compute their actual remaining deadwood.

    # --- Knock phase ---
    knock_phase = (phase == PHASE_KNOCK) & not_chance
    knock_discard = knock_phase & (action < NUM_CARDS)
    knock_meld = knock_phase & (action >= ACTION_MELD_BASE) & (action < ACTION_MELD_BASE + NUM_MELDS)
    knock_pass = knock_phase & (action == ACTION_PASS)

    # Discard in knock (when have 11 cards)
    hand_k = get_hand(new_state, player)
    hand_after_k = hand_k.at[action].set(0)
    new_state['player0_hand'] = jnp.where(knock_discard & (player == 0), hand_after_k, new_state['player0_hand'])
    new_state['player1_hand'] = jnp.where(knock_discard & (player == 1), hand_after_k, new_state['player1_hand'])
    # Discarded card becomes upcard
    new_state['upcard'] = jnp.where(knock_discard, jnp.int8(action), new_state['upcard'])
    old_up = new_state['upcard']
    new_state['discard_pile'] = jnp.where(
        knock_discard & (state['upcard'] >= 0),
        new_state['discard_pile'].at[state['upcard']].set(1),
        new_state['discard_pile']
    )

    # Lay meld in knock - mark meld as laid AND remove cards from hand
    meld_id = action - ACTION_MELD_BASE
    safe_meld_id = jnp.clip(meld_id, 0, NUM_MELDS - 1)
    new_state['layed_melds'] = jnp.where(
        knock_meld,
        new_state['layed_melds'].at[player, safe_meld_id].set(True),
        new_state['layed_melds']
    )
    # Remove meld cards from hand
    meld_cards = MELD_MASKS[safe_meld_id]
    hand_after_meld_k = get_hand(new_state, player) - meld_cards
    new_state['player0_hand'] = jnp.where(knock_meld & (player == 0), hand_after_meld_k, new_state['player0_hand'])
    new_state['player1_hand'] = jnp.where(knock_meld & (player == 1), hand_after_meld_k, new_state['player1_hand'])

    # Pass in knock -> go to layoff
    # Compute knocker's final deadwood (after discarding and laying melds)
    # The remaining cards ARE deadwood - just sum their values, don't look for melds
    # (Any melds the knocker didn't lay are their choice, remaining cards are deadwood)
    knocker_final_hand = get_hand(new_state, player)
    knocker_final_dw = hand_total_points(knocker_final_hand)
    # Store knocker's deadwood in a scalar field to avoid JAX JIT array issues
    new_state['knocker_deadwood'] = jnp.where(knock_pass, knocker_final_dw, new_state['knocker_deadwood'])

    new_state['phase'] = jnp.where(knock_pass, jnp.int8(PHASE_LAYOFF), new_state['phase'])
    new_state['current_player'] = jnp.where(knock_pass, jnp.int8(1 - player), new_state['current_player'])

    # --- Layoff phase ---
    layoff_phase = (phase == PHASE_LAYOFF) & not_chance
    finished = state['finished_layoffs']
    # If knocker has gin (0 deadwood), treat as finished (no layoffs allowed, skip to meld phase)
    knocker_deadwood_l = state['knocker_deadwood']
    is_gin_l = knocker_deadwood_l == 0
    finished = finished | is_gin_l
    layoff_card = layoff_phase & ~finished & (action < NUM_CARDS)
    layoff_pass_pre = layoff_phase & ~finished & (action == ACTION_PASS)
    layoff_meld = layoff_phase & finished & (action >= ACTION_MELD_BASE) & (action < ACTION_MELD_BASE + NUM_MELDS)
    layoff_pass_post = layoff_phase & finished & (action == ACTION_PASS)

    # Lay off card onto knocker's melds
    hand_l = get_hand(new_state, player)
    hand_after_l = hand_l.at[action].set(0)
    new_state['player0_hand'] = jnp.where(layoff_card & (player == 0), hand_after_l, new_state['player0_hand'])
    new_state['player1_hand'] = jnp.where(layoff_card & (player == 1), hand_after_l, new_state['player1_hand'])
    new_state['layoffs_mask'] = jnp.where(layoff_card, new_state['layoffs_mask'].at[action].set(True), new_state['layoffs_mask'])

    # Pass before finishing -> done with layoffs
    new_state['finished_layoffs'] = jnp.where(layoff_pass_pre, True, new_state['finished_layoffs'])

    # Lay own meld - mark meld as laid AND remove cards from hand
    meld_id_l = action - ACTION_MELD_BASE
    safe_meld_id_l = jnp.clip(meld_id_l, 0, NUM_MELDS - 1)
    new_state['layed_melds'] = jnp.where(
        layoff_meld,
        new_state['layed_melds'].at[player, safe_meld_id_l].set(True),
        new_state['layed_melds']
    )
    # Remove meld cards from hand
    meld_cards_l = MELD_MASKS[safe_meld_id_l]
    hand_after_meld_l = get_hand(new_state, player) - meld_cards_l
    new_state['player0_hand'] = jnp.where(layoff_meld & (player == 0), hand_after_meld_l, new_state['player0_hand'])
    new_state['player1_hand'] = jnp.where(layoff_meld & (player == 1), hand_after_meld_l, new_state['player1_hand'])

    # Pass after finishing -> calculate scores and end game
    knocker_p = new_state['knocker']
    opponent = 1 - knocker_p
    knocker_hand = jnp.where(knocker_p == 0, new_state['player0_hand'], new_state['player1_hand'])
    opponent_hand = jnp.where(opponent == 0, new_state['player0_hand'], new_state['player1_hand'])

    knocker_dw = calculate_deadwood(knocker_hand)
    opponent_dw = calculate_deadwood(opponent_hand)

    is_gin = knocker_dw == 0
    is_undercut = opponent_dw <= knocker_dw

    # Scoring
    gin_score = opponent_dw + GIN_BONUS
    undercut_score = knocker_dw - opponent_dw + UNDERCUT_BONUS
    normal_score = knocker_dw - opponent_dw

    knocker_wins = ~is_gin & ~is_undercut
    opponent_wins = is_undercut & ~is_gin

    final_score = jnp.where(is_gin, gin_score,
                  jnp.where(is_undercut, -undercut_score, -normal_score))

    p0_score = jnp.where(knocker_p == 0, final_score, -final_score)

    new_state['deadwood'] = jnp.where(layoff_pass_post,
        new_state['deadwood'].at[knocker_p].set(knocker_dw).at[opponent].set(opponent_dw),
        new_state['deadwood'])
    new_state['p0_score'] = jnp.where(layoff_pass_post, p0_score, new_state['p0_score'])
    new_state['p1_score'] = jnp.where(layoff_pass_post, -p0_score, new_state['p1_score'])
    new_state['done'] = jnp.where(layoff_pass_post, True, new_state['done'])
    winner = jnp.where(p0_score > 0, 0, jnp.where(p0_score < 0, 1, 2))
    new_state['winner'] = jnp.where(layoff_pass_post, jnp.int8(winner), new_state['winner'])
    new_state['phase'] = jnp.where(layoff_pass_post, jnp.int8(PHASE_GAME_OVER), new_state['phase'])

    return new_state


@jax.jit
def get_returns(state):
    """Get returns for both players."""
    return jnp.array([state['p0_score'], state['p1_score']], dtype=jnp.float32)


# =============================================================================
# OpenSpiel Wrapper
# =============================================================================

_GAME_TYPE = pyspiel.GameType(
    short_name="python_gin_rummy_jax",
    long_name="Python Gin Rummy JAX",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=NUM_ACTIONS,
    max_chance_outcomes=NUM_CARDS,
    num_players=2,
    min_utility=-100.0,
    max_utility=100.0,
    utility_sum=0.0,
    max_game_length=300,
)


class GinRummyJaxStateV2(pyspiel.State):
    """OpenSpiel state wrapper around JAX state."""

    def __init__(self, game, jax_state=None):
        super().__init__(game)
        if jax_state is None:
            self._jax_state = init_state()
        else:
            self._jax_state = jax_state
        self._cached_legal_actions = None

    def current_player(self):
        if self._jax_state['done']:
            return pyspiel.PlayerId.TERMINAL
        if is_chance_node(self._jax_state):
            return pyspiel.PlayerId.CHANCE
        return int(self._jax_state['current_player'])

    def is_chance_node(self):
        return bool(is_chance_node(self._jax_state)) and not self._jax_state['done']

    def chance_outcomes(self):
        """Return list of (action, probability) pairs for chance node."""
        mask = chance_outcomes_mask(self._jax_state)
        mask_np = np.asarray(mask)
        legal_cards = np.where(mask_np)[0]
        n = len(legal_cards)
        if n == 0:
            return []
        prob = 1.0 / n
        return [(int(card), prob) for card in legal_cards]

    def legal_actions(self, player=None):
        if self._cached_legal_actions is None:
            mask = legal_actions_mask(self._jax_state)
            mask_np = np.asarray(mask)
            self._cached_legal_actions = list(np.where(mask_np)[0])
        return self._cached_legal_actions

    def legal_actions_mask(self, player=None):
        mask = legal_actions_mask(self._jax_state)
        return list(np.asarray(mask, dtype=np.int32))

    def _apply_action(self, action):
        self._jax_state = step(self._jax_state, action)
        self._cached_legal_actions = None

    def is_terminal(self):
        return bool(self._jax_state['done'])

    def returns(self):
        returns = get_returns(self._jax_state)
        return [float(returns[0]), float(returns[1])]

    def player_return(self, player):
        return self.returns()[player]

    def __str__(self):
        p0_hand = self._jax_state['player0_hand']
        p1_hand = self._jax_state['player1_hand']
        upcard = int(self._jax_state['upcard'])
        phase = int(self._jax_state['phase'])
        player = int(self._jax_state['current_player'])

        phase_names = ['Deal', 'FirstUpcard', 'Draw', 'Discard', 'Knock', 'Layoff', 'Wall', 'GameOver']

        def cards_str(mask):
            cards = [i for i in range(NUM_CARDS) if mask[i]]
            return ' '.join(card_name(c) for c in cards)

        def card_name(c):
            if c < 0:
                return 'XX'
            ranks = 'A23456789TJQK'
            suits = 'scdh'
            # C++ encoding: card = suit * 13 + rank
            suit = c // 13
            rank = c % 13
            return ranks[rank] + suits[suit]

        lines = [
            f"Phase: {phase_names[phase]}, Player: {player}",
            f"Upcard: {card_name(upcard)}",
            f"P0 hand: {cards_str(p0_hand)}",
            f"P1 hand: {cards_str(p1_hand)}",
            f"Deck size: {int(jnp.sum(self._jax_state['deck']))}",
        ]
        return '\n'.join(lines)

    def clone(self):
        cloned = GinRummyJaxStateV2(self.get_game())
        cloned._jax_state = jax.tree.map(lambda x: x.copy(), self._jax_state)
        return cloned


class GinRummyJaxGameV2(pyspiel.Game):
    """OpenSpiel game wrapper."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})

    def new_initial_state(self):
        return GinRummyJaxStateV2(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return None


# Register the game
pyspiel.register_game(_GAME_TYPE, GinRummyJaxGameV2)


if __name__ == "__main__":
    # Quick test
    game = pyspiel.load_game("python_gin_rummy_jax")
    state = game.new_initial_state()

    print("Initial state (before deal):")
    print(state)
    print(f"Is chance: {state.is_chance_node()}")
    print(f"Chance outcomes: {len(state.chance_outcomes())} cards")

    # Deal all 21 cards
    import random
    random.seed(42)
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        action, _ = random.choice(outcomes)
        state.apply_action(action)

    print("\nAfter dealing:")
    print(state)
    print(f"Current player: {state.current_player()}")
    print(f"Legal actions: {state.legal_actions()}")

"""JAX-accelerated Gin Rummy for OpenSpiel.

This implements Gin Rummy with a JAX core for fast batched simulation,
wrapped in an OpenSpiel-compatible Python game interface.

Card encoding: 0-51 where card = rank * 4 + suit
  Ranks: A=0, 2=1, ..., K=12
  Suits: spades=0, clubs=1, diamonds=2, hearts=3
"""

from typing import List, Optional
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import pyspiel

# =============================================================================
# Constants
# =============================================================================
NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4
HAND_SIZE = 10
MAX_DECK_SIZE = 52
KNOCK_THRESHOLD = 10

# Card point values (A=1, 2-9=face, 10/J/Q/K=10)
CARD_POINTS = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=jnp.int32)

# Game phases
PHASE_DEAL = 0
PHASE_FIRST_UPCARD = 1
PHASE_DRAW = 2
PHASE_DISCARD = 3
PHASE_KNOCK = 4
PHASE_LAYOFF = 5
PHASE_GAME_OVER = 6

# Actions
ACTION_DRAW_UPCARD = 52
ACTION_DRAW_STOCK = 53
ACTION_PASS = 54
ACTION_KNOCK = 55
# Actions 56-240 are meld declarations (we'll compute these)

NUM_ACTIONS = 241  # 0-51 cards, 52-55 special, 56-240 melds (185 melds)

# =============================================================================
# Meld encoding
# =============================================================================
# Melds are combinations of 3+ cards that form sets (same rank) or runs (consecutive same suit)
# We pre-compute all possible melds

def generate_all_melds():
    """Generate all valid melds (sets and runs).

    Matches C++ encoding: 185 melds total
    - 65 rank melds (sets): 52 three-of-a-kind + 13 four-of-a-kind
    - 120 suit melds (runs): 44 size-3 + 40 size-4 + 36 size-5

    C++ MeldToInt encoding:
    - Rank meld size 3: rank * 5 + missing_suit (0-51)
    - Rank meld size 4: rank * 5 + 4 (52-64)
    - Suit meld size 3: 65 + suit * 11 + start_rank (65-108)
    - Suit meld size 4: 109 + suit * 10 + start_rank (109-148)
    - Suit meld size 5: 149 + suit * 9 + start_rank (149-184)
    """
    melds = [None] * 185  # Pre-allocate to match C++ indexing

    # Rank melds (sets) - indices 0-64
    for rank in range(NUM_RANKS):
        # Size 3: missing one suit
        for missing_suit in range(NUM_SUITS):
            meld_id = rank * 5 + missing_suit
            cards = tuple(rank * NUM_SUITS + s for s in range(NUM_SUITS) if s != missing_suit)
            melds[meld_id] = cards
        # Size 4: all suits
        meld_id = rank * 5 + 4
        cards = tuple(rank * NUM_SUITS + s for s in range(NUM_SUITS))
        melds[meld_id] = cards

    # Suit melds (runs) - indices 65-184
    offset = 65
    # Size 3 runs: 11 per suit (start ranks 0-10)
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 2):  # 0-10
            meld_id = offset + suit * 11 + start_rank
            cards = tuple(r * NUM_SUITS + suit for r in range(start_rank, start_rank + 3))
            melds[meld_id] = cards

    offset = 109  # 65 + 44
    # Size 4 runs: 10 per suit (start ranks 0-9)
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 3):  # 0-9
            meld_id = offset + suit * 10 + start_rank
            cards = tuple(r * NUM_SUITS + suit for r in range(start_rank, start_rank + 4))
            melds[meld_id] = cards

    offset = 149  # 109 + 40
    # Size 5 runs: 9 per suit (start ranks 0-8)
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS - 4):  # 0-8
            meld_id = offset + suit * 9 + start_rank
            cards = tuple(r * NUM_SUITS + suit for r in range(start_rank, start_rank + 5))
            melds[meld_id] = cards

    return melds

ALL_MELDS = generate_all_melds()
NUM_MELDS = len(ALL_MELDS)

# Create meld lookup: action 56 + meld_idx
def meld_to_mask(meld):
    """Convert meld tuple to 52-bit mask."""
    mask = np.zeros(NUM_CARDS, dtype=np.int8)
    for card in meld:
        mask[card] = 1
    return mask

MELD_MASKS = jnp.array([meld_to_mask(m) for m in ALL_MELDS], dtype=jnp.int8)


# =============================================================================
# Card utilities
# =============================================================================

def card_rank(card):
    """Get rank (0-12) from card (0-51)."""
    return card // NUM_SUITS


def card_suit(card):
    """Get suit (0-3) from card (0-51)."""
    return card % NUM_SUITS


def card_points_fn(card):
    """Get point value of a card."""
    return CARD_POINTS[card_rank(card)]


# =============================================================================
# Deadwood calculation (the heart of Gin Rummy scoring)
# =============================================================================

# Precompute meld overlap matrix (1 = overlapping, 0 = compatible)
# Vectorized: two melds overlap if they share any card
_meld_masks_np = np.array(MELD_MASKS)  # Convert once to numpy
_meld_overlap = np.zeros((NUM_MELDS, NUM_MELDS), dtype=np.int8)
for _i in range(NUM_MELDS):
    for _j in range(NUM_MELDS):
        _meld_overlap[_i, _j] = 1 if np.any(_meld_masks_np[_i] & _meld_masks_np[_j]) else 0
MELD_OVERLAP = jnp.array(_meld_overlap)
MELD_COMPAT = 1 - MELD_OVERLAP

# Precompute meld point values
MELD_POINTS = jnp.array([
    sum(int(CARD_POINTS[c // NUM_SUITS]) for c in meld)
    for meld in ALL_MELDS
], dtype=jnp.int32)

# Max valid melds per hand (for static sizing in JAX)
MAX_VALID_MELDS = 32


@jax.jit
def hand_total_points(hand_mask):
    """Calculate total points of all cards in hand."""
    card_pts = CARD_POINTS[jnp.arange(NUM_CARDS) // NUM_SUITS]
    return jnp.sum(hand_mask * card_pts)


@jax.jit
def valid_melds_mask(hand_mask):
    """Return boolean mask of which melds are valid for this hand."""
    return jnp.all(MELD_MASKS <= hand_mask, axis=1)


@jax.jit
def calculate_deadwood(hand_mask):
    """
    Calculate minimum deadwood using optimized matrix operations.

    Uses exhaustive search over all combinations of 1, 2, 3 non-overlapping melds.
    Optimized for GPU with full matrix operations.
    """
    total_points = hand_total_points(hand_mask)
    valid = valid_melds_mask(hand_mask)

    # No melds = all deadwood
    best_savings = jnp.int32(0)

    # Get indices of valid melds, padded with index 0
    valid_indices = jnp.where(valid, size=MAX_VALID_MELDS, fill_value=0)[0]
    num_valid = jnp.sum(valid)
    valid_mask = jnp.arange(MAX_VALID_MELDS) < num_valid

    # Get properties for valid melds
    v_points = MELD_POINTS[valid_indices]

    # === Single melds: vectorized max ===
    single_savings = jnp.where(valid_mask, v_points, 0)
    best_savings = jnp.maximum(best_savings, jnp.max(single_savings))

    # === Build compatibility matrix for valid melds ===
    v_compat = MELD_COMPAT[valid_indices][:, valid_indices]

    # === Pairs: matrix operations ===
    upper_tri_2 = jnp.triu(jnp.ones((MAX_VALID_MELDS, MAX_VALID_MELDS), dtype=jnp.bool_), k=1)
    valid_pair = (valid_mask[:, None] & valid_mask[None, :] &
                  v_compat.astype(jnp.bool_) & upper_tri_2)
    pair_savings = jnp.where(valid_pair, v_points[:, None] + v_points[None, :], 0)
    best_savings = jnp.maximum(best_savings, jnp.max(pair_savings))

    # === Triples: 3D tensor operations ===
    idx = jnp.arange(MAX_VALID_MELDS)
    i_lt_j = idx[:, None, None] < idx[None, :, None]
    j_lt_k = idx[None, :, None] < idx[None, None, :]
    upper_tri_3 = i_lt_j & j_lt_k

    all_valid = (valid_mask[:, None, None] &
                 valid_mask[None, :, None] &
                 valid_mask[None, None, :])

    compat_ij = v_compat[:, :, None].astype(jnp.bool_)
    compat_ik = v_compat[:, None, :].astype(jnp.bool_)
    compat_jk = v_compat[None, :, :].astype(jnp.bool_)
    all_compat = compat_ij & compat_ik & compat_jk

    valid_triple = all_valid & all_compat & upper_tri_3
    triple_savings = jnp.where(
        valid_triple,
        v_points[:, None, None] + v_points[None, :, None] + v_points[None, None, :],
        0
    )
    best_savings = jnp.maximum(best_savings, jnp.max(triple_savings))

    return jnp.maximum(total_points - best_savings, 0)


# Batched version for GPU efficiency
batched_deadwood = jax.vmap(calculate_deadwood)


# =============================================================================
# JAX State Representation
# =============================================================================

@jax.jit
def init_state(rng_key):
    """Initialize a new game state with shuffled deck."""
    # Shuffle deck
    deck_order = jax.random.permutation(rng_key, NUM_CARDS)

    # Deal 10 cards to each player, 1 upcard
    player0_hand = jnp.zeros(NUM_CARDS, dtype=jnp.int8)
    player1_hand = jnp.zeros(NUM_CARDS, dtype=jnp.int8)

    # First 10 cards to player 0
    for i in range(HAND_SIZE):
        player0_hand = player0_hand.at[deck_order[i]].set(1)
    # Next 10 to player 1
    for i in range(HAND_SIZE, 2 * HAND_SIZE):
        player1_hand = player1_hand.at[deck_order[i]].set(1)

    # Card 21 is initial upcard
    upcard = deck_order[2 * HAND_SIZE]

    # Remaining cards form stock
    stock_mask = jnp.zeros(NUM_CARDS, dtype=jnp.int8)
    for i in range(2 * HAND_SIZE + 1, NUM_CARDS):
        stock_mask = stock_mask.at[deck_order[i]].set(1)

    return {
        'player0_hand': player0_hand,
        'player1_hand': player1_hand,
        'stock_mask': stock_mask,
        'discard_mask': jnp.zeros(NUM_CARDS, dtype=jnp.int8),
        'upcard': jnp.int8(upcard),
        'stock_top_idx': jnp.int8(0),  # Index into shuffled stock order
        'stock_order': deck_order[2 * HAND_SIZE + 1:],  # Order of stock cards
        'current_player': jnp.int8(0),
        'phase': jnp.int8(PHASE_FIRST_UPCARD),
        'done': jnp.bool_(False),
        'winner': jnp.int8(-1),  # -1=ongoing, 0=p0, 1=p1, 2=draw
        'p0_score': jnp.int32(0),
        'p1_score': jnp.int32(0),
        'drawn_from_discard': jnp.int8(-1),  # Card drawn from discard (-1 if none/stock)
        'repeated_move': jnp.bool_(False),  # True if prev player drew upcard and discarded it back
        'pass_count': jnp.int8(0),  # For first upcard phase
        # Knock/layoff state
        'knocked': jnp.zeros(2, dtype=jnp.bool_),  # knocked[i] = True if player i knocked
        'knocker': jnp.int8(-1),  # Which player knocked (-1 = no one)
        'layed_melds': jnp.zeros((2, NUM_MELDS), dtype=jnp.bool_),  # layed_melds[player, meld_id]
        'layoffs_mask': jnp.zeros(NUM_CARDS, dtype=jnp.bool_),  # Cards laid off onto knocker's melds
        'finished_layoffs': jnp.bool_(False),  # True after opponent finishes laying off cards
        'deadwood': jnp.zeros(2, dtype=jnp.int32),  # Current deadwood for each player
    }


@jax.jit
def get_hand(state, player):
    """Get hand mask for a player."""
    return jnp.where(player == 0, state['player0_hand'], state['player1_hand'])


@jax.jit
def set_hand(state, player, hand):
    """Set hand for a player."""
    new_p0 = jnp.where(player == 0, hand, state['player0_hand'])
    new_p1 = jnp.where(player == 1, hand, state['player1_hand'])
    return {**state, 'player0_hand': new_p0, 'player1_hand': new_p1}


# =============================================================================
# Knock/Layoff Helper Functions
# =============================================================================

# Precompute meld type info
def _compute_meld_info():
    """Compute meld type and extension info for layoffs."""
    is_rank_meld = np.zeros(NUM_MELDS, dtype=np.bool_)
    is_suit_meld = np.zeros(NUM_MELDS, dtype=np.bool_)
    meld_size = np.zeros(NUM_MELDS, dtype=np.int8)

    for meld_id, meld in enumerate(ALL_MELDS):
        meld_size[meld_id] = len(meld)
        if meld is None:
            continue
        # Check if rank meld (all same rank)
        ranks = [c // NUM_SUITS for c in meld]
        if len(set(ranks)) == 1:
            is_rank_meld[meld_id] = True
        else:
            is_suit_meld[meld_id] = True

    return jnp.array(is_rank_meld), jnp.array(is_suit_meld), jnp.array(meld_size)

IS_RANK_MELD, IS_SUIT_MELD, MELD_SIZE = _compute_meld_info()

# Meld action offset
ACTION_MELD_BASE = 56


# Precompute layoff extension info for each meld
def _compute_meld_layoff_info():
    """Precompute which cards can extend each meld."""
    # For each meld, which cards can be laid off onto it
    layoff_cards = np.zeros((NUM_MELDS, NUM_CARDS), dtype=np.bool_)

    for meld_id, meld in enumerate(ALL_MELDS):
        if meld is None:
            continue

        if IS_RANK_MELD[meld_id]:
            # For 3-of-a-kind, the 4th card can be laid off
            if len(meld) == 3:
                rank = meld[0] // NUM_SUITS
                for suit in range(NUM_SUITS):
                    card = rank * NUM_SUITS + suit
                    if card not in meld:
                        layoff_cards[meld_id, card] = True
        else:
            # IS_SUIT_MELD - for runs, can extend at either end
            suit = meld[0] % NUM_SUITS
            ranks = sorted([c // NUM_SUITS for c in meld])
            min_rank, max_rank = ranks[0], ranks[-1]
            if min_rank > 0:
                layoff_cards[meld_id, (min_rank - 1) * NUM_SUITS + suit] = True
            if max_rank < NUM_RANKS - 1:
                layoff_cards[meld_id, (max_rank + 1) * NUM_SUITS + suit] = True

    return jnp.array(layoff_cards)

MELD_LAYOFF_CARDS = _compute_meld_layoff_info()


@jax.jit
def compute_layoff_cards(layed_melds_mask, layoffs_mask):
    """Compute which cards can be laid off onto the knocker's melds.

    For rank melds of size 3: the 4th card of same rank can be laid off
    For suit melds: cards that extend the run at either end

    Uses precomputed lookup tables for JIT compatibility.
    """
    # For each meld that's laid, OR in its possible layoff cards
    # Shape: (NUM_MELDS, NUM_CARDS) * (NUM_MELDS, 1) -> (NUM_MELDS, NUM_CARDS)
    meld_contributions = MELD_LAYOFF_CARDS * layed_melds_mask[:, None]
    # OR across all melds -> (NUM_CARDS,)
    possible = jnp.any(meld_contributions, axis=0)
    return possible


# =============================================================================
# Legal Actions
# =============================================================================

@jax.jit
def legal_actions_mask(state):
    """Return mask of legal actions (vectorized, JIT-compatible)."""
    phase = state['phase']
    player = state['current_player']
    hand = get_hand(state, player)

    # Initialize action mask
    # Actions 0-51: card actions (discard or layoff)
    # 52: draw upcard, 53: draw stock, 54: pass, 55: knock
    # 56+: meld actions

    # Create masks for different action types
    card_in_hand = (hand > 0)  # Shape: (52,)
    valid_melds = valid_melds_mask(hand)  # Shape: (NUM_MELDS,)

    # Calculate deadwood for knock eligibility
    deadwood = calculate_deadwood(hand)
    hand_total = hand_total_points(hand)

    # === Phase: FirstUpcard ===
    first_upcard = (phase == PHASE_FIRST_UPCARD)
    first_upcard_draw = first_upcard  # Can draw upcard
    first_upcard_pass = first_upcard  # Can pass

    # === Phase: Draw ===
    draw_phase = (phase == PHASE_DRAW)
    stock_count = jnp.sum(state['stock_mask'])
    is_wall = stock_count <= 2
    has_upcard = (state['upcard'] >= 0)
    draw_upcard = draw_phase & ~is_wall & has_upcard
    draw_stock = draw_phase & ~is_wall & (stock_count > 0)
    draw_pass = draw_phase & is_wall  # Wall phase: only pass

    # === Phase: Discard ===
    discard_phase = (phase == PHASE_DISCARD)
    can_knock_discard = discard_phase & (deadwood <= KNOCK_THRESHOLD)

    # === Phase: Knock ===
    knock_phase = (phase == PHASE_KNOCK)
    hand_count = jnp.sum(hand)
    has_11_cards = hand_count == 11

    # With 11 cards: can discard cards that keep deadwood <= threshold
    # Simplified: allow discarding any card in hand (C++ does more complex check)
    # The proper check would require calculating deadwood for each possible discard
    knock_11_cards = knock_phase & has_11_cards

    # With 10 cards or fewer: can lay melds or pass
    # Note: C++ restricts melds/discards to valid arrangements, but we simplify
    # by allowing any valid meld and always allowing pass (to prevent getting stuck)
    knock_10_cards = knock_phase & (hand_count <= 10)
    # Always allow pass in knock phase with <= 10 cards to prevent getting stuck
    can_pass_knock = knock_10_cards

    # === Phase: Layoff ===
    layoff_phase = (phase == PHASE_LAYOFF)
    finished_layoffs = state['finished_layoffs']
    knocker = state['knocker']
    # Use where to safely index even when knocker is -1
    safe_knocker = jnp.where(knocker >= 0, knocker, 0)
    knocker_melds = state['layed_melds'][safe_knocker]
    knocker_valid = knocker >= 0

    # Compute layoff cards
    layoff_possible = compute_layoff_cards(knocker_melds, state['layoffs_mask'])
    layoff_card_actions = layoff_phase & ~finished_layoffs & knocker_valid  # Can do layoff cards

    # After layoffs: can lay own melds
    layoff_meld_actions = layoff_phase & finished_layoffs

    # === Build the mask ===
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)

    # Card actions (0-51)
    # Discard in discard phase OR knock phase with 11 cards OR layoff phase
    card_mask = (
        (discard_phase & card_in_hand) |
        (knock_11_cards & card_in_hand) |
        (layoff_card_actions & card_in_hand & layoff_possible)
    )
    mask = mask.at[:NUM_CARDS].set(card_mask)

    # Special actions (52-55)
    mask = mask.at[ACTION_DRAW_UPCARD].set(first_upcard_draw | draw_upcard)
    mask = mask.at[ACTION_DRAW_STOCK].set(draw_stock)
    mask = mask.at[ACTION_PASS].set(first_upcard_pass | draw_pass | can_pass_knock | layoff_phase)
    mask = mask.at[ACTION_KNOCK].set(can_knock_discard)

    # Meld actions (56+)
    meld_mask = (knock_10_cards | layoff_meld_actions) & valid_melds
    mask = mask.at[ACTION_MELD_BASE:ACTION_MELD_BASE + NUM_MELDS].set(meld_mask)

    return mask


# =============================================================================
# State Transitions
# =============================================================================

@jax.jit
def step(state, action):
    """Apply action and return new state."""
    phase = state['phase']
    player = state['current_player']

    # Get initial hands
    p0_hand = state['player0_hand']
    p1_hand = state['player1_hand']
    hand = jnp.where(player == 0, p0_hand, p1_hand)

    # Initialize all state fields
    new_p0_hand = p0_hand
    new_p1_hand = p1_hand
    new_stock_mask = state['stock_mask']
    new_discard_mask = state['discard_mask']
    new_upcard = state['upcard']
    new_stock_top_idx = state['stock_top_idx']
    new_current_player = state['current_player']
    new_phase = state['phase']
    new_done = state['done']
    new_winner = state['winner']
    new_pass_count = state['pass_count']
    new_drawn_from_discard = state['drawn_from_discard']
    new_repeated_move = state['repeated_move']

    # Initialize knock/layoff state fields
    new_knocked = state['knocked']
    new_knocker = state['knocker']
    new_layed_melds = state['layed_melds']
    new_layoffs_mask = state['layoffs_mask']
    new_finished_layoffs = state['finished_layoffs']
    new_deadwood = state['deadwood']
    new_p0_score = state['p0_score']
    new_p1_score = state['p1_score']

    # Handle FirstUpcard phase
    is_first_upcard = (phase == PHASE_FIRST_UPCARD)

    # Action: Pass in first upcard
    is_pass_first = is_first_upcard & (action == ACTION_PASS)
    new_pass_count = jnp.where(is_pass_first, new_pass_count + 1, new_pass_count)
    both_passed = (new_pass_count >= 2)
    new_phase = jnp.where(is_pass_first & both_passed, jnp.int8(PHASE_DRAW), new_phase)
    new_current_player = jnp.where(is_pass_first & both_passed, jnp.int8(0), new_current_player)
    new_current_player = jnp.where(is_pass_first & ~both_passed, jnp.int8(1 - player), new_current_player)
    # When both pass on first upcard, that upcard can no longer be drawn
    new_upcard = jnp.where(is_pass_first & both_passed, jnp.int8(-1), new_upcard)

    # Action: Draw upcard in first upcard
    is_draw_up_first = is_first_upcard & (action == ACTION_DRAW_UPCARD)
    upcard = state['upcard']
    updated_hand = hand.at[upcard].set(1)
    new_p0_hand = jnp.where(is_draw_up_first & (player == 0), updated_hand, new_p0_hand)
    new_p1_hand = jnp.where(is_draw_up_first & (player == 1), updated_hand, new_p1_hand)
    new_upcard = jnp.where(is_draw_up_first, jnp.int8(-1), new_upcard)
    new_phase = jnp.where(is_draw_up_first, jnp.int8(PHASE_DISCARD), new_phase)
    # Track that we drew this card from discard (can't discard it back)
    new_drawn_from_discard = jnp.where(is_draw_up_first, jnp.int8(upcard), new_drawn_from_discard)

    # Handle Draw phase
    is_draw_phase = (phase == PHASE_DRAW)

    # Action: Draw from stock
    is_draw_stock = is_draw_phase & (action == ACTION_DRAW_STOCK)
    stock_idx = state['stock_top_idx']
    drawn_card_stock = state['stock_order'][stock_idx]
    updated_hand_stock = hand.at[drawn_card_stock].set(1)
    new_p0_hand = jnp.where(is_draw_stock & (player == 0), updated_hand_stock, new_p0_hand)
    new_p1_hand = jnp.where(is_draw_stock & (player == 1), updated_hand_stock, new_p1_hand)
    new_stock_mask = jnp.where(is_draw_stock, new_stock_mask.at[drawn_card_stock].set(0), new_stock_mask)
    new_stock_top_idx = jnp.where(is_draw_stock, stock_idx + 1, new_stock_top_idx)
    new_phase = jnp.where(is_draw_stock, jnp.int8(PHASE_DISCARD), new_phase)
    # Drawing from stock - can discard any card
    new_drawn_from_discard = jnp.where(is_draw_stock, jnp.int8(-1), new_drawn_from_discard)

    # Action: Draw upcard (in draw phase)
    is_draw_up = is_draw_phase & (action == ACTION_DRAW_UPCARD)
    upcard_draw = state['upcard']
    updated_hand_up = hand.at[upcard_draw].set(1)
    new_p0_hand = jnp.where(is_draw_up & (player == 0), updated_hand_up, new_p0_hand)
    new_p1_hand = jnp.where(is_draw_up & (player == 1), updated_hand_up, new_p1_hand)
    new_upcard = jnp.where(is_draw_up, jnp.int8(-1), new_upcard)
    new_phase = jnp.where(is_draw_up, jnp.int8(PHASE_DISCARD), new_phase)
    # Track that we drew this card from discard (can't discard it back)
    new_drawn_from_discard = jnp.where(is_draw_up, jnp.int8(upcard_draw), new_drawn_from_discard)

    # Action: Pass in draw phase (Wall) - ends game in draw
    is_pass_wall = is_draw_phase & (action == ACTION_PASS)
    new_phase = jnp.where(is_pass_wall, jnp.int8(PHASE_GAME_OVER), new_phase)
    new_done = jnp.where(is_pass_wall, True, new_done)
    new_winner = jnp.where(is_pass_wall, jnp.int8(2), new_winner)  # 2 = draw

    # Handle Discard phase - need to use updated hands
    current_p0 = jnp.where(is_draw_up_first & (player == 0), updated_hand,
                          jnp.where(is_draw_stock & (player == 0), updated_hand_stock,
                                   jnp.where(is_draw_up & (player == 0), updated_hand_up, new_p0_hand)))
    current_p1 = jnp.where(is_draw_up_first & (player == 1), updated_hand,
                          jnp.where(is_draw_stock & (player == 1), updated_hand_stock,
                                   jnp.where(is_draw_up & (player == 1), updated_hand_up, new_p1_hand)))

    is_discard = (new_phase == PHASE_DISCARD) & (action < NUM_CARDS)
    current_hand_for_discard = jnp.where(player == 0, current_p0, current_p1)
    discarded_hand = current_hand_for_discard.at[action].set(0)

    new_p0_hand = jnp.where(is_discard & (player == 0), discarded_hand, current_p0)
    new_p1_hand = jnp.where(is_discard & (player == 1), discarded_hand, current_p1)
    new_discard_mask = jnp.where(is_discard, new_discard_mask.at[action].set(1), new_discard_mask)
    new_upcard = jnp.where(is_discard, jnp.int8(action), new_upcard)

    # Check for repeated move: discarding the same card just drawn from discard
    # C++ logic: if upcard == prev_upcard (discarded same card drawn from discard):
    #   - if repeated_move was already true: game ends as draw
    #   - else: set repeated_move = true
    # Otherwise: repeated_move = false
    this_is_repeat = is_discard & (action == state['drawn_from_discard'])
    both_repeated = this_is_repeat & state['repeated_move']  # Both players did it

    # Update repeated_move flag for next turn
    new_repeated_move = jnp.where(is_discard & this_is_repeat, True, new_repeated_move)
    new_repeated_move = jnp.where(is_discard & ~this_is_repeat, False, new_repeated_move)

    # After discard, switch player and go to Draw phase (unless both repeated or stock empty)
    stock_empty = ~jnp.any(new_stock_mask > 0)
    # Both players repeated = game ends as draw
    new_phase = jnp.where(is_discard & both_repeated, jnp.int8(PHASE_GAME_OVER), new_phase)
    new_done = jnp.where(is_discard & both_repeated, True, new_done)
    new_winner = jnp.where(is_discard & both_repeated, jnp.int8(2), new_winner)
    # Stock empty also ends game
    new_phase = jnp.where(is_discard & ~both_repeated & stock_empty, jnp.int8(PHASE_GAME_OVER), new_phase)
    new_done = jnp.where(is_discard & ~both_repeated & stock_empty, True, new_done)
    new_winner = jnp.where(is_discard & ~both_repeated & stock_empty, jnp.int8(2), new_winner)
    # Normal case: continue to draw phase
    new_phase = jnp.where(is_discard & ~both_repeated & ~stock_empty, jnp.int8(PHASE_DRAW), new_phase)
    new_current_player = jnp.where(is_discard & ~both_repeated, jnp.int8(1 - player), new_current_player)

    # Handle Knock action from Discard phase - enters KNOCK phase
    is_knock = (state['phase'] == PHASE_DISCARD) & (action == ACTION_KNOCK)
    # Set knocked flag and knocker
    new_knocked = jnp.where(is_knock, new_knocked.at[player].set(True), new_knocked)
    new_knocker = jnp.where(is_knock, jnp.int8(player), new_knocker)
    # Update deadwood for both players (now tracking total card values)
    new_deadwood = jnp.where(is_knock, jnp.array([
        hand_total_points(new_p0_hand),
        hand_total_points(new_p1_hand)
    ], dtype=jnp.int32), new_deadwood)
    new_phase = jnp.where(is_knock, jnp.int8(PHASE_KNOCK), new_phase)

    # Handle KNOCK phase actions
    is_knock_phase = (phase == PHASE_KNOCK)

    # In KNOCK phase with 11 cards: must discard first
    has_11_cards = jnp.sum(hand) == 11
    is_knock_discard = is_knock_phase & has_11_cards & (action < NUM_CARDS)
    discarded_hand_knock = hand.at[action].set(0)
    new_p0_hand = jnp.where(is_knock_discard & (player == 0), discarded_hand_knock, new_p0_hand)
    new_p1_hand = jnp.where(is_knock_discard & (player == 1), discarded_hand_knock, new_p1_hand)
    new_discard_mask = jnp.where(is_knock_discard, new_discard_mask.at[action].set(1), new_discard_mask)
    # Update deadwood after discard
    new_hand_after_knock_discard = jnp.where(player == 0, new_p0_hand, new_p1_hand)
    knock_discard_deadwood = hand_total_points(new_hand_after_knock_discard)
    new_deadwood = jnp.where(is_knock_discard, new_deadwood.at[player].set(knock_discard_deadwood), new_deadwood)

    # In KNOCK phase with 10 cards: can lay melds or pass
    is_meld_action = (action >= ACTION_MELD_BASE) & (action < ACTION_MELD_BASE + NUM_MELDS)
    is_knock_meld = is_knock_phase & ~has_11_cards & is_meld_action
    meld_id = action - ACTION_MELD_BASE
    meld_mask = MELD_MASKS[meld_id]
    hand_after_meld = hand - meld_mask
    new_p0_hand = jnp.where(is_knock_meld & (player == 0), hand_after_meld, new_p0_hand)
    new_p1_hand = jnp.where(is_knock_meld & (player == 1), hand_after_meld, new_p1_hand)
    # Track layed melds
    new_layed_melds = jnp.where(is_knock_meld, new_layed_melds.at[player, meld_id].set(True), new_layed_melds)
    # Update deadwood
    knock_meld_deadwood = hand_total_points(hand_after_meld)
    new_deadwood = jnp.where(is_knock_meld, new_deadwood.at[player].set(knock_meld_deadwood), new_deadwood)

    # In KNOCK phase: pass means done laying melds, go to LAYOFF
    is_knock_pass = is_knock_phase & ~has_11_cards & (action == ACTION_PASS)
    # Get current hand for deadwood calculation
    current_hand_for_pass = jnp.where(player == 0, new_p0_hand, new_p1_hand)
    final_knock_deadwood = hand_total_points(current_hand_for_pass)
    new_deadwood = jnp.where(is_knock_pass, new_deadwood.at[player].set(final_knock_deadwood), new_deadwood)
    # If knocker has gin (deadwood 0), opponent can't lay off cards
    knocker_deadwood = new_deadwood[player]
    new_finished_layoffs = jnp.where(is_knock_pass & (knocker_deadwood == 0), True, new_finished_layoffs)
    # Switch to opponent and enter LAYOFF phase
    new_current_player = jnp.where(is_knock_pass, jnp.int8(1 - player), new_current_player)
    new_phase = jnp.where(is_knock_pass, jnp.int8(PHASE_LAYOFF), new_phase)

    # Handle LAYOFF phase actions
    is_layoff_phase = (phase == PHASE_LAYOFF)
    is_finished_layoffs = state['finished_layoffs']

    # Layoff a card (before finished_layoffs)
    is_layoff_card = is_layoff_phase & ~is_finished_layoffs & (action < NUM_CARDS)
    layoff_hand = hand.at[action].set(0)
    new_p0_hand = jnp.where(is_layoff_card & (player == 0), layoff_hand, new_p0_hand)
    new_p1_hand = jnp.where(is_layoff_card & (player == 1), layoff_hand, new_p1_hand)
    new_layoffs_mask = jnp.where(is_layoff_card, new_layoffs_mask.at[action].set(True), new_layoffs_mask)
    # Update deadwood
    layoff_deadwood = hand_total_points(layoff_hand)
    new_deadwood = jnp.where(is_layoff_card, new_deadwood.at[player].set(layoff_deadwood), new_deadwood)

    # Pass to finish laying off cards
    is_layoff_pass_cards = is_layoff_phase & ~is_finished_layoffs & (action == ACTION_PASS)
    new_finished_layoffs = jnp.where(is_layoff_pass_cards, True, new_finished_layoffs)

    # Lay meld (after finished_layoffs)
    is_layoff_meld = is_layoff_phase & is_finished_layoffs & is_meld_action
    layoff_meld_mask = MELD_MASKS[meld_id]
    hand_after_layoff_meld = hand - layoff_meld_mask
    new_p0_hand = jnp.where(is_layoff_meld & (player == 0), hand_after_layoff_meld, new_p0_hand)
    new_p1_hand = jnp.where(is_layoff_meld & (player == 1), hand_after_layoff_meld, new_p1_hand)
    new_layed_melds = jnp.where(is_layoff_meld, new_layed_melds.at[player, meld_id].set(True), new_layed_melds)
    # Update deadwood
    layoff_meld_deadwood = hand_total_points(hand_after_layoff_meld)
    new_deadwood = jnp.where(is_layoff_meld, new_deadwood.at[player].set(layoff_meld_deadwood), new_deadwood)

    # Pass to end game (after finished_layoffs)
    is_layoff_pass_end = is_layoff_phase & is_finished_layoffs & (action == ACTION_PASS)
    # Finalize deadwood
    final_layoff_hand = jnp.where(player == 0, new_p0_hand, new_p1_hand)
    new_deadwood = jnp.where(is_layoff_pass_end, new_deadwood.at[player].set(hand_total_points(final_layoff_hand)), new_deadwood)
    new_phase = jnp.where(is_layoff_pass_end, jnp.int8(PHASE_GAME_OVER), new_phase)
    new_done = jnp.where(is_layoff_pass_end, True, new_done)

    return {
        'player0_hand': new_p0_hand,
        'player1_hand': new_p1_hand,
        'stock_mask': new_stock_mask,
        'discard_mask': new_discard_mask,
        'upcard': new_upcard,
        'stock_top_idx': new_stock_top_idx,
        'stock_order': state['stock_order'],
        'current_player': new_current_player,
        'phase': new_phase,
        'done': new_done,
        'winner': new_winner,
        'p0_score': new_p0_score,
        'p1_score': new_p1_score,
        'drawn_from_discard': new_drawn_from_discard,
        'repeated_move': new_repeated_move,
        'pass_count': new_pass_count,
        # Knock/layoff state
        'knocked': new_knocked,
        'knocker': new_knocker,
        'layed_melds': new_layed_melds,
        'layoffs_mask': new_layoffs_mask,
        'finished_layoffs': new_finished_layoffs,
        'deadwood': new_deadwood,
    }


# Scoring constants (matching C++ defaults)
GIN_BONUS = 25
UNDERCUT_BONUS = 25


@jax.jit
def get_returns(state):
    """Get returns for both players.

    Scoring:
    - If player knocked: score = opponent_deadwood - knocker_deadwood
    - If knocker has gin (deadwood=0): add gin_bonus
    - If score < 0 (undercut): subtract undercut_bonus
    - If neither knocked: both get 0 (draw)
    """
    knocked = state['knocked']
    deadwood = state['deadwood']

    # Initialize returns to 0
    p0_return = jnp.float32(0.0)
    p1_return = jnp.float32(0.0)

    # Player 0 knocked
    p0_knocked = knocked[0]
    p0_score = jnp.float32(deadwood[1] - deadwood[0])
    # Gin bonus
    p0_score = jnp.where(deadwood[0] == 0, p0_score + GIN_BONUS, p0_score)
    # Undercut bonus (subtracted if score < 0)
    p0_score = jnp.where(p0_score < 0, p0_score - UNDERCUT_BONUS, p0_score)

    p0_return = jnp.where(p0_knocked, p0_score, p0_return)
    p1_return = jnp.where(p0_knocked, -p0_score, p1_return)

    # Player 1 knocked
    p1_knocked = knocked[1]
    p1_score = jnp.float32(deadwood[0] - deadwood[1])
    # Gin bonus
    p1_score = jnp.where(deadwood[1] == 0, p1_score + GIN_BONUS, p1_score)
    # Undercut bonus
    p1_score = jnp.where(p1_score < 0, p1_score - UNDERCUT_BONUS, p1_score)

    p0_return = jnp.where(p1_knocked, -p1_score, p0_return)
    p1_return = jnp.where(p1_knocked, p1_score, p1_return)

    # If neither knocked (draw), both get 0 (already initialized)

    return jnp.array([p0_return, p1_return])


# =============================================================================
# Batched operations
# =============================================================================

batched_init = jax.vmap(init_state)
batched_step = jax.vmap(step)
batched_legal_actions_mask = jax.vmap(legal_actions_mask)
batched_get_returns = jax.vmap(get_returns)


def init_batch(batch_size, rng_key):
    """Initialize a batch of games."""
    keys = jax.random.split(rng_key, batch_size)
    return batched_init(keys)


# =============================================================================
# OpenSpiel Python Game Interface
# =============================================================================

_GAME_TYPE = pyspiel.GameType(
    short_name="python_gin_rummy_jax",
    long_name="Python Gin Rummy (JAX)",
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
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=300,
)


class GinRummyJaxState(pyspiel.State):
    """OpenSpiel state wrapper around JAX state."""

    def __init__(self, game, jax_state=None, rng_key=None):
        super().__init__(game)
        if jax_state is None:
            if rng_key is None:
                rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))
            self._jax_state = init_state(rng_key)
        else:
            self._jax_state = jax_state
        self._cached_legal_actions = None

    def current_player(self):
        if self._jax_state['done']:
            return pyspiel.PlayerId.TERMINAL
        return int(self._jax_state['current_player'])

    def is_chance_node(self):
        # We handle chance internally via RNG, so no explicit chance nodes
        return False

    def legal_actions(self, player=None):
        if self._cached_legal_actions is None:
            mask = legal_actions_mask(self._jax_state)
            self._cached_legal_actions = [i for i in range(NUM_ACTIONS) if mask[i]]
        return self._cached_legal_actions

    def legal_actions_mask(self, player=None):
        mask = legal_actions_mask(self._jax_state)
        return [int(m) for m in mask]

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

    def observation_tensor(self, player=None):
        if player is None:
            player = self.current_player()
        if player < 0:
            player = 0

        # Build observation: current player's hand + visible info
        hand = get_hand(self._jax_state, player)
        discard = self._jax_state['discard_mask']
        upcard_one_hot = jnp.zeros(NUM_CARDS)
        if self._jax_state['upcard'] >= 0:
            upcard_one_hot = upcard_one_hot.at[self._jax_state['upcard']].set(1)

        obs = jnp.concatenate([hand, discard, upcard_one_hot])
        return list(obs.astype(float))

    def observation_string(self, player=None):
        return str(self)

    def __str__(self):
        p0_hand = self._jax_state['player0_hand']
        p1_hand = self._jax_state['player1_hand']
        upcard = int(self._jax_state['upcard'])
        phase = int(self._jax_state['phase'])
        player = int(self._jax_state['current_player'])

        phase_names = ['Deal', 'FirstUpcard', 'Draw', 'Discard', 'Knock', 'Layoff', 'GameOver']

        def cards_str(mask):
            cards = [i for i in range(NUM_CARDS) if mask[i]]
            return ' '.join(card_name(c) for c in cards)

        def card_name(c):
            if c < 0:
                return 'XX'
            ranks = 'A23456789TJQK'
            suits = 'scdh'
            return ranks[c // 4] + suits[c % 4]

        lines = [
            f"Phase: {phase_names[phase]}, Player: {player}",
            f"Upcard: {card_name(upcard)}",
            f"P0 hand: {cards_str(p0_hand)}",
            f"P1 hand: {cards_str(p1_hand)}",
        ]
        return '\n'.join(lines)

    def clone(self):
        cloned = GinRummyJaxState(self.get_game())
        cloned._jax_state = jax.tree.map(lambda x: x.copy(), self._jax_state)
        return cloned


class GinRummyJaxGame(pyspiel.Game):
    """OpenSpiel game wrapper for JAX Gin Rummy."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})

    def new_initial_state(self):
        return GinRummyJaxState(self)

    def num_distinct_actions(self):
        return NUM_ACTIONS

    def num_players(self):
        return 2

    def min_utility(self):
        return -1.0

    def max_utility(self):
        return 1.0

    def observation_tensor_shape(self):
        return [NUM_CARDS * 3]  # hand + discard + upcard

    def observation_tensor_size(self):
        return NUM_CARDS * 3

    def max_game_length(self):
        return 300


# Register the game
pyspiel.register_game(_GAME_TYPE, GinRummyJaxGame)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=== Testing JAX Gin Rummy ===\n")

    # Test single game
    print("1. Testing OpenSpiel interface:")
    game = pyspiel.load_game("python_gin_rummy_jax")
    state = game.new_initial_state()
    print(f"Initial state:\n{state}\n")
    print(f"Current player: {state.current_player()}")
    print(f"Legal actions: {state.legal_actions()}")
    print(f"Is terminal: {state.is_terminal()}")

    # Play a few moves
    print("\n2. Playing a few moves:")
    for i in range(10):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        action = legal[0]
        print(f"  Player {state.current_player()}: action {action}")
        state.apply_action(action)

    print(f"\nState after moves:\n{state}")

    # Play a complete game
    print("\n3. Playing complete game:")
    state = game.new_initial_state()
    move_count = 0
    while not state.is_terminal() and move_count < 200:
        legal = state.legal_actions()
        action = np.random.choice(legal)
        state.apply_action(action)
        move_count += 1

    print(f"Game ended after {move_count} moves")
    print(f"Returns: {state.returns()}")

    # Test batched operations
    print("\n4. Testing batched operations:")
    key = jax.random.PRNGKey(42)
    batch_size = 100
    states = init_batch(batch_size, key)
    print(f"Initialized {batch_size} games")

    # Check shapes
    print(f"  player0_hand shape: {states['player0_hand'].shape}")
    print(f"  phase shape: {states['phase'].shape}")

    print("\n=== Done ===")

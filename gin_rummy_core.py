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

# =============================================================================
# Vectorized meld combination search (K=24 max melds)
# =============================================================================

MAX_MELDS_IN_HAND = 24

def _generate_selection_matrix(k):
    from itertools import combinations
    subsets = []
    subsets.append([0] * k)
    for i in range(k):
        row = [0] * k
        row[i] = 1
        subsets.append(row)
    for i, j in combinations(range(k), 2):
        row = [0] * k
        row[i] = 1
        row[j] = 1
        subsets.append(row)
    for i, j, l in combinations(range(k), 3):
        row = [0] * k
        row[i] = 1
        row[j] = 1
        row[l] = 1
        subsets.append(row)
    return np.array(subsets, dtype=np.int8)

SELECTION_MATRIX = jnp.array(_generate_selection_matrix(MAX_MELDS_IN_HAND))
NUM_COMBOS = SELECTION_MATRIX.shape[0]

# Precompute card values (point value of each card 0-51)
CARD_VALUES = jnp.array([CARD_POINTS[i % NUM_RANKS] for i in range(NUM_CARDS)], dtype=jnp.int8)

@jax.jit
def hand_total_points(hand):
    return jnp.sum(hand * CARD_VALUES, dtype=jnp.int16)  # int16 to avoid overflow (max ~110)

@jax.jit
def valid_melds_mask(hand):
    hand_expanded = hand[None, :]
    meld_card_counts = jnp.sum(MELD_MASKS * hand_expanded, axis=1)
    meld_sizes = jnp.sum(MELD_MASKS, axis=1)
    return meld_card_counts == meld_sizes

@jax.jit
def calculate_deadwood(hand):
    total_points = hand_total_points(hand)
    valid = valid_melds_mask(hand)
    v_points = jnp.where(valid, MELD_POINTS, 0)
    valid_indices = jnp.where(valid, size=32, fill_value=0)[0]
    best_single = jnp.max(v_points)
    v_overlap = MELD_OVERLAP[valid_indices][:, valid_indices]
    v_pts = v_points[valid_indices]
    valid_pair = ~v_overlap
    pair_savings = jnp.where(valid_pair, v_pts[:, None] + v_pts[None, :], 0)
    best_pair = jnp.max(pair_savings)
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
    total_hand_points = jnp.sum(hand * CARD_VALUES, dtype=jnp.int16)
    is_valid = valid_melds_mask(hand)
    sort_key = jnp.where(is_valid, -MELD_POINTS, 1000)
    top_k_indices = jnp.argsort(sort_key)[:MAX_MELDS_IN_HAND]
    top_k_masks = MELD_MASKS[top_k_indices]
    top_k_valid = is_valid[top_k_indices]
    top_k_points = MELD_POINTS[top_k_indices]
    card_usage = jnp.dot(SELECTION_MATRIX, top_k_masks)
    is_disjoint = jnp.all(card_usage <= 1, axis=1)
    selected_valid_count = jnp.dot(SELECTION_MATRIX, top_k_valid.astype(jnp.int8))
    selected_total_count = jnp.sum(SELECTION_MATRIX, axis=1)
    all_selected_valid = (selected_valid_count == selected_total_count)
    combo_valid = is_disjoint & all_selected_valid
    combo_points = jnp.dot(SELECTION_MATRIX, top_k_points)
    card_not_used = (card_usage == 0)
    valid_for_discard = combo_valid[:, None] & card_not_used
    combo_points_expanded = combo_points[:, None]
    masked_points = jnp.where(valid_for_discard, combo_points_expanded, -1)
    best_meld_points = jnp.max(masked_points, axis=0)
    deadwood = (total_hand_points - CARD_VALUES) - best_meld_points
    deadwood = jnp.where(hand > 0, deadwood, 999)
    return deadwood

@jax.jit
def min_deadwood_after_discard(hand):
    all_dw = calculate_deadwood_all_discards(hand)
    return jnp.min(all_dw)

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
    """
    phase = state['phase']
    player = state['current_player']
    hand = get_hand(state, player)
    upcard = state['upcard']

    # === Chance node: pick first available card from deck ===
    is_chance = (phase == PHASE_DEAL) | state['waiting_stock_draw']
    deck = state['deck']
    chance_action = jnp.argmax(deck)  # First card in deck

    # === First upcard phase ===
    # Take upcard if it would be in a meld or enable knock
    hand_with_upcard = jnp.where(upcard >= 0, hand.at[upcard].set(1), hand)
    min_dw_with_upcard = min_deadwood_after_discard(hand_with_upcard)
    upcard_enables_knock = min_dw_with_upcard <= KNOCK_THRESHOLD
    upcard_in_meld = jnp.where(upcard >= 0, card_would_be_in_meld(hand, upcard), False)
    take_upcard = upcard_enables_knock | upcard_in_meld
    first_upcard_action = jnp.where(take_upcard, ACTION_DRAW_UPCARD, ACTION_PASS)

    # === Draw phase ===
    # Similar logic, but also handle wall
    stock_count = jnp.sum(deck)
    at_wall = stock_count <= WALL_STOCK_SIZE
    draw_action = jnp.where(take_upcard & (upcard >= 0), ACTION_DRAW_UPCARD, ACTION_DRAW_STOCK)
    # At wall with no good upcard: pass
    draw_action = jnp.where(at_wall & ~take_upcard, ACTION_PASS, draw_action)

    # === Discard phase ===
    min_dw = min_deadwood_after_discard(hand)
    can_knock = min_dw <= KNOCK_THRESHOLD
    discard_card = best_discard(hand)
    discard_action = jnp.where(can_knock, ACTION_KNOCK, discard_card)

    # === Knock phase ===
    # If 11 cards, discard; if 10 or fewer, pass (simplified: skip meld laying)
    hand_count = jnp.sum(hand)
    knock_action = jnp.where(hand_count >= 11, best_discard(hand), ACTION_PASS)

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

print(f"SELECTION_MATRIX shape: {SELECTION_MATRIX.shape}")
print(f"NUM_COMBOS: {NUM_COMBOS}")

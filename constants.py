"""Shared constants for Gin Rummy implementations.

This module defines constants used across gin_rummy_core.py and gin_rummy_jax.py.
These values match the C++ OpenSpiel implementation for compatibility.

Card Encoding:
    card_index = suit * 13 + rank
    - Suits: spades=0, clubs=1, diamonds=2, hearts=3
    - Ranks: A=0, 2=1, 3=2, ..., 10=9, J=10, Q=11, K=12

Action Encoding:
    - 0-51: Discard card at index
    - 52: Draw upcard
    - 53: Draw from stock
    - 54: Pass
    - 55: Knock
    - 56-240: Lay meld (meld_id = action - 56)
"""

# Card constants
NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4

# Hand constants
HAND_SIZE = 10  # Initial hand size
KNOCK_THRESHOLD = 10  # Maximum deadwood to knock

# Card values for deadwood calculation
# Face cards (J, Q, K) = 10, Ace = 1, others = face value
CARD_VALUES_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # Per rank

# Game phases
PHASE_DEAL = 0           # Chance node: dealing cards
PHASE_FIRST_UPCARD = 1   # First player decides on upcard
PHASE_DRAW = 2           # Player draws (upcard or stock)
PHASE_DISCARD = 3        # Player discards
PHASE_KNOCK = 4          # Knocker lays melds
PHASE_LAYOFF = 5         # Defender lays melds and layoffs
PHASE_WALL = 6           # Stock exhausted, game ends
PHASE_GAME_OVER = 7      # Terminal state

PHASE_NAMES = [
    'DEAL', 'FIRST_UPCARD', 'DRAW', 'DISCARD',
    'KNOCK', 'LAYOFF', 'WALL', 'GAME_OVER'
]

# Action constants
ACTION_DRAW_UPCARD = 52
ACTION_DRAW_STOCK = 53
ACTION_PASS = 54
ACTION_KNOCK = 55
ACTION_MELD_BASE = 56

# Meld constants
NUM_MELDS = 185  # Total number of valid melds
# Melds 0-52: Sets (3 or 4 of a kind)
# Melds 53-184: Runs (3+ consecutive cards of same suit)

NUM_ACTIONS = 241  # 0-51 cards, 52-55 special, 56-240 melds

# Scoring constants
GIN_BONUS = 25       # Bonus for going gin (0 deadwood)
UNDERCUT_BONUS = 25  # Bonus when defender has <= knocker deadwood


def card_rank(card_index):
    """Get rank (0-12) from card index."""
    return card_index % NUM_RANKS


def card_suit(card_index):
    """Get suit (0-3) from card index."""
    return card_index // NUM_RANKS


def make_card(rank, suit):
    """Create card index from rank and suit."""
    return suit * NUM_RANKS + rank


def card_value(card_index):
    """Get point value of a card (for deadwood calculation)."""
    rank = card_rank(card_index)
    return CARD_VALUES_LIST[rank]


def card_str(card_index):
    """Convert card index to human-readable string (e.g., 'As', 'Kh')."""
    if card_index < 0 or card_index >= NUM_CARDS:
        return "??"
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['s', 'c', 'd', 'h']
    return ranks[card_rank(card_index)] + suits[card_suit(card_index)]


def action_str(action):
    """Convert action index to human-readable string."""
    if action < 52:
        return f"discard_{card_str(action)}"
    elif action == ACTION_DRAW_UPCARD:
        return "draw_upcard"
    elif action == ACTION_DRAW_STOCK:
        return "draw_stock"
    elif action == ACTION_PASS:
        return "pass"
    elif action == ACTION_KNOCK:
        return "knock"
    else:
        meld_id = action - ACTION_MELD_BASE
        return f"meld_{meld_id}"

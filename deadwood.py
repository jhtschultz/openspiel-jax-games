"""
Exact deadwood calculation for Gin Rummy using JAX.

The key insight is that with 10-11 cards and melds of size 3+,
you can have at most 3 melds. This bounds the search space.

For each hand:
1. Find which melds are valid (all cards present)
2. Enumerate combinations of 1, 2, 3 non-overlapping melds
3. Return minimum deadwood
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# =============================================================================
# Constants
# =============================================================================
NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4

# Card point values: A=1, 2-10=face, J/Q/K=10
CARD_POINTS = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=jnp.int32)


def card_to_rank(card):
    return card // NUM_SUITS


def card_to_suit(card):
    return card % NUM_SUITS


# =============================================================================
# Precompute All Melds
# =============================================================================

def generate_all_melds():
    """
    Generate all valid melds.

    Returns:
        melds: List of tuples, each tuple is card indices in the meld
        meld_masks: (num_melds, 52) binary array
        meld_points: (num_melds,) point value of each meld
    """
    melds = []

    # Sets: 3 or 4 cards of same rank, different suits
    for rank in range(NUM_RANKS):
        base_cards = [rank * NUM_SUITS + s for s in range(NUM_SUITS)]

        # All 3-card combinations
        for i in range(NUM_SUITS):
            for j in range(i + 1, NUM_SUITS):
                for k in range(j + 1, NUM_SUITS):
                    melds.append(tuple(sorted([base_cards[i], base_cards[j], base_cards[k]])))

        # 4-card set
        melds.append(tuple(sorted(base_cards)))

    # Runs: 3+ consecutive cards of same suit
    for suit in range(NUM_SUITS):
        for start_rank in range(NUM_RANKS):
            for length in range(3, NUM_RANKS - start_rank + 1):
                cards = tuple(r * NUM_SUITS + suit for r in range(start_rank, start_rank + length))
                melds.append(cards)

    # Convert to arrays
    num_melds = len(melds)
    meld_masks = np.zeros((num_melds, NUM_CARDS), dtype=np.int8)
    meld_points = np.zeros(num_melds, dtype=np.int32)

    for i, meld in enumerate(melds):
        for card in meld:
            meld_masks[i, card] = 1
            meld_points[i] += int(CARD_POINTS[card // NUM_SUITS])

    return melds, jnp.array(meld_masks), jnp.array(meld_points)


# Generate melds at module load time
ALL_MELDS, MELD_MASKS, MELD_POINTS = generate_all_melds()
NUM_MELDS = len(ALL_MELDS)

# Precompute meld compatibility (which pairs don't overlap)
# Two melds are compatible if they share no cards
_meld_overlap = np.zeros((NUM_MELDS, NUM_MELDS), dtype=np.int8)
for i in range(NUM_MELDS):
    for j in range(NUM_MELDS):
        # Check if any card appears in both melds
        overlap = np.any(np.array(MELD_MASKS[i]) & np.array(MELD_MASKS[j]))
        _meld_overlap[i, j] = 1 if overlap else 0

MELD_OVERLAP = jnp.array(_meld_overlap)  # 1 if melds overlap, 0 if compatible

print(f"Generated {NUM_MELDS} melds")
print(f"  Sets (3-card): {13 * 4}")  # 13 ranks × C(4,3) = 52
print(f"  Sets (4-card): {13}")       # 13 ranks
print(f"  Runs: {NUM_MELDS - 65}")    # Rest are runs


# =============================================================================
# Hand Utilities
# =============================================================================

@jax.jit
def hand_total_points(hand_mask):
    """Calculate total points of all cards in hand."""
    card_pts = CARD_POINTS[jnp.arange(NUM_CARDS) // NUM_SUITS]
    return jnp.sum(hand_mask * card_pts)


@jax.jit
def valid_melds_mask(hand_mask):
    """
    Return boolean mask of which melds are valid for this hand.
    A meld is valid if all its cards are present in the hand.
    """
    # For each meld, check if (meld_mask & hand_mask) == meld_mask
    # Equivalent to: meld_mask <= hand_mask for all positions
    return jnp.all(MELD_MASKS <= hand_mask, axis=1)


# =============================================================================
# Deadwood Calculation - Optimized Matrix Approach
# =============================================================================

# Maximum number of valid melds we expect per hand (for static sizing)
MAX_VALID_MELDS = 32

# Precompute compatibility matrix (1 = compatible, 0 = overlapping)
MELD_COMPAT = 1 - MELD_OVERLAP  # (NUM_MELDS, NUM_MELDS)


@jax.jit
def min_deadwood_exact(hand_mask):
    """
    Calculate minimum deadwood for a hand using full matrix operations.

    Strategy:
    1. Find all valid melds (usually ~10-30)
    2. Extract valid meld indices to fixed-size array
    3. Use matrix ops for singles, pairs, and triples
    """
    total_points = hand_total_points(hand_mask)
    valid = valid_melds_mask(hand_mask)  # (NUM_MELDS,) bool

    # No melds = all deadwood
    best_savings = jnp.int32(0)

    # Get indices of valid melds, padded with index 0 (will be masked out)
    valid_indices = jnp.where(valid, size=MAX_VALID_MELDS, fill_value=0)[0]
    num_valid = jnp.sum(valid)
    valid_mask = jnp.arange(MAX_VALID_MELDS) < num_valid  # (MAX,)

    # Get properties for valid melds
    v_points = MELD_POINTS[valid_indices]  # (MAX,)

    # === Single melds: vectorized max ===
    single_savings = jnp.where(valid_mask, v_points, 0)
    best_savings = jnp.maximum(best_savings, jnp.max(single_savings))

    # === Build compatibility matrix for valid melds ===
    # v_compat[i,j] = can valid_indices[i] and valid_indices[j] coexist?
    v_compat = MELD_COMPAT[valid_indices][:, valid_indices]  # (MAX, MAX)

    # === Pairs: matrix operations ===
    # Only consider upper triangle (j > i) to avoid duplicates
    upper_tri_2 = jnp.triu(jnp.ones((MAX_VALID_MELDS, MAX_VALID_MELDS), dtype=jnp.bool_), k=1)
    valid_pair = (valid_mask[:, None] & valid_mask[None, :] &
                  v_compat.astype(jnp.bool_) & upper_tri_2)

    pair_savings = jnp.where(
        valid_pair,
        v_points[:, None] + v_points[None, :],
        0
    )
    best_savings = jnp.maximum(best_savings, jnp.max(pair_savings))

    # === Triples: 3D tensor operations ===
    # valid_triple[i,j,k] = can use melds i, j, k together? (with i < j < k)
    # Only need upper triangular in all dimensions
    idx = jnp.arange(MAX_VALID_MELDS)
    i_lt_j = idx[:, None, None] < idx[None, :, None]  # i < j
    j_lt_k = idx[None, :, None] < idx[None, None, :]  # j < k
    upper_tri_3 = i_lt_j & j_lt_k

    # All three must be valid
    all_valid = (valid_mask[:, None, None] &
                 valid_mask[None, :, None] &
                 valid_mask[None, None, :])

    # Pairwise compatibility: (i,j), (i,k), (j,k) all compatible
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


# Batched version
batched_min_deadwood = jax.vmap(min_deadwood_exact)


# =============================================================================
# Testing
# =============================================================================

def hand_from_cards(cards):
    """Create hand mask from list of card indices."""
    mask = np.zeros(NUM_CARDS, dtype=np.int8)
    for c in cards:
        mask[c] = 1
    return jnp.array(mask)


def card_name(c):
    """Get human-readable card name."""
    ranks = 'A23456789TJQK'
    suits = 'scdh'
    return ranks[c // 4] + suits[c % 4]


def print_hand(hand_mask):
    """Print cards in hand."""
    cards = [i for i in range(NUM_CARDS) if hand_mask[i]]
    return ' '.join(card_name(c) for c in cards)


if __name__ == "__main__":
    print("\n=== Testing Deadwood Calculation ===\n")

    # Test 1: Hand with no melds (random high cards)
    # Ks, Qh, Jd, Tc, 9s, 8h, 7d, 6c, 5s, 4h
    hand1 = hand_from_cards([48, 47, 42, 37, 32, 31, 26, 21, 16, 15])
    print(f"Hand 1: {print_hand(hand1)}")
    print(f"  Total points: {hand_total_points(hand1)}")
    dead1 = min_deadwood_exact(hand1)
    print(f"  Min deadwood: {dead1}")

    # Test 2: Hand with one run (As, 2s, 3s + random)
    # As=0, 2s=4, 3s=8, plus Kh, Qd, Jc, Ts, 9h, 8d, 7c
    hand2 = hand_from_cards([0, 4, 8, 51, 46, 41, 36, 35, 30, 25])
    print(f"\nHand 2: {print_hand(hand2)}")
    print(f"  Total points: {hand_total_points(hand2)}")
    dead2 = min_deadwood_exact(hand2)
    print(f"  Min deadwood: {dead2} (should be total - 6 for A23 run)")

    # Test 3: Hand with one set (Ks, Kc, Kd + random)
    # Ks=48, Kc=49, Kd=50, plus Qh, Jd, Tc, 9s, 8h, 7d, 6c
    hand3 = hand_from_cards([48, 49, 50, 47, 42, 37, 32, 31, 26, 21])
    print(f"\nHand 3: {print_hand(hand3)}")
    print(f"  Total points: {hand_total_points(hand3)}")
    dead3 = min_deadwood_exact(hand3)
    print(f"  Min deadwood: {dead3} (should be total - 30 for KKK set)")

    # Test 4: Hand with two melds
    # As,2s,3s (run=6pts) + Kh,Kd,Kc (set=30pts) + 9h,8d,7c,6s
    hand4 = hand_from_cards([0, 4, 8, 51, 50, 49, 35, 30, 25, 20])
    print(f"\nHand 4: {print_hand(hand4)}")
    print(f"  Total points: {hand_total_points(hand4)}")
    dead4 = min_deadwood_exact(hand4)
    print(f"  Min deadwood: {dead4} (should be total - 36 for run + set)")

    # Test 5: Gin hand (all cards in melds)
    # As,2s,3s + 4s,5s,6s + 7s,8s,9s,Ts (run of 4)
    hand5 = hand_from_cards([0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
    print(f"\nHand 5: {print_hand(hand5)}")
    print(f"  Total points: {hand_total_points(hand5)}")
    dead5 = min_deadwood_exact(hand5)
    print(f"  Min deadwood: {dead5} (should be 0 for gin)")

    # Benchmark
    print("\n=== Benchmark ===")
    import time

    # Create batch of random hands
    key = jax.random.PRNGKey(42)

    for batch_size in [1000, 10000, 100000]:
        print(f"\n--- Batch size: {batch_size} ---")

        # Generate random hands (10 cards each)
        def random_hand(key):
            cards = jax.random.permutation(key, NUM_CARDS)[:10]
            mask = jnp.zeros(NUM_CARDS, dtype=jnp.int8)
            return mask.at[cards].set(1)

        keys = jax.random.split(key, batch_size)
        hands = jax.vmap(random_hand)(keys)

        # Warmup
        print(f"  Warming up...")
        _ = batched_min_deadwood(hands)
        jax.block_until_ready(_)

        # Benchmark
        print("  Benchmarking...")
        start = time.perf_counter()
        for _ in range(10):
            deadwoods = batched_min_deadwood(hands)
            jax.block_until_ready(deadwoods)
        elapsed = time.perf_counter() - start

        hands_per_sec = 10 * batch_size / elapsed
        print(f"  {hands_per_sec:,.0f} hands/sec")
        print(f"  Mean deadwood: {jnp.mean(deadwoods):.1f}")
        print(f"  Hands with deadwood <= 10: {jnp.sum(deadwoods <= 10)}/{batch_size}")

    print("\n=== Done ===")

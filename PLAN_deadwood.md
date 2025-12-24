# Plan: JAX-Compatible Deadwood Calculation

## The Problem

Given a hand of 10-11 cards, find the minimum deadwood by selecting non-overlapping melds.

**Input:** Hand as 52-bit mask (1 = card present)
**Output:** Minimum deadwood value (0-100+)

**Constraints:**
- Must be JIT-compilable (no Python control flow)
- Must be vmap-able (work on batches of 100k+ hands)
- Must be reasonably fast (can't enumerate all combinations)

## Key Insight: Max 3 Melds

With 10-11 cards and melds of size 3+, you can have at most 3 melds:
- 3 + 3 + 3 = 9 cards (1-2 deadwood cards)
- 3 + 3 + 4 = 10 cards (0-1 deadwood cards)
- 3 + 4 + 4 = 11 cards (0 deadwood cards)
- etc.

This means we only need to search depth 3!

## Approach: Fixed-Depth Vectorized Search

### Step 1: Precompute All Melds

```python
# ~185 possible melds
# Sets: 13 ranks × 5 combinations (C(4,3) + C(4,4)) = 65
# Runs: 4 suits × (11+10+9+8+7+6+5+4+3+2+1) = 4 × 66 = ~264 (overlapping counted)
# Actually ~185 unique melds after dedup

MELD_MASKS = jnp.array([...])  # (185, 52) binary masks
MELD_POINTS = jnp.array([...])  # (185,) point value of each meld
```

### Step 2: Check Valid Melds for Hand

```python
@jax.jit
def valid_melds(hand_mask):
    """Returns mask of which melds are valid for this hand."""
    # Meld is valid if all its cards are in hand
    # (meld & hand) == meld  <=>  meld is subset of hand
    return jnp.all(MELD_MASKS <= hand_mask, axis=1)  # (185,) bool
```

### Step 3: Enumerate Meld Combinations (Depth 3)

Since max 3 melds, we can enumerate all valid combinations:

```python
@jax.jit
def min_deadwood(hand_mask):
    total_points = hand_points(hand_mask)
    valid = valid_melds(hand_mask)  # (185,) bool

    best = total_points  # No melds = all deadwood

    # Try each single meld
    for m1 in range(NUM_MELDS):
        if not valid[m1]:
            continue
        remaining1 = hand_mask & ~MELD_MASKS[m1]
        dead1 = total_points - MELD_POINTS[m1]
        best = min(best, dead1)

        # Try each second meld (non-overlapping)
        valid2 = valid_melds(remaining1)
        for m2 in range(m1+1, NUM_MELDS):
            if not valid2[m2]:
                continue
            remaining2 = remaining1 & ~MELD_MASKS[m2]
            dead2 = dead1 - MELD_POINTS[m2]
            best = min(best, dead2)

            # Try each third meld (non-overlapping)
            valid3 = valid_melds(remaining2)
            for m3 in range(m2+1, NUM_MELDS):
                if not valid3[m3]:
                    continue
                dead3 = dead2 - MELD_POINTS[m3]
                best = min(best, dead3)

    return best
```

**Problem:** This has Python loops, not JIT-able!

### Step 4: Vectorize the Search

**Option A: Precompute Valid Meld Pairs/Triples**

Precompute which meld pairs are non-overlapping:
```python
# (185, 185) bool matrix: can these two melds coexist?
MELD_COMPAT = (MELD_MASKS[:, None, :] & MELD_MASKS[None, :, :]).sum(axis=2) == 0
```

Then for a hand:
```python
valid = valid_melds(hand)  # (185,)
valid_pairs = valid[:, None] & valid[None, :] & MELD_COMPAT  # (185, 185)
# Find best pair...
```

**Problem:** 185² = 34k pairs, 185³ = 6.3M triples. Still large.

**Option B: Greedy with Local Search**

1. Find best single meld (greedy)
2. Find best second meld (greedy)
3. Find best third meld (greedy)
4. Try swapping melds to find better solution

Not optimal but O(n) per hand.

**Option C: DP on Card Subsets**

Since hand is only 10-11 cards, use DP on the 2^10 or 2^11 subsets:

```python
# dp[mask] = min deadwood achievable using only cards in mask
# For each mask, try removing each valid meld

# This is 2048 states × ~185 melds = 379k operations per hand
# But it's parallelizable!
```

**Option D: Constraint - Only Consider "Useful" Melds**

For a 10-card hand, there are at most ~20-30 melds that could be formed (most melds require cards not in hand).

```python
# For each hand, find which melds are possible
possible_melds = valid_melds(hand)  # Usually ~10-30 true
# Then enumerate combinations of these few melds
```

With ~20 possible melds: C(20,1) + C(20,2) + C(20,3) = 20 + 190 + 1140 = 1350 combinations.

This is tractable!

## Recommended Approach: Option D

### Implementation Plan

1. **Precompute meld data:**
   ```python
   MELD_MASKS: (185, 52) int8
   MELD_POINTS: (185,) int32
   MELD_SIZES: (185,) int8
   ```

2. **For each hand, find valid melds:**
   ```python
   valid = (MELD_MASKS <= hand).all(axis=1)  # (185,) bool
   valid_indices = jnp.where(valid, size=MAX_VALID_MELDS, fill_value=-1)
   ```

3. **Enumerate combinations using vmap:**
   ```python
   # Use jax.lax.scan or fixed unrolling to try combinations
   # Key: limit to MAX_VALID_MELDS (e.g., 30) possible melds per hand
   ```

4. **Parallel over batch:**
   ```python
   batched_min_deadwood = jax.vmap(min_deadwood)
   ```

### Complexity Analysis

- Valid melds per hand: ~10-30 (call it V)
- Combinations to check: V + V² + V³ ≈ V³ ≈ 27,000 worst case
- Per combination: O(1) overlap check + O(1) deadwood calc
- Total per hand: ~30k operations
- Batch of 100k hands: 3B operations
- T4 GPU: ~8 TFLOPS = 8e12 ops/sec
- Time: 3e9 / 8e12 = 0.4ms per batch

**This should be fast enough!**

## Alternative: Neural Deadwood Estimator

Train a small neural network to estimate deadwood:
- Input: 52-bit hand encoding
- Output: deadwood estimate (0-100)
- Train on exact deadwood from C++ implementation

Pros:
- Constant time O(1) per hand
- Trivially batchable

Cons:
- Not exact (but could be within ±1)
- Need to generate training data
- For knock decision, need to be conservative (underestimate deadwood)

## Next Steps

1. [ ] Implement meld enumeration (Option D)
2. [ ] Test on sample hands against C++ implementation
3. [ ] Benchmark batched deadwood calculation
4. [ ] Integrate into step() for knock legality
5. [ ] Add gin detection (deadwood = 0)
6. [ ] Add layoff phase
7. [ ] Add proper scoring

## Open Questions

1. How does C++ handle meld disambiguation? (e.g., 3♠4♠5♠ could be part of different runs)
2. Does the original track which cards are in melds, or just the deadwood value?
3. For layoff, do we need to know the opponent's melds explicitly?

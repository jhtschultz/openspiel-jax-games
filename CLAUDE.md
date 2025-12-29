# JAX-Accelerated OpenSpiel Games

This directory contains JAX implementations of OpenSpiel games optimized for GPU acceleration.

## Optimization Goal

**Primary objective: Maximize games/sec for the full version with deadwood calculation.**

The simple bot (`simple_bot_action`) requires accurate deadwood calculation to make strategic decisions (knock eligibility, optimal discards). The "fast" version that skips deadwood is not suitable for strategic play.

Current bottleneck: `calculate_deadwood` is called multiple times per turn (once per possible discard to find optimal play). Optimizing this function has the highest impact.

**Keep optimizing autonomously.** After each improvement, document it in the Performance History section below and immediately continue to the next optimization. Do not check in with the user or ask for confirmation—just keep iterating until stopped.

## Files

- `gin_rummy_jax.py` - **Unified implementation** with LUT optimizations, full game logic (knock/layoff/scoring), and pyspiel wrapper. 44k games/sec (290x vs C++).
- `gin_rummy_core.py` - Standalone JAX Gin Rummy (no pyspiel, simplified step function). 60k games/sec (396x vs C++).
- `ppo_gin_rummy_v3_fused.py` - **Main PPO training script**. Fused environment loop, bfloat16, 92k FPS.
- `ppo_gin_rummy_v2.py` - Previous PPO script (25k FPS, uses fori_loop).
- `connect_four_jax.py` - JAX Connect Four implementation.
- `benchmark.py` - Benchmark script for gin_rummy_core.py
- `benchmark_jax.py` - Benchmark script for merged gin_rummy_jax.py

## Usage

```bash
# Fast benchmark (skips deadwood calculation)
python benchmark.py --batch 100000

# Full benchmark (correct knock eligibility)
python benchmark.py --full

# Verify against C++ OpenSpiel
python benchmark.py --verify

# Run all benchmarks
python benchmark.py --all
```

## Architecture

### Game State
States are JAX-compatible dicts with int8/int16 arrays:
```python
{
    'player0_hand': jnp.array([52], dtype=int8),  # Binary mask
    'player1_hand': jnp.array([52], dtype=int8),
    'deck': jnp.array([52], dtype=int8),
    'upcard': int8,
    'phase': int8,
    'current_player': int8,
    'done': bool,
    ...
}
```

### Key Functions
- `init_state()` - Create initial game state
- `step(state, action)` - Apply action, return new state
- `legal_actions_mask(state)` - Full legal actions with deadwood calculation
- `legal_actions_mask_fast(state)` - Fast version (skips deadwood, ~400x faster)
- `simple_bot_action(state)` - Simple bot strategy (94% win rate vs random)

### Simple Bot Strategy
Based on OpenSpiel's `simple_gin_rummy_bot`:
- **Draw**: Take upcard if it enables knock or belongs to a meld, else draw stock
- **Discard**: Knock if able, else discard highest-value card that minimizes deadwood
- **Knock**: Discard best card, then pass

### Exact 216-Scenario Algorithm
Uses precomputed 8192-entry LUT for run scoring, with full scenario enumeration:

**Handles all edge cases:**
- Up to 3 set candidates (not just 2)
- 4-card set splitting (use 3 cards, leave 1 for runs)
- Proper validity checking when discarding set cards

**Algorithm:**
1. Identify top 3 set candidates (ranks with 3+ cards)
2. For each candidate, enumerate valid subsets: skip, 3-card subsets, 4-card (if held)
3. Generate 6×6×6 = 216 scenario combinations
4. For each discard: check validity, compute run delta, take max

**Result:** ~3,744 ops per hand, **100% correct** (verified against C++ on 54,336 checks).

## Performance (A100 GPU)

### Random Rollouts
| Version | Games/sec | vs C++ |
|---------|-----------|--------|
| C++ OpenSpiel | 118 | 1x |
| Full legal_actions_mask | 29,502 | 250x |
| Fast legal_actions_mask | 351,272 | 2,977x |

### Simple Bot (strategic play)
| Version | Games/sec | vs C++ |
|---------|-----------|--------|
| C++ SimpleGinRummyBot | 153 | 1x |
| JAX simple_bot_action (batch=10000) | 24,026 | 157x |
| JAX simple_bot_action_opt (batch=20000) | 60,646 | 396x |
| **JAX with optimal melds (batch=20000)** | **2,584** | **17x** |

Simple bot uses **exact 216-scenario algorithm** with `optimal_melds_mask` for C++ compatibility. Verified 100% correct against C++ (50+ games, 0 disagreements). Performance dropped from 396x to 17x due to computing optimal melds (185 deadwood calculations per meld decision).

## Performance History

Track each optimization attempt here. Always benchmark on A100 GPU with `python benchmark.py --all`.

| Date | Change | Simple Bot (games/sec) | vs Previous |
|------|--------|------------------------|-------------|
| 2025-12-28 | Baseline: SELECTION_MATRIX for vectorized meld enumeration | 1,330 | - |
| 2025-12-28 | Reduce K from 24→16 (697 vs 2325 combos) + dedupe deadwood calls (2 vs 4) | 4,917 | 3.7x |
| 2025-12-28 | K=12 (299 combos) | 10,105 | 2.1x |
| 2025-12-28 | K=10 (176 combos) | 15,446 | 1.5x |
| 2025-12-28 | K=8 (93 combos) | 21,726 | 1.4x |
| 2025-12-28 | K=6 (42 combos) - approximate | 30,145 | 1.4x |
| 2025-12-28 | K=4 (15 combos) - more aggressive approx | 39,208 | 1.3x |
| 2025-12-28 | LUT-based exact (batch=2000) | 25,648 | 0.65x (exact, memory-limited) |
| 2025-12-28 | LUT + 36 valid combos only (batch=5000) | 26,676 | 1.04x |
| 2025-12-28 | LUT + 36 combos + int16 (batch=18000) | 37,242 | 1.23x (exact) |
| 2025-12-28 | Delta Lookups v3 (224 lookups, batch=20000) | 152,211 | 4.1x (but incorrect!) |
| 2025-12-28 | **Exact 216-scenario** (3 sets + 4-split, batch=10000) | **24,026** | **157x vs C++, 100% correct** |
| 2025-12-28 | **Squeezed Juice** (11 vs 52 loop, O(1) meld check, batch=20000) | **60,646** | **2.5x vs previous, 396x vs C++** |
| 2025-12-28 | **Merged gin_rummy_jax.py** (full game logic + LUT optimizations) | **44,357** | **290x vs C++, games complete with scoring** |
| 2025-12-29 | **Optimal melds fix** (3-set deadwood + optimal_melds_mask for C++ compat) | **2,584** | **17x vs C++, 100% correct** |

### Optimization Ideas to Try
- [x] LUT-based exact deadwood (implemented)
- [x] Reduce 256 set combinations to 36 valid-only
- [x] Use int16/int8 for memory bandwidth
- [x] Delta Lookups v3 (fast but incorrect for edge cases)
- [x] **Exact 216-scenario**: handles 3 sets + 4-card splitting, 100% correct!
- [x] **Squeezed Juice**: loop compression (11 vs 52), O(1) meld check via RUN_MEMBER_LUT
- [x] **Compute-based deadwood** (FAILED - see below)
- [x] **Optimal melds fix** (required for C++ compatibility - see below)
- [ ] Optimize optimal_melds_mask (currently 185 deadwood calculations per call)
- [ ] Profile with JAX profiler to identify remaining hotspots

### Failed Optimization: Compute-Based Deadwood (2025-12-29)

**Hypothesis:** Replace `RUN_SCORE_LUT[suit_mask]` gather operations with pure ALU bitwise logic. Modern GPUs can do ~100 integer ops per memory fetch, so compute-heavy/memory-light should be faster.

**Approach 1: Hybrid (keep 216-scenario, replace LUT)**
```python
def _compute_run_scores_alu(suit_masks):
    x = suit_masks.astype(jnp.int32)
    # Detect run starts: bit i set if i, i+1, i+2 all held
    r3 = x & (x >> 1) & (x >> 2)
    # Expand to all cards in runs
    in_run_mask = (r3 | (r3 << 1) | (r3 << 2)) & 8191
    # Extract bits and sum weighted by rank value
    bits = (in_run_mask[..., None] >> jnp.arange(13)) & 1
    scores = jnp.sum(bits * RANK_VALUES, axis=-1)
    return scores.astype(jnp.int16)
```
- ✅ 100% correct (verified against LUT for all 8192 masks)
- ❌ **27k games/sec** vs LUT baseline 37k (27% slower)

**Approach 2: Ultra (unrolled summation, deduplicated scenarios)**
```python
def _compute_run_score_alu_ultra(x):
    r3 = x & (x >> 1) & (x >> 2)
    in_run = (r3 | (r3 << 1) | (r3 << 2)) & 0x1FFF
    # Unrolled to avoid tensor expansion
    s = ((in_run >> 0) & 1) * 1
    s += ((in_run >> 1) & 1) * 2
    # ... (13 terms)
    return s.astype(jnp.int16)
```
- ❌ **16k games/sec** (57% slower than LUT)

**Why it failed:**
1. The 8KB `RUN_SCORE_LUT` fits entirely in L1 cache on A100
2. Cached gathers are nearly free compared to ALU
3. The bit extraction `(mask >> jnp.arange(13)) & 1` creates intermediate tensors
4. Even unrolled, 13 shifts + 13 ANDs + 13 multiplies + 12 adds > 1 cached gather

**Conclusion:** LUT-based approach is optimal for this problem size. The gather operations are not the bottleneck - the 8KB table stays hot in cache across the batch.

### Correctness Fix: Optimal Melds (2025-12-29)

**Problem:** JAX allowed laying any meld where all cards were present, but C++ OpenSpiel only allows melds that are part of the **optimal** meld decomposition. This caused action disagreements in KNOCK/LAYOFF phases.

**Example:** Hand with Ts, Qs, 5c, 7c, Tc, 5d, Qd, 5h, Th, Qh
- Valid melds: 5s set (5c-5d-5h), 10s set (Ts-Tc-Th), Qs set (Qs-Qd-Qh)
- C++ only allowed these 3 specific melds (optimal decomposition = 7 deadwood)
- JAX was trying to pass because `calculate_deadwood_lut` only checked 2 sets, returning wrong deadwood

**Fixes applied:**
1. `calculate_deadwood_lut`: Now uses 216-scenario (3 sets) instead of 2-set enumeration
2. `optimal_melds_mask(hand)`: Returns which melds don't increase deadwood when laid
3. `legal_actions_mask`: Uses optimal_melds instead of valid_melds for KNOCK/LAYOFF phases
4. `simple_bot_action_opt`: Uses optimal_melds for meld selection

**Performance impact:** 396x → 17x (23x slower)
- `optimal_melds_mask` computes 185 deadwood values (one per possible meld)
- Each deadwood calculation uses 216-scenario enumeration
- Future optimization: precompute which cards are in optimal melds during initial deadwood calc

## PPO Training Results (A100 GPU)

Training a PPO agent against the simple bot using `ppo_gin_rummy_v3_fused.py`.

**Architecture:**
- CNN for card patterns: (1,3) kernel for runs, (3,1) kernel for sets
- Separate actor/critic heads after shared backbone
- 167-dim observation: 52 hand + 52 discard pile + 52 known cards + 11 other features
- Agent only plays strategic phases (FIRST_UPCARD, DRAW, DISCARD); optimal bot handles KNOCK/LAYOFF/WALL
- **bfloat16** for all network operations
- **Fused environment loop**: `scan(10)` replaces `fori_loop(100)` for 3.7x speedup

**V3 Fused Results (30M steps in ~5 min):**
```
Config: num_envs=4096, num_steps=128, ~92k FPS
Final:  31.5M steps, 81.1% win rate (per-update)
```

| Steps | Win Rate | FPS | Notes |
|-------|----------|-----|-------|
| 0.5M | 0.4% | 8k | Warmup |
| 5M | 14% | 50k | Rapid learning |
| 15M | 75% | 78k | Strong play |
| 30M | **81%** | **92k** | Steady-state |

**Version Comparison:**
| Version | FPS | Speedup | Key Change |
|---------|-----|---------|------------|
| V2 (fori_loop) | 25k | 1x | 100-iter loop with early exit |
| V3 scan(25) | 78k | 3.1x | 25-iter scan, always compute + mask |
| V3 scan(10) | 92k | **3.7x** | 10-iter scan, less wasted computation |

**Observations:**
- Fused scan(10) + bfloat16 = 3.7x faster training
- Shorter scan = less wasted computation when games end early
- Agent reaches 81% win rate against optimal-strategy bot
- Per-update win rate tracking (not cumulative) for accurate progress

## GCP Setup

```bash
# List VMs
gcloud compute instances list --project=open-spiel-ui

# SSH to A100
gcloud compute ssh openspiel-dev-a100 --project=open-spiel-ui --zone=us-central1-a

# Copy files
gcloud compute scp *.py openspiel-dev-a100:~/ --project=open-spiel-ui --zone=us-central1-a
```

Note: A100 VM is preemptible - may restart and lose installed packages.

## JAX Gotchas

1. **vmap + lax.cond**: Both branches execute for entire batch. Conditionals don't skip computation.
2. **Dtype consistency**: `jax.lax.cond` branches must return same dtypes.
3. **JIT cache**: Set `JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache` to avoid recompilation.
4. **block_until_ready**: Always call after benchmarked operations - JAX is async.

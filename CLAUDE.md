# JAX-Accelerated OpenSpiel Games

This directory contains JAX implementations of OpenSpiel games optimized for GPU acceleration.

## Optimization Goal

**Primary objective: Maximize games/sec for the full version with deadwood calculation.**

The simple bot (`simple_bot_action`) requires accurate deadwood calculation to make strategic decisions (knock eligibility, optimal discards). The "fast" version that skips deadwood is not suitable for strategic play.

Current bottleneck: `calculate_deadwood` is called multiple times per turn (once per possible discard to find optimal play). Optimizing this function has the highest impact.

**Keep optimizing autonomously.** After each improvement, document it in the Performance History section below and immediately continue to the next optimization. Do not check in with the user or ask for confirmation—just keep iterating until stopped.

## Directory Structure

```
work/
  # Game implementations
  gin_rummy_jax.py      # Unified implementation with LUT optimizations (290x vs C++)
  gin_rummy_core.py     # Standalone JAX Gin Rummy, no pyspiel (396x vs C++)

  # Shared modules (extracted for reuse)
  constants.py          # Shared constants (card encoding, phases, actions)
  gin_rummy_luts.py     # Precomputed lookup tables for deadwood
  gin_rummy_melds.py    # Meld generation and encoding
  gin_rummy_deadwood.py # Deadwood calculation functions

  # Training & evaluation
  ppo_gin_rummy_v3_fused.py  # Main PPO training (92k FPS)
  evaluate_checkpoint.py     # Evaluation with action history export
  compare_bots.py       # JAX vs C++ bot verification
  benchmark.py          # Benchmark for gin_rummy_core
  benchmark_jax.py      # Benchmark for gin_rummy_jax

  # Debugging utilities
  trace_pyspiel.py      # Game tracing for debugging
  debug_bot.py          # Bot debugging utilities
  debug_game.py         # Game state debugging

  # Testing & output
  tests/                # Pytest test suite
  output/               # Generated data files (JSONL, JSON)
  _archive/             # Obsolete/experimental code (see below)
```

### Tests

Run tests with pytest:
```bash
pytest tests/ -v                      # All tests
pytest tests/test_deadwood.py -v      # Deadwood calculation tests
pytest tests/test_bot_equivalence.py  # Bot vs C++ comparison (requires pyspiel)
pytest tests/ -m "not performance"    # Skip slow performance tests
```

### Archived Files

Located in `_archive/`, kept for reference:
- `_archive/ppo/` - Old PPO versions (v1, v2) superseded by v3_fused
- `_archive/debug/` - One-off debug scripts (debug_ppo*.py)
- `_archive/experiments/` - Incomplete experiments (connect_four_jax.py)
- `_archive/test_*.py` - Old manual test scripts (replaced by pytest suite)

## Active Files

### Game Implementations
- `gin_rummy_jax.py` - **Unified implementation** with LUT optimizations, full game logic (knock/layoff/scoring), and pyspiel wrapper. 44k games/sec (290x vs C++).
- `gin_rummy_core.py` - Standalone JAX Gin Rummy (no pyspiel, simplified step function). 60k games/sec (396x vs C++).

### Shared Modules
- `constants.py` - Shared constants (card encoding, phases, actions).
- `gin_rummy_luts.py` - Precomputed lookup tables (RUN_SCORE_LUT, SUBSET_TABLE, etc.).
- `gin_rummy_melds.py` - Meld generation and encoding (ALL_MELDS, MELD_MASKS, MELD_POINTS).
- `gin_rummy_deadwood.py` - Deadwood calculation functions (calculate_deadwood_lut, etc.).

### Training & Evaluation
- `ppo_gin_rummy_v3_fused.py` - **Main PPO training script**. Fused environment loop, bfloat16, 92k FPS.
- `evaluate_checkpoint.py` - **Evaluation script**. Runs games with trained checkpoint, saves action histories to `output/`.
- `compare_bots.py` - Compare JAX bot vs C++ SimpleGinRummyBot for verification.
- `benchmark.py` - Benchmark script for gin_rummy_core.py
- `benchmark_jax.py` - Benchmark script for gin_rummy_jax.py

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

# Evaluate trained checkpoint (saves game histories to JSONL)
python evaluate_checkpoint.py --n-games 100 --output games.jsonl

# Compare JAX bot vs C++ bot
python compare_bots.py --n-games 500
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

### C++ SimpleGinRummyBot Reference

**Source code**: https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/bots/gin_rummy

**Draw Phase Logic** (from `Step` function):
```cpp
// Draw upcard if doing so permits a knock, or if the upcard would not be
// in the "best" deadwood (=> upcard would be in a "best" meld).
if (utils_.MinDeadwood(hand, upcard) <= knock_card ||
    !absl::c_linear_search(GetBestDeadwood(hand, upcard), upcard)) {
  return kDrawUpcardAction;
}
```
- `GetBestDeadwood(hand, upcard)` returns cards NOT in the optimal meld group
- If upcard is NOT in this list → upcard IS part of an optimal meld → take it

**Discard Logic** (from `GetDiscard` function):
```cpp
std::vector<int> deadwood = GetBestDeadwood(hand);
std::sort(deadwood.begin(), deadwood.end(), RankComparator(kDefaultNumRanks));
return deadwood.back();
```
- Get deadwood (cards not in optimal melds)
- Sort by `RankComparator`: primary=rank (card % 13), secondary=card index
- Return `back()` = highest rank, then highest card index for ties

**RankComparator** (from `gin_rummy_utils.h`):
```cpp
struct RankComparator {
  int CardRank(int card) { return card % num_ranks; }
  bool operator()(int card_1, int card_2) {
    if (CardRank(card_1) == CardRank(card_2)) {
      return card_1 < card_2;  // Same rank: sort by card index
    }
    return CardRank(card_1) < CardRank(card_2);  // Different rank: sort by rank
  }
};
```

**Meld Selection** (from `GetMelds` function):
```cpp
for (const auto& meld : utils_.BestMeldGroup(hand)) {
  rv.push_back(utils_.meld_to_int.at(meld));
}
```
- Returns melds in the order from `BestMeldGroup`
- No specific ordering applied

**Known Bug: GIN Layoff Phase**
When opponent knocks with GIN, the C++ bot queues actions: `[pass, melds..., pass]`
```cpp
if (!layed_melds.empty()) {
  next_actions_.push_back(kPassAction);  // Bot never lays off.
  for (int meld_id : GetMelds(hand)) {
    next_actions_.push_back(kMeldActionBase + meld_id);
  }
  next_actions_.push_back(kPassAction);
}
```
However, when facing GIN, `finished_layoffs=true` in the game, so the first PASS ends the game immediately before the bot can lay its melds. This causes the defender's entire hand to be counted as deadwood instead of just the remaining cards after laying melds.

**JAX bot behavior**: Correctly lays melds when facing GIN, resulting in lower opponent deadwood and different scores. This is intentional - we don't match the C++ bot's buggy behavior.

**Tie-breaking in BestMeldGroup** (from `gin_rummy_utils.cc`):
```cpp
// AllMelds returns: RankMelds first, then SuitMelds
VecVecInt rank_melds = RankMelds(cards);
VecVecInt suit_melds = SuitMelds(cards);
rank_melds.insert(rank_melds.end(), suit_melds.begin(), suit_melds.end());

// BestMeldGroup picks FIRST group with max value (uses > not >=)
if (meld_group_total_value > best_meld_group_total_value) {
  best_meld_group = meld_group;
}
```
- **Key insight**: When meld points are tied, SETS are preferred over RUNS
- Because: `AllMelds` returns rank melds (sets) before suit melds (runs)
- JAX must match: use set_card_count as tie-breaker in `_compute_optimal_meld_info`

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
| **JAX with optimal melds v2 (batch=20000)** | **32,382** | **212x** |

Simple bot uses **exact 216-scenario algorithm** with `optimal_melds_mask` for C++ compatibility. Verified 100% correct against C++ (200+ games, 0 disagreements). Uses `RUN_DECOMP_LUT` for O(1) run meld decomposition.

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
| 2025-12-29 | Optimal melds v1 (3-set deadwood + optimal_melds_mask) | 2,584 | 17x vs C++ (slow due to 185 deadwood calcs) |
| 2025-12-29 | **Optimal melds v2** (RUN_DECOMP_LUT for O(1) meld decomposition) | **32,382** | **212x vs C++, 100% correct** |

### Optimization Ideas to Try
- [x] LUT-based exact deadwood (implemented)
- [x] Reduce 256 set combinations to 36 valid-only
- [x] Use int16/int8 for memory bandwidth
- [x] Delta Lookups v3 (fast but incorrect for edge cases)
- [x] **Exact 216-scenario**: handles 3 sets + 4-card splitting, 100% correct!
- [x] **Squeezed Juice**: loop compression (11 vs 52), O(1) meld check via RUN_MEMBER_LUT
- [x] **Compute-based deadwood** (FAILED - see below)
- [x] **Optimal melds fix** (required for C++ compatibility - see below)
- [x] **RUN_DECOMP_LUT** for O(1) run meld decomposition (solved the 185 deadwood problem)
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

**Example 1:** Hand with Ts, Qs, 5c, 7c, Tc, 5d, Qd, 5h, Th, Qh
- Valid melds: 5s set (5c-5d-5h), 10s set (Ts-Tc-Th), Qs set (Qs-Qd-Qh)
- C++ only allowed these 3 specific melds (optimal decomposition = 7 deadwood)
- JAX was trying to pass because `calculate_deadwood_lut` only checked 2 sets

**Example 2:** Hand with 5c-6c-7c-8c-9c-Tc (6-card clubs run)
- C++ allowed: 5c-6c-7c (meld #80) + 8c-9c-Tc (meld #83) = two 3-card runs
- JAX was trying: 6c-7c-8c-9c-Tc (5-card run) - leaves 5c as deadwood!

**Fixes applied:**
1. `calculate_deadwood_lut`: Now uses 216-scenario (3 sets) instead of 2-set enumeration
2. `RUN_DECOMP_LUT`: New (8192, 5) LUT that decomposes runs into valid melds (e.g., 6-card → 3+3)
3. `_compute_optimal_meld_info`: Returns exact meld indices, not just melded card mask
4. `optimal_melds_mask(hand)`: Returns which specific melds are in the optimal decomposition
5. `legal_actions_mask` and `simple_bot_action_opt`: Use optimal melds for KNOCK/LAYOFF

**Performance:** 396x → 17x (v1, slow) → **212x** (v2, with RUN_DECOMP_LUT)
- v1 computed 185 deadwood values per call (one per possible meld)
- v2 uses O(1) LUT lookups to get exact meld IDs

## PPO Training Results (A100 GPU)

Training a PPO agent against the simple bot using `ppo_gin_rummy_v3_fused.py`.

**Architecture:**
- CNN for card patterns: (1,3) kernel for runs, (3,1) kernel for sets
- Separate actor/critic heads after shared backbone
- 167-dim observation: 52 hand + 52 discard pile + 52 known cards + 11 other features
- Agent only plays strategic phases (FIRST_UPCARD, DRAW, DISCARD); optimal bot handles KNOCK/LAYOFF/WALL
- **bfloat16** for all network operations
- **Fused environment loop**: `scan(10)` replaces `fori_loop(100)` for 3.7x speedup
- **Orbax checkpointing**: saves/restores training state for resumable training

### Training Commands

```bash
# Start training (auto-resumes from checkpoint if exists)
python ppo_gin_rummy_v3_fused.py

# Start fresh, ignoring existing checkpoints
python ppo_gin_rummy_v3_fused.py --fresh

# Custom step count
python ppo_gin_rummy_v3_fused.py --steps 100000000

# Custom checkpoint directory
python ppo_gin_rummy_v3_fused.py --checkpoint-dir ./my_checkpoints
```

Checkpoints are saved every 10 updates to `./checkpoints/ppo_gin_rummy_v3/`. The last 3 checkpoints are kept.

**V3 Fused Results (10 min on A100):**
```
Config: num_envs=4096, num_steps=128
Final:  34.6M steps, 79.7% win rate, 61k FPS
```

| Steps | Win Rate | FPS | Notes |
|-------|----------|-----|-------|
| 0.5M | 0.3% | 7k | Warmup/JIT compile |
| 5M | 14% | 37k | Rapid learning |
| 10M | 65% | 48k | Strong play |
| 20M | 77% | 57k | Near convergence |
| 35M | **80%** | **61k** | Steady-state |

**Version Comparison:**
| Version | FPS | Speedup | Key Change |
|---------|-----|---------|------------|
| V2 (fori_loop) | 25k | 1x | 100-iter loop with early exit |
| V3 scan(10) | 61k | **2.4x** | 10-iter scan, always compute + mask |

**Observations:**
- Fused scan(10) + bfloat16 = 2.4x faster training vs V2
- Agent reaches ~80% win rate against optimal-strategy bot in 10 minutes
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

## Bug Fixes

### 2024-12-29: gin_rummy_core PHASE_KNOCK handling

**Problem:** Games using `gin_rummy_core.py` would get stuck in an infinite loop at PHASE_KNOCK (phase 4). The bot returned discard/pass actions, but the step function ignored them.

**Root Cause:** The `step()` function was missing the handler for PHASE_KNOCK. When a player knocked:
1. Game entered PHASE_KNOCK
2. Bot returned valid actions (discard card, then pass)
3. `step()` had no code for this phase → actions ignored → infinite loop

**Fix:** Added PHASE_KNOCK handling (lines 1115-1142):
- Process discard actions (player with 11 cards discards to 10)
- Process pass action → calculate deadwoods, determine winner, transition to GAME_OVER
- Fixed `new_done` propagation bug (was being overwritten with `state['done']`)

**Verification:** All 54 tests pass (25 game completion + 14 bot equivalence + 15 deadwood).

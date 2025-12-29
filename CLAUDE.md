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
- `ppo_gin_rummy_v3_fused.py` - **Main PPO training script**. Fused environment loop, bfloat16, 78k FPS.
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
| **JAX simple_bot_action_opt (batch=20000)** | **60,646** | **396x** |

Simple bot uses **exact 216-scenario algorithm** - verified 100% correct against C++ (54,336 checks, 0 disagreements). Optimized version uses loop compression (11 vs 52 iterations) and O(1) meld check via LUT.

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

### Optimization Ideas to Try
- [x] LUT-based exact deadwood (implemented)
- [x] Reduce 256 set combinations to 36 valid-only
- [x] Use int16/int8 for memory bandwidth
- [x] Delta Lookups v3 (fast but incorrect for edge cases)
- [x] **Exact 216-scenario**: handles 3 sets + 4-card splitting, 100% correct!
- [x] **Squeezed Juice**: loop compression (11 vs 52), O(1) meld check via RUN_MEMBER_LUT
- [ ] Profile with JAX profiler to identify remaining hotspots

## PPO Training Results (A100 GPU)

Training a PPO agent against the simple bot using `ppo_gin_rummy_v3_fused.py`.

**Architecture:**
- CNN for card patterns: (1,3) kernel for runs, (3,1) kernel for sets
- Separate actor/critic heads after shared backbone
- 167-dim observation: 52 hand + 52 discard pile + 52 known cards + 11 other features
- Agent only plays strategic phases (FIRST_UPCARD, DRAW, DISCARD); optimal bot handles KNOCK/LAYOFF/WALL
- **bfloat16** for all network operations
- **Fused environment loop**: `scan(25)` replaces `fori_loop(100)` for 3.1x speedup

**V3 Fused Results (50M steps in ~10 min):**
```
Config: num_envs=4096, num_steps=128, ~78k FPS
Final:  49.8M steps, 83.5% win rate (per-update)
```

| Steps | Win Rate | FPS | Notes |
|-------|----------|-----|-------|
| 0.5M | 0.5% | 8k | Warmup |
| 5M | 15% | 44k | Rapid learning |
| 15M | 75% | 65k | Strong play |
| 50M | **83.5%** | **78k** | Steady-state |

**Version Comparison:**
| Version | FPS | Speedup | Key Change |
|---------|-----|---------|------------|
| V2 (fori_loop) | 25k | 1x | 100-iter loop with early exit |
| V3 (fused scan) | 78k | **3.1x** | 25-iter scan, always compute + mask |

**Observations:**
- Fused scan + bfloat16 = 3.1x faster training
- Agent reaches 83% win rate against optimal-strategy bot
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

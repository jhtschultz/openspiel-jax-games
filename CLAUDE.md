# JAX-Accelerated OpenSpiel Games

This directory contains JAX implementations of OpenSpiel games optimized for GPU acceleration.

## Files

- `gin_rummy_core.py` - Standalone JAX Gin Rummy (no pyspiel). Use for GPU benchmarks.
- `gin_rummy_jax.py` - Gin Rummy with pyspiel wrapper. Matches C++ OpenSpiel exactly.
- `connect_four_jax.py` - JAX Connect Four implementation.
- `benchmark.py` - Benchmark script with C++ verification (`--verify` flag).

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

### SELECTION_MATRIX Optimization
Precomputed (2325, 24) matrix enumerating all subsets of size 0-3 from K=24 melds.
Enables vectorized deadwood calculation via dense matrix ops instead of O(32³) loops.

## Performance (A100 GPU)

### Random Rollouts
| Version | Games/sec | vs C++ |
|---------|-----------|--------|
| C++ OpenSpiel | 118 | 1x |
| Full legal_actions_mask | 1,333 | 11x |
| Fast legal_actions_mask | 343,670 | 2,912x |

### Simple Bot (strategic play)
| Version | Games/sec | vs C++ |
|---------|-----------|--------|
| C++ SimpleGinRummyBot | 153 | 1x |
| JAX simple_bot_action (batched) | 1,330 | 8.7x |

Fast version is ideal for MCTS rollouts. Simple bot for strategic play evaluation.

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

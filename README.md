# JAX-Accelerated OpenSpiel Games

This directory contains JAX implementations of OpenSpiel games optimized for massively parallel GPU simulation.

## Results Summary

| Game | Original C++ | JAX Batched | Speedup | Max Parallel |
|------|-------------|-------------|---------|--------------|
| Connect Four | 4,664/sec | 3,750,234/sec | **804x** | 16k+ |
| Gin Rummy | 146/sec | 4,245/sec | **29x** | 16k+ |
| Deadwood Calc | N/A | 2,095,183/sec | N/A | 100k+ |

*Benchmarked on Tesla T4 GPU (15GB)*

**Note:** Gin Rummy includes proper meld detection for exact deadwood calculation (exhaustive search over all valid meld combinations using optimized matrix operations).

## Files

### Connect Four (Complete)
- `connect_four_jax.py` - Full implementation with OpenSpiel wrapper
- `test_agreement.py` - Verifies 1000 random games match original
- `test_wins.py` - Tests all win conditions (horizontal, vertical, diagonal, draw)
- `benchmark.py` - Performance comparison

### Gin Rummy (Complete)
- `gin_rummy_jax.py` - Full implementation with proper deadwood/meld detection
- `deadwood.py` - Standalone deadwood calculation module with tests
- `benchmark_gin.py` - Performance comparison
- `PLAN_deadwood.md` - Design document for deadwood algorithm

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  OpenSpiel Python Interface                         │
│  - GameState wrapper for compatibility              │
│  - pyspiel.register_game() integration              │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│  JAX Core (Pure Functions)                          │
│  - init_state(rng) -> state dict                    │
│  - step(state, action) -> state dict                │
│  - legal_actions_mask(state) -> bool[num_actions]   │
│  - All operations are jit-compiled                  │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│  Batched Operations (vmap)                          │
│  - batched_init, batched_step, etc.                 │
│  - Run 100k+ games in parallel on GPU               │
└─────────────────────────────────────────────────────┘
```

## Usage

### Single Game (OpenSpiel Compatible)
```python
import pyspiel
import connect_four_jax  # Registers the game

game = pyspiel.load_game("python_connect_four_jax")
state = game.new_initial_state()

while not state.is_terminal():
    action = random.choice(state.legal_actions())
    state.apply_action(action)

print(state.returns())
```

### Batched Simulation (Fast)
```python
import jax
from connect_four_jax import init_batch, batched_step, batched_legal_actions_mask

# Initialize 10,000 parallel games
key = jax.random.PRNGKey(42)
states = init_batch(10000)

# Step all games
actions = ...  # Select actions for each game
states = batched_step(states, actions)
```

## Gin Rummy: Current Status

**Verified:** 100 random games match C++ OpenSpiel implementation (see `test_jax_vs_cpp.py`)

### What's Implemented ✓
- Card dealing (internal RNG, no explicit chance nodes)
- Game phases: FirstUpcard → Draw → Discard → (repeat)
- Drawing from stock or discard pile
- Discarding cards
- **Proper deadwood calculation** with exact meld detection
- **Knock legality** (deadwood ≤ 10)
- **Gin detection** (deadwood = 0)
- **Wall phase** (stock ≤ 2 cards → pass or knock only)
- **FirstUpcard rules** (both pass → upcard to discard pile)
- **Repeated move detection** (draw upcard + discard same → tracked, both players = draw)
- OpenSpiel-compatible wrapper
- Batched GPU simulation (~4k games/sec)

### What's Missing ✗
- **Knock phase** (meld declarations after knock)
- **Layoff phase** (opponent lays off cards onto knocker's melds)
- Proper scoring (undercut bonus, gin bonus)
- Oklahoma variant support

### The Meld Problem (SOLVED!)

The hardest part was calculating minimum deadwood - finding the optimal non-overlapping meld arrangement.

**Solution:** Exhaustive search using matrix operations:
1. Precompute all 329 possible melds (65 sets + 264 runs)
2. For each hand, find valid melds (~10-30)
3. Use 2D matrix ops for pairs, 3D tensor for triples
4. Maximum 3 melds (since 3+3+3=9 cards minimum)

**Performance:** 2.1M hands/sec on Tesla T4

```python
# Key insight: at most 32 valid melds per hand
valid_indices = jnp.where(valid, size=32, fill_value=0)[0]

# 3D tensor for all valid triples - only 32³ = 32,768 elements
triple_savings = jnp.where(valid_triple,
    v_points[:, None, None] + v_points[None, :, None] + v_points[None, None, :],
    0)
```

## GCP Development Workflow

```bash
# Create GPU VM
make gcp-create-gpu-spot

# SSH in
make gcp-ssh-gpu

# On VM
source ~/workspace/venv/bin/activate
python benchmark.py

# When done
exit
make gcp-delete-gpu
```

## Cost

- Tesla T4 spot instance: ~$0.15/hr
- Sufficient for development and benchmarking
- A100 would provide ~3-5x more throughput

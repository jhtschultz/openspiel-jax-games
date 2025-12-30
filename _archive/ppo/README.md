# Archived PPO Scripts

These are older versions of the PPO training script, superseded by `ppo_gin_rummy_v3_fused.py`.

## Files

- `ppo_gin_rummy.py` - v1: Initial implementation using gin_rummy_core
- `ppo_gin_rummy_v2.py` - v2: Uses gin_rummy_jax, added opponent model tracking, 25k FPS

## Why Archived

`ppo_gin_rummy_v3_fused.py` (kept in work/) achieves 92k FPS through:
- Fused environment loop with `scan(10)` instead of `fori_loop(100)`
- bfloat16 for network operations
- Better memory efficiency

See CLAUDE.md "PPO Training Results" section for performance comparison.

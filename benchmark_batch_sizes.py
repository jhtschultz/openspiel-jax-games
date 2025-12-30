"""Batch Size Benchmark for PPO Training.

Tests different configurations of num_envs × num_steps to find optimal
memory access patterns for the A100 GPU.

The total samples per update = num_envs × num_steps. We test:
1. Fixed total samples, varying the ratio (more envs vs more steps)
2. Different total batch sizes

Usage:
    python benchmark_batch_sizes.py

Expected insights:
- More envs = better parallelism but more memory for env states
- More steps = better temporal locality but longer scan chains
- Sweet spot depends on GPU memory hierarchy
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from typing import NamedTuple
import time
import gc

# Import game logic
import gin_rummy_jax as gin

# Constants
DTYPE = jnp.bfloat16
NUM_ACTIONS = 241
OBS_DIM = 167

# Configurations to test
# Format: (num_envs, num_steps, description)
CONFIGS = [
    # Fixed ~500k samples per update (like current setup)
    (4096, 128, "baseline"),      # 524k samples
    (8192, 64, "more envs"),      # 524k samples
    (2048, 256, "more steps"),    # 524k samples
    (16384, 32, "max envs"),      # 524k samples
    (1024, 512, "max steps"),     # 524k samples

    # Smaller batches (faster iteration, less memory)
    (2048, 64, "small batch"),    # 131k samples
    (4096, 32, "small/wide"),     # 131k samples

    # Larger batches (more samples per update)
    (8192, 128, "large batch"),   # 1M samples
    (4096, 256, "large/deep"),    # 1M samples
]

WARMUP_UPDATES = 2
BENCHMARK_UPDATES = 5


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    legal_mask: jnp.ndarray


# Import env functions from v3
from ppo_gin_rummy_v3_fused import (
    env_init, env_step_fused, env_reset_if_done,
    compute_gae, ppo_loss, ActorCritic
)


def benchmark_config(num_envs: int, num_steps: int, description: str) -> dict:
    """Benchmark a single configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"  num_envs={num_envs}, num_steps={num_steps}")
    print(f"  samples/update={num_envs * num_steps:,}")

    # Force garbage collection and clear JAX caches
    gc.collect()

    try:
        # Initialize
        key = jax.random.PRNGKey(42)
        network = ActorCritic()
        key, init_key = jax.random.split(key)
        params = network.init(init_key, jnp.zeros((1, OBS_DIM), dtype=DTYPE))
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4, eps=1e-5))
        opt_state = tx.init(params)
        apply_fn = jax.jit(network.apply)

        # Initialize environments
        key, *keys = jax.random.split(key, num_envs + 1)
        env_states = jax.vmap(env_init)(jnp.stack(keys))

        # Define functions with this config's batch sizes
        @jax.jit
        def collect_rollout(params, env_states, key):
            def step_fn(carry, _):
                env_states, key = carry
                key, *keys = jax.random.split(key, num_envs + 1)
                keys = jnp.stack(keys)
                obs, masks = env_states['obs'], env_states['legal_mask']

                logits, value = apply_fn(params, obs)
                logits = jnp.where(masks, logits, -1e9)
                action = jax.vmap(lambda k, l: jax.random.categorical(k, l))(keys, logits)
                log_prob = jax.nn.log_softmax(logits)
                action_log_prob = jnp.take_along_axis(log_prob, action[:, None], axis=-1).squeeze(-1)

                next_env_states, reward = jax.vmap(env_step_fused)(env_states, action)
                dones = next_env_states['done']
                next_env_states = jax.vmap(env_reset_if_done)(next_env_states)

                return (next_env_states, key), Transition(obs, action, reward, dones, value, action_log_prob, masks)

            (env_states, key), traj = jax.lax.scan(step_fn, (env_states, key), None, num_steps)
            return env_states, traj, key

        @jax.jit
        def update(params, opt_state, traj, key):
            adv, ret = compute_gae(traj.reward, traj.value, traj.done)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            batch_size = num_steps * num_envs
            batch = (
                traj.obs.reshape(batch_size, -1),
                traj.action.reshape(batch_size),
                traj.log_prob.reshape(batch_size),
                adv.reshape(batch_size),
                ret.reshape(batch_size),
                traj.legal_mask.reshape(batch_size, -1),
                traj.value.reshape(batch_size)
            )

            # Adjust minibatch count based on batch size
            # Keep minibatch size around 64k for fair comparison
            num_minibatches = max(1, batch_size // 65536)

            def epoch_fn(carry, _):
                params, opt_state, key = carry
                key, subkey = jax.random.split(key)
                perm = jax.random.permutation(subkey, batch_size)

                def mb_fn(carry, idx):
                    params, opt_state = carry
                    mb_size = batch_size // num_minibatches
                    idx = jax.lax.dynamic_slice(perm, (idx,), (mb_size,))
                    mb = tuple(x[idx] for x in batch)
                    (l, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, apply_fn, mb)
                    updates, opt_state = tx.update(grads, opt_state, params)
                    return (optax.apply_updates(params, updates), opt_state), (l, aux)

                (params, opt_state), (l, aux) = jax.lax.scan(
                    mb_fn, (params, opt_state),
                    jnp.arange(0, batch_size, batch_size // num_minibatches)
                )
                return (params, opt_state, key), (l.mean(), {k: v.mean() for k,v in aux.items()})

            (params, opt_state, _), (l, aux) = jax.lax.scan(epoch_fn, (params, opt_state, key), None, 4)
            return params, opt_state, l.mean(), {k: v.mean() for k,v in aux.items()}

        # Warmup
        print("  Warming up...", end=" ", flush=True)
        for _ in range(WARMUP_UPDATES):
            key, r_key, u_key = jax.random.split(key, 3)
            env_states, traj, _ = collect_rollout(params, env_states, r_key)
            params, opt_state, loss, aux = update(params, opt_state, traj, u_key)
            jax.block_until_ready(params)
        print("done")

        # Benchmark
        print("  Benchmarking...", end=" ", flush=True)
        start = time.perf_counter()
        total_steps = 0
        for _ in range(BENCHMARK_UPDATES):
            key, r_key, u_key = jax.random.split(key, 3)
            env_states, traj, _ = collect_rollout(params, env_states, r_key)
            params, opt_state, loss, aux = update(params, opt_state, traj, u_key)
            jax.block_until_ready(params)
            total_steps += num_envs * num_steps
        elapsed = time.perf_counter() - start
        print("done")

        fps = total_steps / elapsed
        ms_per_update = elapsed / BENCHMARK_UPDATES * 1000

        print(f"  Result: {fps:,.0f} FPS, {ms_per_update:.1f}ms/update")

        return {
            "num_envs": num_envs,
            "num_steps": num_steps,
            "description": description,
            "samples_per_update": num_envs * num_steps,
            "fps": fps,
            "ms_per_update": ms_per_update,
            "success": True,
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        return {
            "num_envs": num_envs,
            "num_steps": num_steps,
            "description": description,
            "samples_per_update": num_envs * num_steps,
            "fps": 0,
            "ms_per_update": 0,
            "success": False,
            "error": str(e),
        }


def main():
    print("Batch Size Benchmark for PPO Training")
    print(f"Devices: {jax.devices()}")
    print(f"Testing {len(CONFIGS)} configurations...")

    results = []
    for num_envs, num_steps, desc in CONFIGS:
        result = benchmark_config(num_envs, num_steps, desc)
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Config':<20} {'Envs':>8} {'Steps':>8} {'Samples':>10} {'FPS':>12} {'ms/upd':>10}")
    print("-"*80)

    # Sort by FPS
    results.sort(key=lambda x: x["fps"], reverse=True)

    for r in results:
        if r["success"]:
            print(f"{r['description']:<20} {r['num_envs']:>8} {r['num_steps']:>8} "
                  f"{r['samples_per_update']:>10,} {r['fps']:>12,.0f} {r['ms_per_update']:>10.1f}")
        else:
            print(f"{r['description']:<20} {r['num_envs']:>8} {r['num_steps']:>8} "
                  f"{r['samples_per_update']:>10,} {'FAILED':>12} {'-':>10}")

    print("-"*80)

    # Find best in each category
    baseline = next((r for r in results if r["description"] == "baseline"), None)
    if baseline and baseline["success"]:
        print(f"\nBaseline (4096×128): {baseline['fps']:,.0f} FPS")
        print("\nRelative to baseline:")
        for r in results:
            if r["success"] and r != baseline:
                diff = (r["fps"] - baseline["fps"]) / baseline["fps"] * 100
                sign = "+" if diff > 0 else ""
                print(f"  {r['description']:<20}: {sign}{diff:.1f}%")


if __name__ == "__main__":
    main()

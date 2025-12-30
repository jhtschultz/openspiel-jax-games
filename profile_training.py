"""JAX Profiler for PPO Training.

Captures a trace of GPU operations to analyze:
- Kernel boundaries (what's actually fused vs separate)
- Memory movement patterns
- Where time is actually spent

Usage:
    python profile_training.py

Output:
    ./profile_logs/ directory containing TensorBoard-compatible traces

View traces:
    tensorboard --logdir=./profile_logs
    # Then open http://localhost:6006 and go to "Profile" tab
    # Or use Perfetto: https://ui.perfetto.dev/ and load the .trace file
"""

import jax
import jax.numpy as jnp
import time
import os

# Import the training components
import gin_rummy_jax as gin
from ppo_gin_rummy_v3_fused import (
    ActorCritic, Transition, env_init, env_step_fused, env_reset_if_done,
    compute_gae, ppo_loss, OBS_DIM, NUM_ACTIONS, DTYPE
)
import optax

# Profile configuration
PROFILE_DIR = "./profile_logs"
NUM_ENVS = 4096
NUM_STEPS = 128
WARMUP_UPDATES = 3  # Let JIT compile before profiling
PROFILE_UPDATES = 2  # Number of updates to profile


def main():
    print("JAX Profiler for PPO Training")
    print(f"  Devices: {jax.devices()}")
    print(f"  Config: num_envs={NUM_ENVS}, num_steps={NUM_STEPS}")
    print(f"  Profile dir: {PROFILE_DIR}")

    os.makedirs(PROFILE_DIR, exist_ok=True)

    # Initialize
    key = jax.random.PRNGKey(42)
    network = ActorCritic()
    key, init_key = jax.random.split(key)
    params = network.init(init_key, jnp.zeros((1, OBS_DIM), dtype=DTYPE))
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4, eps=1e-5))
    opt_state = tx.init(params)
    apply_fn = jax.jit(network.apply)

    # Initialize environments
    print("Initializing environments...")
    key, *keys = jax.random.split(key, NUM_ENVS + 1)
    env_states = jax.vmap(env_init)(jnp.stack(keys))

    # Define training functions (same as in ppo_gin_rummy_v3_fused.py)
    @jax.jit
    def collect_rollout(params, env_states, key):
        def step_fn(carry, _):
            env_states, key = carry
            key, *keys = jax.random.split(key, NUM_ENVS + 1)
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

        (env_states, key), traj = jax.lax.scan(step_fn, (env_states, key), None, NUM_STEPS)
        return env_states, traj, key

    @jax.jit
    def update(params, opt_state, traj, key):
        adv, ret = compute_gae(traj.reward, traj.value, traj.done)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        batch_size = NUM_STEPS * NUM_ENVS
        batch = (
            traj.obs.reshape(batch_size, -1),
            traj.action.reshape(batch_size),
            traj.log_prob.reshape(batch_size),
            adv.reshape(batch_size),
            ret.reshape(batch_size),
            traj.legal_mask.reshape(batch_size, -1),
            traj.value.reshape(batch_size)
        )

        def epoch_fn(carry, _):
            params, opt_state, key = carry
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, batch_size)

            def mb_fn(carry, idx):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (idx,), (batch_size//8,))
                mb = tuple(x[idx] for x in batch)
                (l, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, apply_fn, mb)
                updates, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), (l, aux)

            (params, opt_state), (l, aux) = jax.lax.scan(mb_fn, (params, opt_state), jnp.arange(0, batch_size, batch_size//8))
            return (params, opt_state, key), (l.mean(), {k: v.mean() for k,v in aux.items()})

        (params, opt_state, _), (l, aux) = jax.lax.scan(epoch_fn, (params, opt_state, key), None, 4)
        return params, opt_state, l.mean(), {k: v.mean() for k,v in aux.items()}

    # Warmup (JIT compilation)
    print(f"Warming up ({WARMUP_UPDATES} updates)...")
    for i in range(WARMUP_UPDATES):
        key, r_key, u_key = jax.random.split(key, 3)
        env_states, traj, _ = collect_rollout(params, env_states, r_key)
        params, opt_state, loss, aux = update(params, opt_state, traj, u_key)
        # Block to ensure completion
        jax.block_until_ready(params)
        print(f"  Warmup {i+1}/{WARMUP_UPDATES} done")

    # Profile
    print(f"\nProfiling ({PROFILE_UPDATES} updates)...")
    print("Starting JAX profiler trace...")

    with jax.profiler.trace(PROFILE_DIR, create_perfetto_link=False):
        for i in range(PROFILE_UPDATES):
            key, r_key, u_key = jax.random.split(key, 3)

            # Profile rollout collection
            env_states, traj, _ = collect_rollout(params, env_states, r_key)

            # Profile update
            params, opt_state, loss, aux = update(params, opt_state, traj, u_key)

            # Block to ensure all ops complete within trace
            jax.block_until_ready(params)
            print(f"  Profiled update {i+1}/{PROFILE_UPDATES}")

    print(f"\nProfile saved to: {PROFILE_DIR}/")
    print("\nTo view the trace:")
    print("  Option 1: tensorboard --logdir=./profile_logs")
    print("  Option 2: Open https://ui.perfetto.dev/ and load the .trace.json.gz file")

    # Quick timing comparison
    print("\n--- Quick Timing Breakdown ---")

    # Time rollout only
    key, r_key = jax.random.split(key)
    start = time.perf_counter()
    for _ in range(5):
        env_states, traj, r_key = collect_rollout(params, env_states, r_key)
        jax.block_until_ready(traj.obs)
    rollout_time = (time.perf_counter() - start) / 5

    # Time update only
    key, u_key = jax.random.split(key)
    start = time.perf_counter()
    for _ in range(5):
        params, opt_state, loss, aux = update(params, opt_state, traj, u_key)
        jax.block_until_ready(params)
    update_time = (time.perf_counter() - start) / 5

    total_time = rollout_time + update_time
    steps_per_update = NUM_ENVS * NUM_STEPS
    fps = steps_per_update / total_time

    print(f"  Rollout: {rollout_time*1000:.1f}ms ({rollout_time/total_time*100:.1f}%)")
    print(f"  Update:  {update_time*1000:.1f}ms ({update_time/total_time*100:.1f}%)")
    print(f"  Total:   {total_time*1000:.1f}ms per update")
    print(f"  FPS:     {fps:,.0f}")


if __name__ == "__main__":
    main()

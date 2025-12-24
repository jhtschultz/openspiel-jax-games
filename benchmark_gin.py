"""Benchmark Gin Rummy: JAX vs Original OpenSpiel."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import pyspiel
import gin_rummy_jax


def benchmark_original_sequential(num_games=100):
    """Benchmark original OpenSpiel gin_rummy."""
    game = pyspiel.load_game("gin_rummy")
    rng = np.random.RandomState(42)

    start = time.perf_counter()
    for _ in range(num_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action, _ = outcomes[rng.randint(len(outcomes))]
            else:
                legal = state.legal_actions()
                action = rng.choice(legal)
            state.apply_action(action)
    elapsed = time.perf_counter() - start

    return num_games / elapsed


def benchmark_jax_sequential(num_games=100):
    """Benchmark JAX gin_rummy through OpenSpiel interface."""
    game = pyspiel.load_game("python_gin_rummy_jax")
    rng = np.random.RandomState(42)

    start = time.perf_counter()
    for _ in range(num_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            legal = state.legal_actions()
            action = rng.choice(legal)
            state.apply_action(action)
    elapsed = time.perf_counter() - start

    return num_games / elapsed


def benchmark_jax_batched(batch_size=256, num_batches=10):
    """Benchmark JAX batched simulation."""

    @jax.jit
    def play_one_step(states, key):
        masks = gin_rummy_jax.batched_legal_actions_mask(states)
        # Sample random valid actions
        num_legal = masks.sum(axis=1)
        rand_idx = jax.random.randint(key, (batch_size,), 0, gin_rummy_jax.NUM_ACTIONS)
        cumsum = jnp.cumsum(masks.astype(jnp.int32), axis=1)
        action_idx = rand_idx % jnp.maximum(num_legal.astype(jnp.int32), 1)
        actions = jnp.argmax(cumsum > action_idx[:, None], axis=1)
        new_states = gin_rummy_jax.batched_step(states, actions)
        return new_states

    print(f"  Warming up (batch_size={batch_size})...")
    key = jax.random.PRNGKey(42)
    states = gin_rummy_jax.init_batch(batch_size, key)
    for i in range(100):  # Max moves per game
        key, subkey = jax.random.split(key)
        states = play_one_step(states, subkey)
    jax.block_until_ready(states['done'])

    print(f"  Running benchmark...")
    key = jax.random.PRNGKey(123)
    total_games = 0
    start = time.perf_counter()

    for batch_idx in range(num_batches):
        states = gin_rummy_jax.init_batch(batch_size, jax.random.fold_in(key, batch_idx))
        for step_idx in range(150):  # Max moves
            key, subkey = jax.random.split(key)
            states = play_one_step(states, subkey)
            # Check if all done
            if jnp.all(states['done']):
                break
        jax.block_until_ready(states['done'])
        total_games += batch_size

    elapsed = time.perf_counter() - start
    return total_games / elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("Gin Rummy Benchmark")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 60)

    print("\n1. Original OpenSpiel (C++, with chance nodes):")
    gps = benchmark_original_sequential(num_games=50)
    print(f"   {gps:.1f} games/sec")

    print("\n2. JAX OpenSpiel wrapper (sequential):")
    gps = benchmark_jax_sequential(num_games=20)
    print(f"   {gps:.1f} games/sec")

    print("\n3. JAX batched (batch=256):")
    gps = benchmark_jax_batched(batch_size=256, num_batches=10)
    print(f"   {gps:.1f} games/sec")

    print("\n4. JAX batched (batch=1024):")
    gps = benchmark_jax_batched(batch_size=1024, num_batches=10)
    print(f"   {gps:.1f} games/sec")

    print("\n5. JAX batched (batch=4096):")
    gps = benchmark_jax_batched(batch_size=4096, num_batches=10)
    print(f"   {gps:.1f} games/sec")

    print("\n6. JAX batched (batch=16384):")
    gps = benchmark_jax_batched(batch_size=16384, num_batches=5)
    print(f"   {gps:.1f} games/sec")

    print("\n" + "=" * 60)
    print("Done!")

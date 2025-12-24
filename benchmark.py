"""Benchmark JAX Connect Four vs Original OpenSpiel."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import pyspiel

# Import to register JAX game
import connect_four_jax
from connect_four_jax import (
    init_batch, batched_step, batched_legal_actions_mask,
    init_state, step, legal_actions_mask
)


def benchmark_original_sequential(num_games=1000):
    """Benchmark original OpenSpiel connect_four sequentially."""
    game = pyspiel.load_game("connect_four")
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


def benchmark_jax_openspiel_sequential(num_games=1000):
    """Benchmark JAX game through OpenSpiel interface sequentially."""
    game = pyspiel.load_game("python_connect_four_jax")
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


def benchmark_jax_batched(batch_size=1024, num_batches=100):
    """Benchmark JAX batched simulation on GPU."""

    # JIT compile a full game step that handles the batch
    @jax.jit
    def play_one_step(states, key):
        masks = batched_legal_actions_mask(states)
        # Sample random valid actions
        # For each game, pick a random column that's legal
        num_legal = masks.sum(axis=1)
        rand_idx = jax.random.randint(key, (batch_size,), 0, 7)
        # Use cumsum trick to get nth legal action
        cumsum = jnp.cumsum(masks, axis=1)
        action_idx = rand_idx % jnp.maximum(num_legal, 1)
        actions = jnp.argmax(cumsum > action_idx[:, None], axis=1)
        # Only apply if game not done
        new_states = batched_step(states, actions)
        return new_states

    # Warmup
    print(f"  Warming up JIT compilation...")
    key = jax.random.PRNGKey(42)
    states = init_batch(batch_size)
    for i in range(42):
        key, subkey = jax.random.split(key)
        states = play_one_step(states, subkey)
    # Force completion
    jax.block_until_ready(states['board'])

    # Benchmark
    print(f"  Running benchmark...")
    key = jax.random.PRNGKey(123)
    total_games = 0
    start = time.perf_counter()

    for batch_idx in range(num_batches):
        states = init_batch(batch_size)
        for step_idx in range(42):  # Max game length
            key, subkey = jax.random.split(key)
            states = play_one_step(states, subkey)
        jax.block_until_ready(states['board'])
        total_games += batch_size

    elapsed = time.perf_counter() - start
    return total_games / elapsed


def benchmark_jax_pure_batched(batch_size=4096, num_batches=100):
    """Benchmark pure JAX batched simulation (no OpenSpiel wrapper)."""

    @jax.jit
    def play_full_game(key):
        """Play a full batch of games to completion."""
        states = init_batch(batch_size)

        def step_fn(carry, _):
            states, key = carry
            key, subkey = jax.random.split(key)
            masks = batched_legal_actions_mask(states)
            num_legal = masks.sum(axis=1)
            rand_idx = jax.random.randint(subkey, (batch_size,), 0, 7)
            cumsum = jnp.cumsum(masks, axis=1)
            action_idx = rand_idx % jnp.maximum(num_legal, 1)
            actions = jnp.argmax(cumsum > action_idx[:, None], axis=1)
            new_states = batched_step(states, actions)
            return (new_states, key), None

        (final_states, _), _ = jax.lax.scan(step_fn, (states, key), None, length=42)
        return final_states

    # Warmup
    print(f"  Warming up JIT compilation (scan version)...")
    key = jax.random.PRNGKey(42)
    result = play_full_game(key)
    jax.block_until_ready(result['board'])

    # Benchmark
    print(f"  Running benchmark...")
    total_games = 0
    start = time.perf_counter()

    for batch_idx in range(num_batches):
        key = jax.random.PRNGKey(batch_idx)
        result = play_full_game(key)
        jax.block_until_ready(result['board'])
        total_games += batch_size

    elapsed = time.perf_counter() - start
    return total_games / elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("Connect Four Benchmark")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 60)

    print("\n1. Original OpenSpiel (C++, sequential):")
    gps = benchmark_original_sequential(num_games=2000)
    print(f"   {gps:,.0f} games/sec")

    print("\n2. JAX OpenSpiel wrapper (sequential):")
    gps = benchmark_jax_openspiel_sequential(num_games=500)
    print(f"   {gps:,.0f} games/sec")

    print("\n3. JAX batched (batch=1024, Python loop):")
    gps = benchmark_jax_batched(batch_size=1024, num_batches=50)
    print(f"   {gps:,.0f} games/sec")

    print("\n4. JAX batched (batch=4096, jax.lax.scan):")
    gps = benchmark_jax_pure_batched(batch_size=4096, num_batches=50)
    print(f"   {gps:,.0f} games/sec")

    print("\n5. JAX batched (batch=16384, jax.lax.scan):")
    gps = benchmark_jax_pure_batched(batch_size=16384, num_batches=20)
    print(f"   {gps:,.0f} games/sec")

    print("\n" + "=" * 60)
    print("Done!")

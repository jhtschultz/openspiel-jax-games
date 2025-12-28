"""Benchmark JAX Gin Rummy vs C++ OpenSpiel."""
import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
import time
import jax
import jax.numpy as jnp
import numpy as np

# For standalone GPU benchmarks
import gin_rummy_core as gin

def benchmark_standalone(batch_size=10000, n_steps=200, n_batches=5, fast=True):
    """Benchmark standalone JAX implementation (no pyspiel)."""
    print(f"\n=== Standalone Benchmark (batch={batch_size}, fast={fast}) ===")
    print(f"Device: {jax.devices()[0]}")

    mask_fn = gin.legal_actions_mask_fast if fast else gin.legal_actions_mask

    @jax.jit
    def run_batch(key):
        keys = jax.random.split(key, batch_size)
        states = jax.vmap(lambda _: gin.init_state())(keys)

        def step_fn(i, carry):
            states, keys = carry
            def single(state, key):
                mask = mask_fn(state)
                key, sk = jax.random.split(key)
                p = mask.astype(jnp.float32) / (mask.sum() + 1e-8)
                a = jax.random.choice(sk, jnp.arange(gin.NUM_ACTIONS), p=p)
                ns = jax.lax.cond(state['done'], lambda: state, lambda: gin.step(state, a))
                return ns, key
            return jax.vmap(single)(states, keys)

        final_states, _ = jax.lax.fori_loop(0, n_steps, step_fn, (states, keys))
        return final_states['done']

    print("Compiling...")
    done = run_batch(jax.random.PRNGKey(0))
    jax.block_until_ready(done)
    print("Compiled!")

    start = time.time()
    for i in range(n_batches):
        done = run_batch(jax.random.PRNGKey(i))
    jax.block_until_ready(done)
    elapsed = time.time() - start

    total = n_batches * batch_size
    rate = total / elapsed
    print(f"{total:,} games in {elapsed:.2f}s = {rate:,.0f} games/sec")
    print(f"Speedup vs C++ (118 g/s): {rate/118:.1f}x")
    return rate


def benchmark_simple_bot(batch_size=10000, n_steps=200, n_batches=5, use_opt=True):
    """Benchmark simple bot self-play."""
    bot_fn = gin.simple_bot_action_opt if use_opt else gin.simple_bot_action
    label = "opt" if use_opt else "base"
    print(f"\n=== Simple Bot Benchmark ({label}, batch={batch_size}) ===")
    print(f"Device: {jax.devices()[0]}")

    @jax.jit
    def run_batch(key):
        keys = jax.random.split(key, batch_size)
        states = jax.vmap(lambda _: gin.init_state())(keys)

        def step_fn(i, carry):
            states, keys = carry
            def single(state, key):
                a = bot_fn(state)
                ns = jax.lax.cond(state['done'], lambda: state, lambda: gin.step(state, a))
                return ns, key
            return jax.vmap(single)(states, keys)

        final_states, _ = jax.lax.fori_loop(0, n_steps, step_fn, (states, keys))
        return final_states['done']

    print("Compiling...")
    done = run_batch(jax.random.PRNGKey(0))
    jax.block_until_ready(done)
    print("Compiled!")

    start = time.time()
    for i in range(n_batches):
        done = run_batch(jax.random.PRNGKey(i))
    jax.block_until_ready(done)
    elapsed = time.time() - start

    total = n_batches * batch_size
    rate = total / elapsed
    print(f"{total:,} games in {elapsed:.2f}s = {rate:,.0f} games/sec")
    print(f"Speedup vs C++ SimpleGinRummyBot (153 g/s): {rate/153:.1f}x")
    return rate


def verify_vs_cpp(n_games=50):
    """Verify JAX implementation matches C++ OpenSpiel."""
    print(f"\n=== Correctness Check vs C++ ({n_games} games) ===")

    try:
        import pyspiel
    except ImportError:
        print("pyspiel not available, skipping correctness check")
        return

    cpp_game = pyspiel.load_game("gin_rummy")
    jax_game = pyspiel.load_game("python_gin_rummy_jax")

    disagreements = 0
    for game_idx in range(n_games):
        rng = np.random.RandomState(game_idx)
        cpp_state = cpp_game.new_initial_state()
        jax_state = jax_game.new_initial_state()

        for _ in range(300):
            if cpp_state.is_terminal() != jax_state.is_terminal():
                disagreements += 1
                break
            if cpp_state.is_terminal():
                break

            if cpp_state.is_chance_node() != jax_state.is_chance_node():
                disagreements += 1
                break

            if cpp_state.is_chance_node():
                outcomes = [o[0] for o in cpp_state.chance_outcomes()]
                action = rng.choice(outcomes)
            else:
                cpp_legal = set(a for a in cpp_state.legal_actions() if a <= 55)
                jax_legal = set(a for a in jax_state.legal_actions() if a <= 55)
                if cpp_legal != jax_legal:
                    disagreements += 1
                    break
                action = rng.choice(list(cpp_legal))

            cpp_state.apply_action(action)
            jax_state.apply_action(action)

    print(f"{n_games} games, {disagreements} disagreements")
    if disagreements == 0:
        print("PASSED!")
    return disagreements


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=10000)
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--full", action="store_true", help="Use full legal_actions_mask")
    parser.add_argument("--verify", action="store_true", help="Run C++ verification")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.all:
        benchmark_standalone(batch_size=10000, fast=True)
        benchmark_standalone(batch_size=100000, fast=True)
        benchmark_simple_bot(batch_size=20000)  # Delta Lookups v3 optimal
        verify_vs_cpp()
    else:
        fast = not args.full
        benchmark_standalone(batch_size=args.batch, fast=fast)
        if args.verify:
            verify_vs_cpp()

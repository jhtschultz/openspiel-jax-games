"""Benchmark merged gin_rummy_jax.py with pyspiel wrapper."""
import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
import time
import jax
import jax.numpy as jnp
import numpy as np

# Import to register the game
import gin_rummy_jax as gin

def benchmark_simple_bot_jax(batch_size=10000, n_steps=200, n_batches=5):
    """Benchmark simple bot using merged gin_rummy_jax.py."""
    print(f"\n=== gin_rummy_jax.py Simple Bot Benchmark (batch={batch_size}) ===")
    print(f"Device: {jax.devices()[0]}")

    @jax.jit
    def run_batch(key):
        keys = jax.random.split(key, batch_size)
        states = jax.vmap(lambda _: gin.init_state())(keys)

        def step_fn(i, carry):
            states, keys = carry
            def single(state, key):
                a = gin.simple_bot_action_opt(state)
                ns = jax.lax.cond(state['done'], lambda: state, lambda: gin.step(state, a))
                return ns, key
            return jax.vmap(single)(states, keys)

        final_states, _ = jax.lax.fori_loop(0, n_steps, step_fn, (states, keys))
        return final_states['done'], final_states['p0_score']

    print("Compiling...")
    done, scores = run_batch(jax.random.PRNGKey(0))
    jax.block_until_ready(done)
    print("Compiled!")

    # Check some games completed
    n_done = jnp.sum(done).item()
    print(f"Sample batch: {n_done}/{batch_size} games completed")
    if n_done > 0:
        avg_score = jnp.sum(jnp.where(done, scores, 0)).item() / n_done
        print(f"Average P0 score (completed games): {avg_score:.1f}")

    start = time.time()
    for i in range(n_batches):
        done, _ = run_batch(jax.random.PRNGKey(i + 100))
    jax.block_until_ready(done)
    elapsed = time.time() - start

    total = n_batches * batch_size
    rate = total / elapsed
    print(f"{total:,} games in {elapsed:.2f}s = {rate:,.0f} games/sec")
    print(f"Speedup vs C++ SimpleGinRummyBot (153 g/s): {rate/153:.1f}x")
    return rate


def verify_scoring(n_games=100):
    """Verify games complete correctly with scoring."""
    print(f"\n=== Game Completion Test ({n_games} games) ===")

    completed = 0
    total_p0_score = 0
    total_p1_score = 0
    knocked = 0
    gin_count = 0
    undercut = 0

    key = jax.random.PRNGKey(42)

    for g in range(n_games):
        state = gin.init_state()

        for step in range(300):
            if state['done']:
                completed += 1
                total_p0_score += state['p0_score'].item()
                total_p1_score += state['p1_score'].item()
                if state['knocker'].item() >= 0:
                    knocked += 1
                if state['knocker_deadwood'].item() == 0 and state['knocker'].item() >= 0:
                    gin_count += 1
                break

            action = gin.simple_bot_action_opt(state)
            state = gin.step(state, action.item())

        if step >= 299 and not state['done']:
            print(f"Game {g} did not complete! Phase: {state['phase'].item()}")

    print(f"Completed: {completed}/{n_games} ({100*completed/n_games:.0f}%)")
    if completed > 0:
        print(f"Avg P0 score: {total_p0_score/completed:.1f}")
        print(f"Avg P1 score: {total_p1_score/completed:.1f}")
        print(f"Games with knock: {knocked}")
        print(f"Zero-sum check: P0+P1 = {total_p0_score + total_p1_score} (should be 0)")

    return completed == n_games


def verify_vs_cpp_jax(n_games=50):
    """Verify merged gin_rummy_jax matches C++ OpenSpiel."""
    print(f"\n=== Correctness Check vs C++ ({n_games} games) ===")

    try:
        import pyspiel
    except ImportError:
        print("pyspiel not available, skipping correctness check")
        return

    cpp_game = pyspiel.load_game("gin_rummy")

    disagreements = 0
    for game_idx in range(n_games):
        rng = np.random.RandomState(game_idx)
        cpp_state = cpp_game.new_initial_state()
        jax_state = gin.init_state()

        for _ in range(300):
            cpp_done = cpp_state.is_terminal()
            jax_done = bool(jax_state['done'])

            if cpp_done != jax_done:
                disagreements += 1
                print(f"Game {game_idx}: Terminal mismatch at step {_}")
                break
            if cpp_done:
                # Check scores match
                cpp_returns = cpp_state.returns()
                jax_returns = [jax_state['p0_score'].item(), jax_state['p1_score'].item()]
                if cpp_returns[0] != jax_returns[0] or cpp_returns[1] != jax_returns[1]:
                    print(f"Game {game_idx}: Score mismatch C++={cpp_returns} JAX={jax_returns}")
                    disagreements += 1
                break

            cpp_is_chance = cpp_state.is_chance_node()
            jax_is_chance = gin.is_chance_node(jax_state)

            if cpp_is_chance != jax_is_chance:
                disagreements += 1
                print(f"Game {game_idx}: Chance mismatch at step {_}")
                break

            if cpp_is_chance:
                outcomes = [o[0] for o in cpp_state.chance_outcomes()]
                action = rng.choice(outcomes)
            else:
                # Use simple bot action for deterministic comparison
                action = gin.simple_bot_action_opt(jax_state).item()

                # Verify it's legal in both
                cpp_legal = set(cpp_state.legal_actions())
                if action not in cpp_legal:
                    phase = jax_state['phase'].item()
                    phase_names = ['DEAL', 'FIRST_UPCARD', 'DRAW', 'DISCARD', 'KNOCK', 'LAYOFF', 'WALL', 'GAME_OVER']
                    phase_name = phase_names[phase] if 0 <= phase < len(phase_names) else f"UNKNOWN({phase})"
                    player = jax_state['current_player'].item()
                    hand = gin.get_hand(jax_state, player)
                    hand_cards = [i for i in range(52) if hand[i] > 0]
                    hand_str = ", ".join(gin.card_str(c) for c in hand_cards)
                    print(f"Game {game_idx} Step {_}: Action {action} not legal in C++")
                    print(f"  Phase: {phase_name}, Player: {player}")
                    print(f"  C++ legal: {cpp_legal}")
                    print(f"  Hand ({len(hand_cards)} cards): {hand_str}")
                    print(f"  finished_layoffs: {jax_state['finished_layoffs'].item()}")
                    print(f"  knocker: {jax_state['knocker'].item()}")
                    print(f"  knocker_deadwood: {jax_state['knocker_deadwood'].item()}")
                    # Debug optimal_melds
                    current_dw = gin.calculate_deadwood_lut(hand)
                    optimal_m = gin.optimal_melds_mask(hand)
                    valid_m = gin.valid_melds_mask(hand)
                    n_optimal = int(jnp.sum(optimal_m))
                    n_valid = int(jnp.sum(valid_m))
                    print(f"  current_deadwood: {int(current_dw)}")
                    print(f"  n_valid_melds: {n_valid}, n_optimal_melds: {n_optimal}")
                    if n_valid > 0 and n_optimal == 0:
                        # Check first few valid melds
                        valid_indices = [i for i in range(185) if valid_m[i]][:5]
                        for mi in valid_indices:
                            remaining = jnp.maximum(hand - gin.MELD_MASKS[mi], 0)
                            rem_dw = gin.calculate_deadwood_lut(remaining)
                            print(f"    Meld {mi}: remaining_dw={int(rem_dw)} (should equal {int(current_dw)})")
                    disagreements += 1
                    break

            cpp_state.apply_action(action)
            jax_state = gin.step(jax_state, action)

    print(f"{n_games} games, {disagreements} disagreements")
    if disagreements == 0:
        print("PASSED!")
    return disagreements


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=10000)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        verify_scoring(100)
        benchmark_simple_bot_jax(batch_size=10000)
        benchmark_simple_bot_jax(batch_size=20000)
        verify_vs_cpp_jax()
    else:
        benchmark_simple_bot_jax(batch_size=args.batch)
        if args.verify:
            verify_vs_cpp_jax()

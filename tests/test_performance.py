"""Performance regression tests.

These tests ensure that key operations maintain acceptable performance.
Thresholds are set conservatively to avoid flaky tests on different hardware.

Run with: pytest tests/test_performance.py -v
Skip with: pytest -m "not performance"
Skip GPU tests: pytest -m "not gpu"
"""

import pytest
import time
import jax
import jax.numpy as jnp

import gin_rummy_jax as gin


# Mark all tests in this module as performance tests
pytestmark = pytest.mark.performance


def is_gpu_available():
    """Check if GPU is available for JAX."""
    try:
        return len(jax.devices('gpu')) > 0
    except RuntimeError:
        return False


def benchmark_function(fn, n_iters=1000, warmup=10):
    """Benchmark a function and return calls per second."""
    # Warmup
    for _ in range(warmup):
        result = fn()
    jax.block_until_ready(result)

    # Benchmark
    start = time.time()
    for _ in range(n_iters):
        result = fn()
    jax.block_until_ready(result)
    elapsed = time.time() - start

    return n_iters / elapsed


class TestDeadwoodPerformance:
    """Test deadwood calculation performance."""

    @pytest.fixture
    def sample_hand_11(self):
        """11-card hand for testing."""
        hand = jnp.zeros(52, dtype=jnp.int8)
        cards = [0, 1, 2, 13, 14, 15, 26, 27, 28, 39, 40]
        for c in cards:
            hand = hand.at[c].set(1)
        return hand

    def test_deadwood_lut_speed(self, sample_hand_11):
        """calculate_deadwood_lut should be fast."""
        hand_10 = sample_hand_11.at[40].set(0)  # 10-card hand

        calls_per_sec = benchmark_function(
            lambda: gin.calculate_deadwood_lut(hand_10),
            n_iters=5000
        )

        # Very conservative threshold - should easily exceed on any modern CPU
        assert calls_per_sec > 1000, \
            f"calculate_deadwood_lut too slow: {calls_per_sec:.0f} calls/sec"

    def test_deadwood_compressed_speed(self, sample_hand_11):
        """calculate_deadwood_compressed should be reasonably fast."""
        calls_per_sec = benchmark_function(
            lambda: gin.calculate_deadwood_compressed(sample_hand_11),
            n_iters=2000
        )

        assert calls_per_sec > 500, \
            f"calculate_deadwood_compressed too slow: {calls_per_sec:.0f} calls/sec"


class TestGameLoopPerformance:
    """Test game loop performance."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not is_gpu_available(), reason="GPU not available")
    def test_step_function_speed_gpu(self):
        """step() function should be fast on GPU."""
        state = gin.init_state()
        key = jax.random.PRNGKey(0)

        # Deal cards
        for _ in range(21):
            if state['phase'] != gin.PHASE_DEAL:
                break
            deck = state['deck'].astype(jnp.float32)
            deck_sum = deck.sum()
            if deck_sum > 0:
                key, subkey = jax.random.split(key)
                action = jax.random.choice(subkey, 52, p=deck / deck_sum)
                state = gin.step(state, action)

        if state['phase'] == gin.PHASE_FIRST_UPCARD:
            action = gin.ACTION_PASS
            _ = gin.step(state, action)
            jax.block_until_ready(_)

            calls_per_sec = benchmark_function(
                lambda: gin.step(state, action),
                n_iters=5000
            )

            assert calls_per_sec > 5000, \
                f"step() too slow: {calls_per_sec:.0f} calls/sec"

    def test_init_state_speed(self):
        """init_state() should be reasonably fast."""
        calls_per_sec = benchmark_function(
            gin.init_state,
            n_iters=5000
        )

        # Conservative threshold that works on CPU
        assert calls_per_sec > 1000, \
            f"init_state() too slow: {calls_per_sec:.0f} calls/sec"


class TestBotPerformance:
    """Test bot decision performance."""

    @pytest.fixture
    def ready_state(self):
        """Create a state ready for bot decision."""
        state = gin.init_state()
        key = jax.random.PRNGKey(42)

        # Deal cards
        for _ in range(100):
            if state['done']:
                break
            if state['phase'] not in [gin.PHASE_DEAL]:
                if not state['waiting_stock_draw']:
                    break

            if state['phase'] == gin.PHASE_DEAL or state['waiting_stock_draw']:
                deck = state['deck'].astype(jnp.float32)
                deck_sum = deck.sum()
                if deck_sum > 0:
                    key, subkey = jax.random.split(key)
                    action = jax.random.choice(subkey, 52, p=deck / deck_sum)
                else:
                    break
            else:
                break

            state = gin.step(state, action)

        return state

    def test_simple_bot_speed(self, ready_state):
        """simple_bot_action_opt should be reasonably fast."""
        if ready_state['done']:
            pytest.skip("Game already ended")

        # Warmup and JIT compile
        _ = gin.simple_bot_action_opt(ready_state)
        jax.block_until_ready(_)

        calls_per_sec = benchmark_function(
            lambda: gin.simple_bot_action_opt(ready_state),
            n_iters=1000
        )

        # Conservative threshold for single-state bot calls
        assert calls_per_sec > 500, \
            f"simple_bot_action_opt too slow: {calls_per_sec:.0f} calls/sec"


class TestBatchedPerformance:
    """Test batched operations performance (GPU-oriented)."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not is_gpu_available(), reason="GPU not available")
    def test_full_game_throughput_gpu(self):
        """Run full games and measure throughput (GPU only)."""
        n_games = 100
        key = jax.random.PRNGKey(0)
        completed_games = 0
        total_steps = 0

        start = time.time()

        for game_idx in range(n_games):
            state = gin.init_state()
            key, game_key = jax.random.split(key)

            for step in range(300):
                if state['done']:
                    completed_games += 1
                    total_steps += step
                    break

                is_chance = (state['phase'] == gin.PHASE_DEAL) or state['waiting_stock_draw']
                if is_chance:
                    deck = state['deck'].astype(jnp.float32)
                    deck_sum = deck.sum()
                    if deck_sum > 0:
                        game_key, subkey = jax.random.split(game_key)
                        action = jax.random.choice(subkey, 52, p=deck / deck_sum)
                    else:
                        break
                else:
                    action = gin.simple_bot_action_opt(state)

                state = gin.step(state, action)

        elapsed = time.time() - start
        games_per_sec = completed_games / elapsed if elapsed > 0 else 0

        # GPU should achieve high throughput
        assert games_per_sec > 5, \
            f"Game throughput too low: {games_per_sec:.1f} games/sec"
        assert completed_games >= n_games * 0.9, \
            f"Too many incomplete games: {completed_games}/{n_games}"

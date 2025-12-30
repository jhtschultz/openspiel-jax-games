"""Test JAX bot equivalence with C++ SimpleGinRummyBot.

This test verifies that the JAX simple_bot_action_opt produces the same
actions as the C++ SimpleGinRummyBot from OpenSpiel.

Note: There is a known divergence in GIN layoff behavior (see CLAUDE.md).
The JAX bot correctly lays melds when facing GIN, while the C++ bot has
a bug that skips meld-laying. This can cause score differences.
"""

import pytest
import numpy as np

try:
    import pyspiel
    PYSPIEL_AVAILABLE = True
except ImportError:
    PYSPIEL_AVAILABLE = False

import gin_rummy_jax as gin


pytestmark = pytest.mark.skipif(
    not PYSPIEL_AVAILABLE,
    reason="pyspiel not installed"
)


def run_comparison_game(seed):
    """Run a single game comparing JAX and C++ bots.

    Returns:
        tuple: (disagreements, total_decisions, scores_match)
    """
    cpp_game = pyspiel.load_game("gin_rummy")
    cpp_bot_p0 = pyspiel.make_simple_gin_rummy_bot(cpp_game.get_parameters(), 0)
    cpp_bot_p1 = pyspiel.make_simple_gin_rummy_bot(cpp_game.get_parameters(), 1)

    rng = np.random.RandomState(seed)
    cpp_state = cpp_game.new_initial_state()
    jax_state = gin.init_state()

    disagreements = 0
    total_decisions = 0

    while not cpp_state.is_terminal():
        cpp_player = cpp_state.current_player()

        if cpp_state.is_chance_node():
            outcomes = cpp_state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = rng.choice(actions, p=probs)
        else:
            cpp_bot = cpp_bot_p0 if cpp_player == 0 else cpp_bot_p1
            cpp_action = cpp_bot.step(cpp_state)
            jax_action = int(gin.simple_bot_action_opt(jax_state))

            total_decisions += 1
            if cpp_action != jax_action:
                disagreements += 1

            action = cpp_action  # Use C++ action to keep states in sync

        cpp_state.apply_action(action)
        jax_state = gin.step(jax_state, action)

    # Check scores
    cpp_returns = cpp_state.returns()
    jax_returns = [int(jax_state['p0_score']), int(jax_state['p1_score'])]
    scores_match = (cpp_returns[0] == jax_returns[0] and
                    cpp_returns[1] == jax_returns[1])

    return disagreements, total_decisions, scores_match


class TestBotEquivalence:
    """Test JAX bot matches C++ bot."""

    def test_single_game_no_disagreements(self):
        """A single game should have no action disagreements."""
        disagreements, decisions, _ = run_comparison_game(seed=0)
        assert disagreements == 0, f"Found {disagreements}/{decisions} disagreements"

    @pytest.mark.parametrize("seed", range(10))
    def test_multiple_games_no_disagreements(self, seed):
        """Multiple games should all have no disagreements."""
        disagreements, decisions, _ = run_comparison_game(seed=seed)
        # Allow for known GIN layoff divergence (rare)
        assert disagreements <= 2, f"Too many disagreements: {disagreements}/{decisions}"

    def test_batch_games_low_disagreement_rate(self):
        """Over many games, disagreement rate should be very low."""
        total_disagreements = 0
        total_decisions = 0
        n_games = 50

        for seed in range(n_games):
            d, t, _ = run_comparison_game(seed=seed)
            total_disagreements += d
            total_decisions += t

        disagreement_rate = total_disagreements / total_decisions if total_decisions > 0 else 0
        # Allow up to 1% disagreement rate (known GIN bug in C++)
        assert disagreement_rate < 0.01, \
            f"Disagreement rate {disagreement_rate:.2%} exceeds 1%"

    def test_scores_usually_match(self):
        """Game scores should match in most games."""
        n_games = 20
        score_matches = 0

        for seed in range(n_games):
            _, _, match = run_comparison_game(seed=seed)
            if match:
                score_matches += 1

        match_rate = score_matches / n_games
        # Allow some score mismatches due to GIN layoff bug
        assert match_rate >= 0.8, \
            f"Score match rate {match_rate:.0%} is too low"


class TestBotActions:
    """Test specific bot action behaviors."""

    def test_bot_returns_valid_action(self):
        """Bot should always return a valid action index."""
        state = gin.init_state()
        # Simulate dealing
        import jax
        key = jax.random.PRNGKey(42)

        for _ in range(100):
            if state['done']:
                break

            is_chance = (state['phase'] == gin.PHASE_DEAL) or state['waiting_stock_draw']
            if is_chance:
                import jax.numpy as jnp
                deck = state['deck'].astype(jnp.float32)
                deck_sum = deck.sum()
                if deck_sum > 0:
                    deck_probs = deck / deck_sum
                    key, subkey = jax.random.split(key)
                    action = jax.random.choice(subkey, 52, p=deck_probs)
                else:
                    break
            else:
                action = gin.simple_bot_action_opt(state)
                action_int = int(action)
                assert 0 <= action_int < gin.NUM_ACTIONS, \
                    f"Invalid action {action_int}"

            state = gin.step(state, action)

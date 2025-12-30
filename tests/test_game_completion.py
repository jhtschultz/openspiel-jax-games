"""Test that games complete properly with the simple bot."""

import pytest
import jax
import jax.numpy as jnp

import gin_rummy_core as gin_core
import gin_rummy_jax as gin_jax


def run_game_to_completion(gin_module, seed=0, max_steps=300):
    """Run a game using the simple bot until completion.

    Returns:
        tuple: (final_state, num_steps, completed)
    """
    key = jax.random.PRNGKey(seed)
    state = gin_module.init_state()

    for i in range(max_steps):
        if state['done']:
            return state, i, True

        # Chance or player action
        is_chance = (state['phase'] == gin_module.PHASE_DEAL) or state['waiting_stock_draw']
        if is_chance:
            deck = state['deck'].astype(jnp.float32)
            deck_probs = deck / jnp.maximum(deck.sum(), 1.0)
            key, subkey = jax.random.split(key)
            action = jax.random.choice(subkey, 52, p=deck_probs)
        else:
            action = gin_module.simple_bot_action_opt(state)

        state = gin_module.step(state, action)

    return state, max_steps, False


class TestGameCompletionCore:
    """Test game completion with gin_rummy_core."""

    def test_single_game_completes(self):
        """A single game should complete within 300 steps."""
        state, steps, completed = run_game_to_completion(gin_core, seed=0)
        assert completed, f"Game did not complete in {steps} steps"
        assert state['done']
        assert state['winner'] in [0, 1]

    @pytest.mark.parametrize("seed", range(10))
    def test_multiple_games_complete(self, seed):
        """Multiple games with different seeds should all complete."""
        state, steps, completed = run_game_to_completion(gin_core, seed=seed)
        assert completed, f"Game {seed} did not complete"

    def test_winner_is_valid(self):
        """Winner should be player 0 or 1."""
        state, _, completed = run_game_to_completion(gin_core, seed=42)
        assert completed
        assert int(state['winner']) in [0, 1]

    def test_knocker_deadwood_recorded(self):
        """Knocker deadwood should be recorded when game ends by knock."""
        state, _, completed = run_game_to_completion(gin_core, seed=0)
        assert completed
        # If there was a knocker, deadwood should be recorded
        if int(state['knocker']) >= 0:
            assert int(state['knocker_deadwood']) >= 0


class TestGameCompletionJax:
    """Test game completion with gin_rummy_jax."""

    def test_single_game_completes(self):
        """A single game should complete within 300 steps."""
        state, steps, completed = run_game_to_completion(gin_jax, seed=0)
        assert completed, f"Game did not complete in {steps} steps"
        assert state['done']

    @pytest.mark.parametrize("seed", range(10))
    def test_multiple_games_complete(self, seed):
        """Multiple games with different seeds should all complete."""
        state, steps, completed = run_game_to_completion(gin_jax, seed=seed)
        assert completed, f"Game {seed} did not complete"

    def test_scores_recorded(self):
        """Final scores should be recorded."""
        state, _, completed = run_game_to_completion(gin_jax, seed=0)
        assert completed
        # At least one player should have a non-zero score (or it's a tie at 0)
        p0_score = int(state['p0_score'])
        p1_score = int(state['p1_score'])
        # Scores should be reasonable (not corrupted)
        assert -200 <= p0_score <= 200
        assert -200 <= p1_score <= 200

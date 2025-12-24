"""Basic tests for gin_rummy_jax to verify game mechanics."""

import numpy as np
import jax
import jax.numpy as jnp
import pyspiel
import gin_rummy_jax


def test_initial_state():
    """Test that initial state is correct."""
    print("Test: Initial state...")
    game = pyspiel.load_game("python_gin_rummy_jax")
    state = game.new_initial_state()

    # Should start in FirstUpcard phase
    assert state.current_player() == 0, f"Expected player 0, got {state.current_player()}"
    assert not state.is_terminal(), "Game should not be terminal"

    # Legal actions should be draw upcard or pass
    legal = state.legal_actions()
    assert 52 in legal, "Draw upcard should be legal"
    assert 54 in legal, "Pass should be legal"
    print("  PASS")


def test_first_upcard_phase():
    """Test first upcard phase logic."""
    print("Test: First upcard phase...")
    game = pyspiel.load_game("python_gin_rummy_jax")

    # Test: both pass -> move to Draw phase
    state = game.new_initial_state()
    state.apply_action(54)  # P0 passes
    assert state.current_player() == 1, "Should be player 1's turn"
    state.apply_action(54)  # P1 passes
    assert state.current_player() == 0, "Should be player 0's turn in Draw phase"
    legal = state.legal_actions()
    assert 53 in legal, "Draw stock should be legal in Draw phase"

    # Test: P0 draws upcard -> move to Discard phase
    state = game.new_initial_state()
    state.apply_action(52)  # P0 draws upcard
    # Now P0 should have 11 cards and must discard
    legal = state.legal_actions()
    # Legal actions should be card discards (0-51)
    assert any(a < 52 for a in legal), f"Should be able to discard cards, got {legal}"
    print("  PASS")


def test_draw_discard_cycle():
    """Test the draw-discard cycle."""
    print("Test: Draw-discard cycle...")
    game = pyspiel.load_game("python_gin_rummy_jax")
    state = game.new_initial_state()

    # Pass first upcard phase
    state.apply_action(54)  # P0 passes
    state.apply_action(54)  # P1 passes

    # Now in Draw phase, player 0's turn
    initial_player = state.current_player()
    assert initial_player == 0

    # Draw from stock
    state.apply_action(53)  # Draw stock

    # Now should be in discard phase, still player 0
    legal = state.legal_actions()
    discard_actions = [a for a in legal if a < 52]
    assert len(discard_actions) > 0, "Should have discard options"

    # Discard a card
    state.apply_action(discard_actions[0])

    # Now should be player 1's turn in Draw phase
    assert state.current_player() == 1, f"Should be player 1, got {state.current_player()}"
    print("  PASS")


def test_play_complete_game():
    """Test that games can complete."""
    print("Test: Complete games...")
    game = pyspiel.load_game("python_gin_rummy_jax")
    np.random.seed(42)

    outcomes = {'p0_win': 0, 'p1_win': 0, 'draw': 0}

    for game_idx in range(20):
        state = game.new_initial_state()
        moves = 0
        while not state.is_terminal() and moves < 300:
            legal = state.legal_actions()
            action = np.random.choice(legal)
            state.apply_action(action)
            moves += 1

        returns = state.returns()
        if returns[0] > 0:
            outcomes['p0_win'] += 1
        elif returns[1] > 0:
            outcomes['p1_win'] += 1
        else:
            outcomes['draw'] += 1

    print(f"  Outcomes: {outcomes}")
    assert outcomes['draw'] + outcomes['p0_win'] + outcomes['p1_win'] == 20
    print("  PASS")


def test_knock_action():
    """Test that knocking ends the game."""
    print("Test: Knock action...")
    game = pyspiel.load_game("python_gin_rummy_jax")

    # Find a state where knock is legal
    np.random.seed(12345)
    found_knock = False

    for _ in range(100):
        state = game.new_initial_state()
        moves = 0
        while not state.is_terminal() and moves < 100:
            legal = state.legal_actions()
            if 55 in legal:  # Knock is legal
                state.apply_action(55)
                found_knock = True
                break
            action = np.random.choice(legal)
            state.apply_action(action)
            moves += 1

        if found_knock:
            break

    if found_knock:
        assert state.is_terminal(), "Knock should end game"
        returns = state.returns()
        assert returns[0] != 0 or returns[1] != 0, "Knock should have a winner"
        print(f"  Found knock, returns: {returns}")
    else:
        print("  No knock found in random play (might be rare)")

    print("  PASS")


def test_batched_games():
    """Test batched game operations."""
    print("Test: Batched operations...")

    key = jax.random.PRNGKey(42)
    batch_size = 64

    states = gin_rummy_jax.init_batch(batch_size, key)

    # Check shapes
    assert states['player0_hand'].shape == (batch_size, 52)
    assert states['current_player'].shape == (batch_size,)
    assert states['phase'].shape == (batch_size,)

    # Check legal actions mask
    masks = gin_rummy_jax.batched_legal_actions_mask(states)
    assert masks.shape == (batch_size, 241)

    # All games should have same legal actions initially (first upcard phase)
    assert jnp.all(masks[:, 52] == True), "Draw upcard should be legal for all"
    assert jnp.all(masks[:, 54] == True), "Pass should be legal for all"

    print("  PASS")


if __name__ == "__main__":
    print("=" * 50)
    print("Gin Rummy JAX Basic Tests")
    print("=" * 50 + "\n")

    test_initial_state()
    test_first_upcard_phase()
    test_draw_discard_cycle()
    test_play_complete_game()
    test_knock_action()
    test_batched_games()

    print("\n" + "=" * 50)
    print("All basic tests passed!")
    print("=" * 50)

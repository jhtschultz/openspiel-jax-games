"""Pytest configuration and shared fixtures."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import jax
import jax.numpy as jnp


@pytest.fixture
def rng_key():
    """Provide a reproducible JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_hand():
    """Create a sample 10-card hand for testing.

    Cards: As, 2s, 3s (spades run), 4h, 5h, 6h (hearts run), 10c, Jc, Qc, Kd
    Expected: 2 runs of 3, deadwood = 10 (Kd)
    """
    hand = jnp.zeros(52, dtype=jnp.int8)
    # Spades run: A(0), 2(1), 3(2)
    # Hearts run: 4(42), 5(43), 6(44) - wait, that's wrong
    # Card encoding: card = rank + suit*13
    # Suits: spades=0, clubs=1, diamonds=2, hearts=3
    # So: As=0, 2s=1, 3s=2, 4h=3+39=42... wait let me recalculate
    # Actually: rank 0-12, suit 0-3, card = suit*13 + rank
    # spades: 0-12, clubs: 13-25, diamonds: 26-38, hearts: 39-51
    # As=0, 2s=1, 3s=2
    # 4h=39+3=42, 5h=43, 6h=44
    # 10c=13+9=22, Jc=23, Qc=24
    # Kd=26+12=38
    cards = [0, 1, 2, 42, 43, 44, 22, 23, 24, 38]
    for c in cards:
        hand = hand.at[c].set(1)
    return hand


@pytest.fixture
def sample_hand_11(sample_hand):
    """Create an 11-card hand (after drawing, before discard)."""
    # Add Ac (clubs ace) = 13
    return sample_hand.at[13].set(1)

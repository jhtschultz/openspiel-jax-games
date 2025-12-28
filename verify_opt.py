"""Verify optimized deadwood matches exact version."""
import jax
import jax.numpy as jnp
import numpy as np
import gin_rummy_core as gin

def verify_deadwood_compressed(n_hands=10000):
    """Compare calculate_deadwood_compressed vs calculate_deadwood_all_discards_exact."""
    print(f"Verifying {n_hands:,} random hands...")
    
    disagreements = 0
    for i in range(n_hands):
        # Generate random 11-card hand
        rng = np.random.RandomState(i)
        cards = rng.choice(52, 11, replace=False)
        hand = np.zeros(52, dtype=np.int8)
        hand[cards] = 1
        hand = jnp.array(hand)
        
        # Exact version (known correct)
        exact_dw = gin.calculate_deadwood_all_discards_exact(hand)
        
        # Compressed version (optimized)
        held_indices, compressed_dw = gin.calculate_deadwood_compressed(hand)
        
        # Compare: for each held card, deadwood should match
        for j in range(11):
            card_idx = int(held_indices[j])
            if hand[card_idx] > 0:  # Valid held card
                exact_val = int(exact_dw[card_idx])
                compressed_val = int(compressed_dw[j])
                if exact_val != compressed_val:
                    disagreements += 1
                    print(f"Hand {i}, card {card_idx}: exact={exact_val}, compressed={compressed_val}")
                    if disagreements >= 10:
                        print("Too many disagreements, stopping...")
                        return disagreements
    
    print(f"Result: {n_hands:,} hands, {disagreements} disagreements")
    if disagreements == 0:
        print("PASSED!")
    return disagreements

def verify_meld_check(n_tests=10000):
    """Verify card_would_be_in_meld_fast matches card_would_be_in_meld."""
    print(f"\nVerifying meld check on {n_tests:,} random (hand, card) pairs...")
    
    disagreements = 0
    for i in range(n_tests):
        rng = np.random.RandomState(i)
        # Generate random 10-card hand
        cards = rng.choice(52, 10, replace=False)
        hand = np.zeros(52, dtype=np.int8)
        hand[cards] = 1
        hand = jnp.array(hand)
        
        # Pick a card not in hand
        available = [c for c in range(52) if c not in cards]
        card = rng.choice(available)
        
        # Compare methods
        slow = bool(gin.card_would_be_in_meld(hand, card))
        fast = bool(gin.card_would_be_in_meld_fast(hand, card))
        
        if slow != fast:
            disagreements += 1
            print(f"Test {i}, card {card}: slow={slow}, fast={fast}")
            if disagreements >= 10:
                print("Too many disagreements, stopping...")
                return disagreements
    
    print(f"Result: {n_tests:,} tests, {disagreements} disagreements")
    if disagreements == 0:
        print("PASSED!")
    return disagreements

if __name__ == "__main__":
    verify_deadwood_compressed(10000)
    verify_meld_check(10000)

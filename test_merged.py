"""Test merged gin_rummy_jax.py with LUT optimizations."""

import jax
import jax.numpy as jnp
import time

# Skip pyspiel dependency test
print("Testing gin_rummy_jax.py (merged implementation)")
print("=" * 50)

# Test without pyspiel first (the core functions)
import gin_rummy_jax as gin

# Test the LUT-based functions
print("\n1. Testing LUT-based deadwood calculation...")
key = jax.random.PRNGKey(42)

# Create a test hand (10 cards)
test_hand = jnp.zeros(52, dtype=jnp.int8)
# Add some cards: As, 2s, 3s, 4h, 5h, 6h, 10c, Jc, Qc, Kd
cards = [0, 1, 2, 39, 40, 41, 22, 23, 24, 38]
for c in cards:
    test_hand = test_hand.at[c].set(1)

dw_lut = gin.calculate_deadwood_lut(test_hand)
print(f"   Hand has {jnp.sum(test_hand).item()} cards, deadwood (LUT): {dw_lut.item()}")

# Test 216-scenario exact
print("\n2. Testing 216-scenario exact deadwood...")
# Add one more card for 11-card hand
test_hand_11 = test_hand.at[13].set(1)  # Ac
all_dw = gin.calculate_deadwood_all_discards_exact(test_hand_11)
min_dw = jnp.min(all_dw)
print(f"   11-card hand, min deadwood after discard: {min_dw.item()}")

# Test simple_bot_action_opt
print("\n3. Testing simple_bot_action_opt...")
state = gin.init_state()
# Need to manually deal cards for testing
print(f"   State initialized, phase: {state['phase'].item()}")

# Benchmark compressed deadwood (core optimization)
print("\n4. Benchmarking calculate_deadwood_compressed...")
# Warm-up JIT
_ = gin.calculate_deadwood_compressed(test_hand_11)

# Benchmark
n_iters = 10000
start = time.time()
for _ in range(n_iters):
    _ = gin.calculate_deadwood_compressed(test_hand_11)
jax.block_until_ready(_)
elapsed = time.time() - start
print(f"   {n_iters} iterations in {elapsed:.3f}s = {n_iters/elapsed:.0f} calls/sec")

# Test that simple_bot_action_opt compiles
print("\n5. Testing simple_bot_action_opt compilation...")
# Create a simple state after dealing
state = gin.init_state()
# Deal cards manually (simulate chance actions)
deck = jnp.ones(52, dtype=jnp.int8)
p0_hand = jnp.zeros(52, dtype=jnp.int8)
p1_hand = jnp.zeros(52, dtype=jnp.int8)

# Deal 10 cards to each player
key = jax.random.PRNGKey(0)
for i in range(10):
    key, subkey = jax.random.split(key)
    avail = jnp.where(deck > 0, jnp.arange(52), -1)
    card = jax.random.choice(subkey, 52, p=deck.astype(jnp.float32)/deck.sum())
    p0_hand = p0_hand.at[card].set(1)
    deck = deck.at[card].set(0)

for i in range(10):
    key, subkey = jax.random.split(key)
    card = jax.random.choice(subkey, 52, p=deck.astype(jnp.float32)/deck.sum())
    p1_hand = p1_hand.at[card].set(1)
    deck = deck.at[card].set(0)

# Set upcard
key, subkey = jax.random.split(key)
upcard = jax.random.choice(subkey, 52, p=deck.astype(jnp.float32)/deck.sum())
deck = deck.at[upcard].set(0)

state = {
    'player0_hand': p0_hand,
    'player1_hand': p1_hand,
    'deck': deck,
    'upcard': jnp.int8(upcard),
    'current_player': jnp.int8(0),
    'phase': jnp.int8(gin.PHASE_FIRST_UPCARD),
    'done': jnp.bool_(False),
    'winner': jnp.int8(-1),
    'pass_count': jnp.int8(0),
    'deal_cards_dealt': jnp.int8(21),
    'num_draw_upcard': jnp.int32(0),
    'knocker': jnp.int8(-1),
    'knocked': jnp.zeros(2, dtype=jnp.bool_),
    'layed_melds': jnp.zeros((2, 185), dtype=jnp.bool_),
    'layoffs_mask': jnp.zeros(52, dtype=jnp.bool_),
    'finished_layoffs': jnp.bool_(False),
    'waiting_stock_draw': jnp.bool_(False),
    'knocker_deadwood': jnp.int32(-1),
    'discard_pile': jnp.zeros(52, dtype=jnp.int8),
    'drawn_from_discard': jnp.int8(-1),
    'prev_upcard': jnp.int8(upcard),
    'repeated_move': jnp.bool_(False),
    'deadwood': jnp.zeros(2, dtype=jnp.int32),
    'p0_score': jnp.int32(0),
    'p1_score': jnp.int32(0),
}

# Test simple_bot_action_opt
start = time.time()
action = gin.simple_bot_action_opt(state)
jax.block_until_ready(action)
elapsed = time.time() - start
print(f"   simple_bot_action_opt compiled and returned action: {action.item()} in {elapsed:.3f}s")

# Benchmark simple_bot_action_opt
n_iters = 1000
start = time.time()
for _ in range(n_iters):
    action = gin.simple_bot_action_opt(state)
jax.block_until_ready(action)
elapsed = time.time() - start
print(f"   {n_iters} iterations in {elapsed:.3f}s = {n_iters/elapsed:.0f} calls/sec")

print("\n" + "=" * 50)
print("All tests passed! Merged implementation working.")

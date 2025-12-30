"""Evaluate PPO checkpoint against simple bot and save game histories.

Reuses the training loop from ppo_gin_rummy_v3_fused.py to ensure identical
game dynamics and observation construction.
"""

import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import jax
import jax.numpy as jnp
import json
import argparse
from datetime import datetime

import orbax.checkpoint as ocp
import gin_rummy_jax as gin

# Import from training script - reuse as much as possible
from ppo_gin_rummy_v3_fused import (
    ActorCritic, OBS_DIM, DTYPE, NUM_ACTIONS, CHECKPOINT_DIR,
    make_observation, get_legal_mask, update_known_cards,
    is_chance_node, is_agent_strategic_turn
)


def load_checkpoint(checkpoint_dir):
    """Load the latest checkpoint."""
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer)

    latest_step = ckpt_manager.latest_step()
    if latest_step is None:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading checkpoint from update {latest_step}")
    ckpt = ckpt_manager.restore(latest_step)
    return ckpt['params'], latest_step


def card_to_str(card_idx):
    """Convert card index to string like 'As', '2c', 'Td'."""
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['s', 'c', 'd', 'h']  # spades=0, clubs=1, diamonds=2, hearts=3
    return ranks[card_idx % 13] + suits[card_idx // 13]


def action_to_str(action):
    """Convert action index to readable string."""
    if action < 52:
        return f"discard_{card_to_str(action)}"
    elif action == 52:
        return "draw_upcard"
    elif action == 53:
        return "draw_stock"
    elif action == 54:
        return "pass"
    elif action == 55:
        return "knock"
    elif 56 <= action < 241:
        return f"meld_{action - 56}"
    return f"action_{action}"


def auto_step(game_state, known_cards, agent_player, key):
    """Execute one auto-play step (chance or bot). Same logic as training's single_auto_step."""
    key, subkey = jax.random.split(key)

    is_chance = bool(is_chance_node(game_state))

    if is_chance:
        # Chance action - sample from deck
        deck = game_state['deck'].astype(jnp.float32)
        deck_probs = deck / jnp.maximum(deck.sum(), 1.0)
        action = int(jax.random.choice(subkey, 52, p=deck_probs))
    else:
        # Bot action
        action = int(gin.simple_bot_action_opt(game_state))

        # Update known cards for opponent actions
        current_player = int(game_state['current_player'])
        if current_player != agent_player:
            known_cards = update_known_cards(game_state, action, known_cards)

    game_state = gin.step(game_state, action)
    return game_state, known_cards, key, action


def run_game(params, apply_fn, seed):
    """Run a single game with full action history.

    Uses the same core logic as training (is_agent_strategic_turn, update_known_cards, etc.)
    but steps manually to capture all actions including chance and opponent.
    """
    key = jax.random.PRNGKey(seed)
    key, player_key, init_key = jax.random.split(key, 3)

    # Random agent player (same as training's env_init)
    agent_player = int(jax.random.randint(player_key, (), 0, 2))

    game_state = gin.init_state()
    known_cards = jnp.zeros((2, 52), dtype=jnp.bool_)
    key = init_key

    game = {
        'seed': seed,
        'agent_player': agent_player,
        'history': [],
    }

    # Fast-forward through deal + opponent turns until agent's strategic turn
    # (Same as training's env_init scan loop)
    for _ in range(50):  # Max iterations for fast-forward
        if game_state['done'] or bool(is_agent_strategic_turn(game_state, agent_player)):
            break
        game_state, known_cards, key, action = auto_step(game_state, known_cards, agent_player, key)
        game['history'].append(action)

    # Main game loop - agent makes decisions at strategic turns
    for step in range(1000):  # Max steps safety
        if game_state['done']:
            break

        # At this point, it should be agent's strategic turn
        # Agent's strategic turn - use PPO policy (greedy)
        obs = make_observation(game_state, known_cards, agent_player)
        logits, _ = apply_fn(params, obs[None, :])
        logits = logits[0]

        legal_mask = get_legal_mask(game_state, agent_player)
        logits = jnp.where(legal_mask, logits, -1e9)
        action = int(jnp.argmax(logits))

        # Update known cards for agent action
        known_cards = update_known_cards(game_state, action, known_cards)

        game['history'].append(action)
        game_state = gin.step(game_state, action)

        # Auto-play until next agent strategic turn or game end
        for _ in range(20):  # Max auto-play steps
            if game_state['done'] or bool(is_agent_strategic_turn(game_state, agent_player)):
                break
            game_state, known_cards, key, auto_action = auto_step(game_state, known_cards, agent_player, key)
            game['history'].append(auto_action)

    game['p0_score'] = int(game_state['p0_score'])
    game['p1_score'] = int(game_state['p1_score'])
    game['agent_won'] = bool((game_state['p0_score'] > 0) == (agent_player == 0))

    return game


def run_evaluation(checkpoint_dir, n_games=100, output_file=None):
    """Run evaluation and save results as JSONL (one game per line)."""
    print(f"Loading checkpoint from {checkpoint_dir}")
    params, update_step = load_checkpoint(checkpoint_dir)

    network = ActorCritic()
    apply_fn = jax.jit(network.apply)

    # Warm up JIT
    print("Warming up JIT...")
    dummy_obs = jnp.zeros((1, OBS_DIM), dtype=DTYPE)
    _ = apply_fn(params, dummy_obs)

    print(f"Running {n_games} games...")

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"games_{timestamp}.jsonl"

    agent_wins = 0

    with open(output_file, 'w') as f:
        for i in range(n_games):
            game = run_game(params, apply_fn, seed=i)

            # Write one game per line
            f.write(json.dumps(game) + '\n')

            if game['agent_won']:
                agent_wins += 1

            if (i + 1) % 10 == 0:
                win_rate = agent_wins / (i + 1)
                print(f"  Games {i+1}/{n_games}: Agent win rate = {win_rate:.1%}")

    win_rate = agent_wins / n_games
    print(f"\nFinal: Agent won {agent_wins}/{n_games} ({win_rate:.1%})")
    print(f"Results saved to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint against simple bot")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help="Checkpoint directory")
    parser.add_argument("--n-games", type=int, default=100,
                        help="Number of games to play")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: eval_results_TIMESTAMP.json)")
    args = parser.parse_args()

    run_evaluation(args.checkpoint_dir, args.n_games, args.output)

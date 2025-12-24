"""JAX-accelerated Connect Four for OpenSpiel.

This implements Connect Four with a JAX core for fast batched simulation,
wrapped in an OpenSpiel-compatible Python game interface.

The game matches the original connect_four exactly:
- Actions: 0-6 (column to drop piece)
- Observation: [3, 6, 7] tensor (empty, player0, player1 planes)
- Board: 6 rows x 7 cols, row 0 = top, row 5 = bottom
"""

from typing import List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

import pyspiel

# =============================================================================
# Constants
# =============================================================================
ROWS = 6
COLS = 7
NUM_ACTIONS = 7
EMPTY = 0
PLAYER0 = 1
PLAYER1 = 2

# =============================================================================
# JAX Core - Pure functions for fast batched simulation
# =============================================================================

@jax.jit
def init_state():
    """Create initial game state as JAX arrays."""
    return {
        'board': jnp.zeros((ROWS, COLS), dtype=jnp.int8),
        'column_heights': jnp.zeros(COLS, dtype=jnp.int8),  # Next empty row per column (from bottom)
        'current_player': jnp.int8(0),
        'done': jnp.bool_(False),
        'winner': jnp.int8(-1),  # -1 = none, 0 = player0, 1 = player1, 2 = draw
    }


@jax.jit
def legal_actions_mask(state) -> jnp.ndarray:
    """Return mask of legal actions (columns that aren't full)."""
    return state['column_heights'] < ROWS


@jax.jit
def check_line(board, r, c, dr, dc, player):
    """Check if there's a 4-in-a-row starting at (r,c) in direction (dr,dc)."""
    player_val = player + 1  # Convert 0/1 to 1/2 for board representation
    count = 0
    for i in range(4):
        nr, nc = r + i * dr, c + i * dc
        in_bounds = (nr >= 0) & (nr < ROWS) & (nc >= 0) & (nc < COLS)
        matches = jnp.where(in_bounds, board[nr % ROWS, nc % COLS] == player_val, False)
        count += matches.astype(jnp.int8)
    return count == 4


@jax.jit
def check_winner_at(board, r, c, player):
    """Check if player has won with a piece at (r, c)."""
    # Check all 4 directions: horizontal, vertical, and both diagonals
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    won = False
    for dr, dc in directions:
        # Check starting positions that could include (r, c)
        for offset in range(4):
            start_r = r - offset * dr
            start_c = c - offset * dc
            won = won | check_line(board, start_r, start_c, dr, dc, player)
    return won


@jax.jit
def check_winner_full(board):
    """Check entire board for a winner. Returns -1 (none), 0, 1, or 2 (draw)."""
    # Check for player 0 win
    p0_wins = False
    p1_wins = False

    for r in range(ROWS):
        for c in range(COLS):
            # Horizontal
            if c <= COLS - 4:
                p0_wins = p0_wins | ((board[r, c] == 1) & (board[r, c+1] == 1) &
                                      (board[r, c+2] == 1) & (board[r, c+3] == 1))
                p1_wins = p1_wins | ((board[r, c] == 2) & (board[r, c+1] == 2) &
                                      (board[r, c+2] == 2) & (board[r, c+3] == 2))
            # Vertical
            if r <= ROWS - 4:
                p0_wins = p0_wins | ((board[r, c] == 1) & (board[r+1, c] == 1) &
                                      (board[r+2, c] == 1) & (board[r+3, c] == 1))
                p1_wins = p1_wins | ((board[r, c] == 2) & (board[r+1, c] == 2) &
                                      (board[r+2, c] == 2) & (board[r+3, c] == 2))
            # Diagonal down-right
            if r <= ROWS - 4 and c <= COLS - 4:
                p0_wins = p0_wins | ((board[r, c] == 1) & (board[r+1, c+1] == 1) &
                                      (board[r+2, c+2] == 1) & (board[r+3, c+3] == 1))
                p1_wins = p1_wins | ((board[r, c] == 2) & (board[r+1, c+1] == 2) &
                                      (board[r+2, c+2] == 2) & (board[r+3, c+3] == 2))
            # Diagonal down-left
            if r <= ROWS - 4 and c >= 3:
                p0_wins = p0_wins | ((board[r, c] == 1) & (board[r+1, c-1] == 1) &
                                      (board[r+2, c-2] == 1) & (board[r+3, c-3] == 1))
                p1_wins = p1_wins | ((board[r, c] == 2) & (board[r+1, c-1] == 2) &
                                      (board[r+2, c-2] == 2) & (board[r+3, c-3] == 2))

    # -1 = no winner, 0 = player0, 1 = player1, 2 = draw
    is_full = jnp.all(board != 0)
    winner = jnp.where(p0_wins, jnp.int8(0),
                       jnp.where(p1_wins, jnp.int8(1),
                                 jnp.where(is_full, jnp.int8(2), jnp.int8(-1))))
    return winner


@jax.jit
def step(state, action):
    """Apply action and return new state. Pure function."""
    board = state['board']
    col_heights = state['column_heights']
    current_player = state['current_player']

    # Get row where piece lands (from bottom: row 5 - height)
    height = col_heights[action]
    row = ROWS - 1 - height

    # Place piece (player 0 -> 1, player 1 -> 2)
    piece = current_player + 1
    new_board = board.at[row, action].set(piece)
    new_heights = col_heights.at[action].set(height + 1)

    # Check for winner
    winner = check_winner_full(new_board)
    done = winner >= 0

    # Switch player
    new_player = 1 - current_player

    return {
        'board': new_board,
        'column_heights': new_heights,
        'current_player': jnp.where(done, current_player, new_player),
        'done': done,
        'winner': winner,
    }


@jax.jit
def get_observation(state, player):
    """Get observation tensor matching OpenSpiel format [3, 6, 7].

    Plane 0: Empty cells
    Plane 1: Player 0's pieces
    Plane 2: Player 1's pieces
    """
    board = state['board']
    empty = (board == 0).astype(jnp.float32)
    player0 = (board == 1).astype(jnp.float32)
    player1 = (board == 2).astype(jnp.float32)
    return jnp.stack([empty, player0, player1], axis=0)


@jax.jit
def get_returns(state):
    """Get returns for both players."""
    winner = state['winner']
    # winner: -1 = ongoing, 0 = p0 wins, 1 = p1 wins, 2 = draw
    p0_return = jnp.where(winner == 0, 1.0, jnp.where(winner == 1, -1.0, 0.0))
    p1_return = jnp.where(winner == 1, 1.0, jnp.where(winner == 0, -1.0, 0.0))
    return jnp.array([p0_return, p1_return])


# =============================================================================
# Batched operations for fast parallel game simulation
# =============================================================================

# Vectorized versions for running many games in parallel
batched_init = jax.vmap(lambda _: init_state())
batched_step = jax.vmap(step)
batched_legal_actions_mask = jax.vmap(legal_actions_mask)
batched_get_observation = jax.vmap(get_observation, in_axes=(0, None))
batched_get_returns = jax.vmap(get_returns)


def init_batch(batch_size):
    """Initialize a batch of games."""
    return batched_init(jnp.arange(batch_size))


# =============================================================================
# OpenSpiel Python Game Interface
# =============================================================================

_GAME_TYPE = pyspiel.GameType(
    short_name="python_connect_four_jax",
    long_name="Python Connect Four (JAX)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=NUM_ACTIONS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=ROWS * COLS,
)


class ConnectFourJaxState(pyspiel.State):
    """OpenSpiel state wrapper around JAX state."""

    def __init__(self, game, jax_state=None):
        super().__init__(game)
        if jax_state is None:
            self._jax_state = init_state()
        else:
            self._jax_state = jax_state
        self._cached_legal_actions = None

    def current_player(self):
        if self._jax_state['done']:
            return pyspiel.PlayerId.TERMINAL
        return int(self._jax_state['current_player'])

    def legal_actions(self, player=None):
        if self._cached_legal_actions is None:
            mask = legal_actions_mask(self._jax_state)
            self._cached_legal_actions = [i for i in range(NUM_ACTIONS) if mask[i]]
        return self._cached_legal_actions

    def legal_actions_mask(self, player=None):
        mask = legal_actions_mask(self._jax_state)
        return [int(m) for m in mask]

    def _apply_action(self, action):
        self._jax_state = step(self._jax_state, action)
        self._cached_legal_actions = None

    def is_terminal(self):
        return bool(self._jax_state['done'])

    def returns(self):
        returns = get_returns(self._jax_state)
        return [float(returns[0]), float(returns[1])]

    def player_return(self, player):
        return self.returns()[player]

    def observation_tensor(self, player=None):
        obs = get_observation(self._jax_state, 0)
        return list(obs.flatten().astype(float))

    def observation_string(self, player=None):
        return str(self)

    def __str__(self):
        board = np.array(self._jax_state['board'])
        chars = {0: '.', 1: 'x', 2: 'o'}
        lines = []
        for row in board:
            lines.append(''.join(chars[int(c)] for c in row))
        return '\n'.join(lines)

    def clone(self):
        cloned = ConnectFourJaxState(self.get_game())
        cloned._jax_state = jax.tree.map(lambda x: x.copy(), self._jax_state)
        return cloned


class ConnectFourJaxGame(pyspiel.Game):
    """OpenSpiel game wrapper for JAX Connect Four."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})

    def new_initial_state(self):
        return ConnectFourJaxState(self)

    def num_distinct_actions(self):
        return NUM_ACTIONS

    def num_players(self):
        return 2

    def min_utility(self):
        return -1.0

    def max_utility(self):
        return 1.0

    def observation_tensor_shape(self):
        return [3, ROWS, COLS]

    def observation_tensor_size(self):
        return 3 * ROWS * COLS

    def max_game_length(self):
        return ROWS * COLS


# Register the game
pyspiel.register_game(_GAME_TYPE, ConnectFourJaxGame)


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Testing JAX Connect Four ===\n")

    # Test single game via OpenSpiel interface
    print("1. Testing OpenSpiel interface:")
    game = pyspiel.load_game("python_connect_four_jax")
    state = game.new_initial_state()
    print(f"   Initial state:\n{state}\n")
    print(f"   Legal actions: {state.legal_actions()}")
    print(f"   Current player: {state.current_player()}")

    # Play a few moves
    state.apply_action(3)
    state.apply_action(3)
    print(f"\n   After moves 3, 3:\n{state}")
    print(f"   Observation shape: {game.observation_tensor_shape()}")

    # Test JAX batched operations
    print("\n2. Testing JAX batched operations:")
    batch_size = 1000
    states = init_batch(batch_size)
    print(f"   Initialized {batch_size} parallel games")

    # Random actions for all games
    key = jax.random.PRNGKey(42)
    actions = jax.random.randint(key, (batch_size,), 0, NUM_ACTIONS)

    # Step all games
    import time
    start = time.perf_counter()
    for _ in range(42):  # Max game length
        masks = batched_legal_actions_mask(states)
        # Simple random valid action selection
        actions = jax.random.randint(jax.random.fold_in(key, _), (batch_size,), 0, NUM_ACTIONS)
        actions = jnp.where(masks.sum(axis=1) > 0, actions % masks.sum(axis=1).astype(jnp.int32), 0)
        states = batched_step(states, actions)
    elapsed = time.perf_counter() - start
    print(f"   Played {batch_size} games to completion in {elapsed*1000:.2f}ms")
    print(f"   Games per second: {batch_size / elapsed:.0f}")

    print("\n=== Done ===")

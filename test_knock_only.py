"""Test knock/layoff flow."""
import numpy as np
import gin_rummy_jax as jax_gin
import pyspiel

np.random.seed(36)
jax_game = pyspiel.load_game("python_gin_rummy_jax")
state = jax_game.new_initial_state()
move_count = 0

while not state.is_terminal() and move_count < 200:
    legal = state.legal_actions()
    if 55 in legal:
        print(f"KNOCK at move {move_count}")
        state.apply_action(55)

        # Keep playing through knock/layoff phases
        while not state.is_terminal():
            legal = state.legal_actions()
            phase = int(state._jax_state['phase'])
            player = state.current_player()
            print(f"Phase: {phase}, Player: {player}, Legal: {sorted(legal)[:15]}")

            if phase == 4:  # Knock phase
                # Lay some melds if available
                meld_actions = [a for a in legal if a >= 56]
                if meld_actions:
                    print(f"  -> Laying meld {meld_actions[0]}")
                    state.apply_action(meld_actions[0])
                elif 54 in legal:
                    print("  -> PASS (knocker done)")
                    state.apply_action(54)
                elif legal:
                    print(f"  -> Discard {legal[0]}")
                    state.apply_action(legal[0])
            elif phase == 5:  # Layoff phase
                fl = state._jax_state['finished_layoffs']
                print(f"  finished_layoffs: {fl}")
                if 54 in legal:
                    print("  -> PASS")
                    state.apply_action(54)
                elif legal:
                    print(f"  -> action {legal[0]}")
                    state.apply_action(legal[0])
            else:
                break

        print(f"\nTerminal: {state.is_terminal()}")
        print(f"Returns: {state.returns()}")
        dw = state._jax_state['deadwood']
        print(f"Deadwood: [{dw[0]}, {dw[1]}]")
        lm = state._jax_state['layed_melds']
        p0_melds = [i for i in range(jax_gin.NUM_MELDS) if lm[0, i]]
        p1_melds = [i for i in range(jax_gin.NUM_MELDS) if lm[1, i]]
        print(f"Player 0 melds: {p0_melds}")
        print(f"Player 1 melds: {p1_melds}")
        break

    action = np.random.choice(legal)
    state.apply_action(action)
    move_count += 1

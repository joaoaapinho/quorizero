"""A genuine challenge game passes review; one with an edited bot does not."""

import random

from engine.actions import legal_actions, step
from engine.pathcheck import shortest_distance
from engine.state import GOAL_ROWS, initial_state
from ingestion.verify_claim import review
from mcts.puct import agent, network_evaluator
from model.network import create_model

SIMS = 4


def race(game):
    """The player: always the pawn move that shortens their path most."""
    pawn_moves = [a for a in legal_actions(game) if a < 12]
    me = game.to_move

    def path_after(a):
        nxt = step(game, a)[0]
        return shortest_distance(nxt.walls, nxt.pawns[me], GOAL_ROWS[me])

    return min(pawn_moves, key=path_after)


def play(bot_move, human_side=1):
    game, moves, done, win = initial_state(), [], False, None
    while not done and len(moves) < 200:
        action = race(game) if game.to_move == human_side else bot_move(game)
        moves.append(action)
        game, done, win = step(game, action)
    return {
        "status": "completed",
        "winner": win,
        "moves": moves,
        "human_side": human_side,
    }


def test_genuine_bot_matches_and_edited_bot_does_not():
    model = create_model(num_blocks=2, channels=16)
    real_bot = agent(network_evaluator(model), SIMS)
    genuine = review(play(real_bot), model, SIMS, check_player=False)
    assert genuine["valid"] and genuine["bot_match"] == 1.0

    rng = random.Random(0)
    edited = review(play(lambda g: rng.choice(legal_actions(g))), model, SIMS, False)
    assert edited["valid"] and edited["bot_match"] < 0.9 and edited["mismatches"]


def test_rejects_an_illegal_record():
    model = create_model(num_blocks=2, channels=16)
    record = {"status": "completed", "winner": 1, "moves": [130, 130], "human_side": 1}
    assert review(record, model, SIMS) == {"valid": False}

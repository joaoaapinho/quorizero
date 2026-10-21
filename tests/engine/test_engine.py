"""Rule-correctness tests for the Quoridor engine (Section 1 checklist)."""

from engine.actions import (
    ACTION_SIZE,
    decode_action,
    encode_move,
    encode_wall,
    legal_action_mask,
    legal_actions,
    step,
)
from engine.moves import pawn_moves, wall_placements
from engine.pathcheck import has_path, wall_keeps_paths
from engine.state import GameState, initial_state, winner


def mk(pawns, walls=(), left=(10, 10), to_move=0):
    return GameState(
        pawns=pawns, walls=frozenset(walls), walls_left=left, to_move=to_move
    )


# --- pawn movement -------------------------------------------------------


def test_basic_movement_all_directions():
    state = mk(((4, 4), (0, 0)))
    assert set(pawn_moves(state)) == {(3, 4), (5, 4), (4, 5), (4, 3)}


def test_movement_blocked_by_wall():
    state = mk(((4, 4), (0, 0)), walls={("H", 4, 4)})
    # The horizontal wall under row 4 removes the downward step.
    assert (5, 4) not in pawn_moves(state)
    assert (3, 4) in pawn_moves(state)


# --- jumps ---------------------------------------------------------------


def test_straight_jump_over_opponent():
    state = mk(((4, 4), (5, 4)))
    moves = pawn_moves(state)
    assert (6, 4) in moves
    assert (5, 4) not in moves


def test_diagonal_jump_when_straight_blocked_by_wall():
    # Opponent below, wall behind opponent -> straight jump illegal, diagonals open.
    state = mk(((4, 4), (5, 4)), walls={("H", 5, 4)})
    moves = pawn_moves(state)
    assert (6, 4) not in moves
    assert (5, 5) in moves and (5, 3) in moves


def test_diagonal_jump_at_board_edge():
    # Opponent on the left edge -> straight jump off-board, diagonals open.
    state = mk(((4, 1), (4, 0)))
    moves = pawn_moves(state)
    assert (4, -1) not in moves
    assert (5, 0) in moves and (3, 0) in moves


# --- wall placement legality --------------------------------------------


def test_wall_rejected_when_overlapping():
    state = mk(((0, 4), (8, 4)), walls={("H", 3, 3)})
    placements = wall_placements(state)
    for overlap in [("H", 3, 3), ("H", 3, 2), ("H", 3, 4), ("V", 3, 3)]:
        assert overlap not in placements
    assert ("H", 3, 5) in placements  # non-conflicting slot still allowed


def test_wall_rejected_when_it_fully_blocks_a_player():
    # H(7,7) seals the top of the (8,7)/(8,8) pocket; V(7,6) would close its exit.
    state = mk(((0, 4), (8, 8)), walls={("H", 7, 7)})
    assert wall_keeps_paths(state, ("V", 7, 6)) is False
    assert ("V", 7, 6) not in wall_placements(state)


def test_wall_allowed_when_it_only_narrows_a_path():
    state = mk(((0, 4), (8, 8)), walls={("H", 7, 7)})
    assert wall_keeps_paths(state, ("V", 7, 5)) is True
    assert ("V", 7, 5) in wall_placements(state)


def test_has_path_detects_full_enclosure():
    walls = frozenset({("H", 7, 7), ("V", 7, 6)})
    assert has_path(walls, (8, 8), goal_row=0) is False
    assert has_path(walls, (0, 4), goal_row=8) is True


def test_distance_map_open_and_detour():
    from engine.pathcheck import distance_map

    dm = distance_map(frozenset(), goal_row=8)
    assert dm[8, 4] == 0  # already on the goal row
    assert dm[0, 4] == 8  # straight down the board
    # Walls in front push cells behind them further from the goal.
    walls = frozenset({("H", 0, 3), ("H", 0, 5)})
    assert distance_map(walls, 8)[0, 4] == 10


# --- win detection -------------------------------------------------------


def test_win_detection_player0():
    state = mk(((7, 4), (0, 0)), to_move=0)
    _, done, win = step(state, encode_move((7, 4), (8, 4)))
    assert done and win == 0


def test_win_detection_player1():
    state = mk(((8, 8), (1, 4)), to_move=1)
    _, done, win = step(state, encode_move((1, 4), (0, 4)))
    assert done and win == 1


def test_winner_helper():
    assert winner(mk(((8, 3), (2, 2)))) == 0
    assert winner(mk(((3, 3), (0, 2)))) == 1
    assert winner(initial_state()) is None


# --- action space --------------------------------------------------------


def test_action_encode_decode_roundtrip():
    state = initial_state()
    a = encode_move((0, 4), (1, 4))
    assert decode_action(state, a) == ("move", (1, 4))
    w = encode_wall(("V", 3, 5))
    assert decode_action(state, w) == ("wall", ("V", 3, 5))
    assert 0 <= a < ACTION_SIZE and 0 <= w < ACTION_SIZE


def test_legal_actions_and_mask_agree():
    state = initial_state()
    actions = legal_actions(state)
    mask = legal_action_mask(state)
    assert len(mask) == ACTION_SIZE
    assert [i for i, v in enumerate(mask) if v] == sorted(actions)
    # 3 pawn moves (up is off-board) + 128 non-conflicting wall slots.
    assert len(actions) == 3 + 128


# --- full scripted game --------------------------------------------------


def test_full_scripted_game_reaches_terminal():
    # P0 marches straight down col 4; P1 shuffles aside to avoid collisions.
    script = [
        ((0, 4), (1, 4)),
        ((8, 4), (8, 3)),
        ((1, 4), (2, 4)),
        ((8, 3), (8, 2)),
        ((2, 4), (3, 4)),
        ((8, 2), (8, 3)),
        ((3, 4), (4, 4)),
        ((8, 3), (8, 2)),
        ((4, 4), (5, 4)),
        ((8, 2), (8, 3)),
        ((5, 4), (6, 4)),
        ((8, 3), (8, 2)),
        ((6, 4), (7, 4)),
        ((8, 2), (8, 3)),
        ((7, 4), (8, 4)),  # P0 reaches goal row
    ]
    state = initial_state()
    done = False
    win = None
    for i, (frm, to) in enumerate(script):
        assert not done, "game ended before the final scripted move"
        assert state.pawns[state.to_move] == frm
        state, done, win = step(state, encode_move(frm, to))
        assert done == (i == len(script) - 1)
    assert win == 0

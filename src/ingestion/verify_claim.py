"""
Review a challenge claim: legal replay, bot moves re-run, engine-help check
QZ_EXPORT_TOKEN=... python -m ingestion.verify_claim <game_id> --api https://<worker>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from engine.actions import decode_action
from engine.state import GameState
from ingestion.games import fetch_game, replay
from mcts.puct import agent, network_evaluator
from model.checkpoint import load_model

BOT_MATCH_OK = 0.9  # below this, the bot was not the real QuoriZero
ENGINE_HELP = 0.85  # above this, look closely for engine assistance


def notation(game: GameState, action: int) -> str:
    """Same coordinates as the website: columns a-i, rows 0-8."""
    kind, payload = decode_action(game, action)
    if kind == "move":
        r, c = payload
        return f"{'abcdefghi'[c]}{r}"
    orient, r, c = payload
    return f"{'abcdefgh'[c]}{r}{orient.lower()}"


def review(record: dict, model, sims: int, check_player: bool = True) -> dict:
    """Re-run the bot over a stored game; return match counts and mismatches."""
    positions = replay(record)
    if positions is None:
        return {"valid": False}
    bot = agent(network_evaluator(model), sims)
    human = record["human_side"]
    bot_moves = bot_same = player_moves = player_same = 0
    mismatches = []
    for ply, (game, played) in enumerate(zip(positions, record["moves"])):
        is_bot = game.to_move != human
        if not is_bot and not check_player:
            continue
        choice = bot(game)
        same = choice == played
        if is_bot:
            bot_moves += 1
            bot_same += same
            if not same:
                mismatches.append(
                    (ply + 1, notation(game, played), notation(game, choice))
                )
        else:
            player_moves += 1
            player_same += same
        who = "bot   " if is_bot else "player"
        mark = "=" if same else "x"
        print(
            f"  {ply + 1:>3}. {who} {notation(game, played):<5} "
            f"QuoriZero: {notation(game, choice):<5} {mark}",
            flush=True,
        )
    return {
        "valid": True,
        "player_won": record["winner"] == human,
        "bot_match": bot_same / max(bot_moves, 1),
        "bot_moves": bot_moves,
        "player_agreement": player_same / max(player_moves, 1),
        "player_moves": player_moves,
        "mismatches": mismatches,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Review a challenge claim")
    p.add_argument("game_id")
    p.add_argument("--api", required=True, help="games API base URL")
    p.add_argument("--runs", default="runs", help="where model versions live")
    p.add_argument("--skip-player", action="store_true", help="skip the engine check")
    args = p.parse_args()

    record = fetch_game(args.api, os.environ["QZ_EXPORT_TOKEN"], args.game_id)
    print(
        f"game {record['id']}: {record['difficulty']}, {record['sims']} sims, "
        f"model {record['model_version']}, {len(record['moves'])} moves"
    )
    # rules first: an illegal game needs no model to be rejected
    if replay(record) is None:
        print(
            "\nREJECT: the game does not replay legally or did not finish as claimed."
        )
        return
    version = record["model_version"]
    run, _, checkpoint = version.partition("/")
    if not run or not checkpoint:
        print(f"\nREJECT: '{version}' is not a QuoriZero model, no real model played.")
        return
    path = Path(args.runs) / version
    if not path.exists() or not (path.parent / "config.json").exists():
        print(
            f"\nNeed the model {version} to re-run the bot. Retrained models live on "
            f"Modal:\n  modal volume get quorizero-runs {version} {args.runs}/{run}/\n"
            f"  modal volume get quorizero-runs {run}/config.json "
            f"{args.runs}/{run}/config.json\nthen run this again."
        )
        return
    model = load_model(path)
    result = review(record, model, record["sims"], not args.skip_player)

    print()
    if not result["valid"]:
        print("REJECT: the game does not replay legally or did not finish as claimed.")
        return
    print(f"player won:        {'yes' if result['player_won'] else 'NO'}")
    print(
        f"bot moves matched: {result['bot_match']:.0%} of {result['bot_moves']} "
        f"(genuine games: nearly all)"
    )
    for ply, played, expected in result["mismatches"]:
        print(f"  move {ply}: bot played {played}, QuoriZero plays {expected}")
    if not args.skip_player:
        print(
            f"player = QuoriZero's first choice: {result['player_agreement']:.0%} "
            f"of {result['player_moves']}"
        )
    print()
    if not result["player_won"]:
        print("REJECT: the player did not win this game.")
    elif result["bot_match"] < BOT_MATCH_OK:
        print(
            "REJECT: the bot's moves are not QuoriZero's (edited bot or wrong model)."
        )
    elif not args.skip_player and result["player_agreement"] >= ENGINE_HELP:
        print("LOOK CLOSER: the player matched the engine almost every move.")
    else:
        print("LOOKS GENUINE: replay the game yourself before paying.")


if __name__ == "__main__":
    main()

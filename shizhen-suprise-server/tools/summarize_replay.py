from __future__ import annotations

import argparse
import json
from pathlib import Path


def _last_record(path: Path) -> dict:
    last = ""
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                last = line
    if not last:
        raise SystemExit(f"{path} is empty")
    return json.loads(last)


def summarize(path: Path, target_name_prefix: str = "target") -> list[dict]:
    record = _last_record(path)
    snapshot = record["state_snapshot"]
    players = snapshot["players"]
    entities = snapshot["entities"].values()

    rows: dict[str, dict] = {}
    for pid, player in players.items():
        rows[pid] = {
            "player_id": pid,
            "name": player["name"],
            "alive": player["alive"],
            "turn": snapshot["turn_number"],
            "gold": int(player.get("resources", {}).get("gold", 0)),
            "bases": 0,
            "buildings": 0,
            "units": 0,
            "hp": 0,
            "target": str(player["name"]).startswith(target_name_prefix),
        }

    for entity in entities:
        owner = entity.get("owner_id")
        if owner not in rows:
            continue
        rows[owner]["hp"] += int(entity.get("hp", 0))
        if entity.get("type") == "Base":
            rows[owner]["bases"] += 1
        if "is_complete" in entity:
            rows[owner]["buildings"] += 1
        else:
            rows[owner]["units"] += 1

    for row in rows.values():
        row["score"] = (
            row["bases"] * 10_000
            + row["buildings"] * 300
            + row["units"] * 100
            + row["hp"]
            + row["gold"]
        )

    return sorted(rows.values(), key=lambda row: row["score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--target-prefix", default="target")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    rows = summarize(args.replay, args.target_prefix)
    target_rows = [row for row in rows if row["target"]]
    print(f"Replay: {args.replay}")
    print(f"Turn: {rows[0]['turn'] if rows else 0}")
    for index, row in enumerate(rows[: args.top], start=1):
        marker = " *" if row["target"] else ""
        print(
            f"{index:2d}. {row['name']:<18} score={row['score']:<6} "
            f"bases={row['bases']} buildings={row['buildings']} "
            f"units={row['units']} hp={row['hp']} gold={row['gold']}"
            f"{marker}"
        )
    if target_rows:
        target = target_rows[0]
        rank = rows.index(target) + 1
        print(
            f"Target rank: {rank}/{len(rows)} score={target['score']} "
            f"bases={target['bases']} buildings={target['buildings']} "
            f"units={target['units']} hp={target['hp']} gold={target['gold']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from json import JSONDecodeError
from pathlib import Path
from statistics import mean


def family(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("gpt-") or "shizhen-gpt" in lowered or lowered == "gpt":
        return "gpt"
    if lowered.startswith("target-gpt"):
        return "gpt"
    if lowered.startswith("gemini") or "shizhen-gemini" in lowered:
        return "gemini"
    if lowered.startswith("yilin") or "yilin" in lowered:
        return "yilin"
    if lowered.startswith("v1-") or "shizhen-v1" in lowered:
        return "v1"
    if lowered.startswith("baseline") or lowered.startswith("random"):
        return "baseline"
    if lowered.startswith("xinyang"):
        return "xinyang"
    return lowered.split("-")[0]


def canonical_name(name: str) -> str:
    lowered = name.lower()
    lowered = re.sub(r"-\d+$", "", lowered)
    lowered = re.sub(r"-0\d+$", "", lowered)
    lowered = re.sub(r"-(?:clone|frozen)-?\d*$", "", lowered)
    return lowered


def last_record(path: Path) -> tuple[int, dict, int]:
    last = None
    count = 0
    bad = 0
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.strip():
                try:
                    last = json.loads(line)
                    count += 1
                except JSONDecodeError:
                    bad += 1
    if last is None:
        raise ValueError(f"{path} is empty")
    return count, last, bad


def analyze_replay(path: Path) -> dict:
    records, record, bad_lines = last_record(path)
    snapshot = record["state_snapshot"]
    players = snapshot["players"]
    entities = snapshot["entities"].values()

    rows: dict[str, dict] = {}
    for player_id, player in players.items():
        name = player["name"]
        rows[player_id] = {
            "match_id": path.stem,
            "player_id": player_id,
            "name": name,
            "canonical": canonical_name(name),
            "family": family(name),
            "alive": bool(player.get("alive")),
            "turn": int(snapshot.get("turn_number", 0)),
            "records": records,
            "gold": int(player.get("resources", {}).get("gold", 0)),
            "bases": 0,
            "buildings": 0,
            "units": 0,
            "hp": 0,
        }

    for entity in entities:
        owner = entity.get("owner_id")
        row = rows.get(owner)
        if row is None:
            continue
        row["hp"] += int(entity.get("hp", 0))
        if entity.get("type") == "Base":
            row["bases"] += 1
        if "is_complete" in entity:
            row["buildings"] += 1
        else:
            row["units"] += 1

    for row in rows.values():
        row["material"] = (
            row["bases"] * 10_000
            + row["buildings"] * 300
            + row["units"] * 100
            + row["hp"]
        )
        row["score"] = row["material"] + row["gold"]

    ranked = sorted(rows.values(), key=lambda row: row["score"], reverse=True)
    gold_ranked = sorted(rows.values(), key=lambda row: row["gold"], reverse=True)
    material_ranked = sorted(rows.values(), key=lambda row: row["material"], reverse=True)
    alive = [row for row in rows.values() if row["alive"]]
    sole_survivor = alive[0] if len(alive) == 1 else None

    return {
        "match_id": path.stem,
        "path": str(path),
        "turn": int(snapshot.get("turn_number", 0)),
        "records": records,
        "bad_lines": bad_lines,
        "players": len(rows),
        "alive_count": len(alive),
        "rows": ranked,
        "top_score": ranked[0],
        "top_gold": gold_ranked[0],
        "top_material": material_ranked[0],
        "sole_survivor": sole_survivor,
    }


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=Path("replays"))
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    replay_paths = sorted(args.replay_dir.glob("*.jsonl"))
    summaries = [analyze_replay(path) for path in replay_paths]

    top_score_counter = Counter(s["top_score"]["canonical"] for s in summaries)
    top_gold_counter = Counter(s["top_gold"]["canonical"] for s in summaries)
    sole_counter = Counter(
        s["sole_survivor"]["canonical"] for s in summaries if s["sole_survivor"] is not None
    )
    family_score_counter = Counter(s["top_score"]["family"] for s in summaries)
    family_gold_counter = Counter(s["top_gold"]["family"] for s in summaries)
    family_sole_counter = Counter(
        s["sole_survivor"]["family"] for s in summaries if s["sole_survivor"] is not None
    )

    all_rows = [row for summary in summaries for row in summary["rows"]]
    by_canonical: dict[str, list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_canonical[row["canonical"]].append(row)
        by_family[row["family"]].append(row)

    def aggregate(groups: dict[str, list[dict]]) -> list[dict]:
        output = []
        for name, rows in groups.items():
            output.append(
                {
                    "name": name,
                    "entries": len(rows),
                    "alive_rate": sum(1 for row in rows if row["alive"]) / len(rows),
                    "avg_score": mean(row["score"] for row in rows),
                    "avg_gold": mean(row["gold"] for row in rows),
                    "max_gold": max(row["gold"] for row in rows),
                    "max_score": max(row["score"] for row in rows),
                    "max_material": max(row["material"] for row in rows),
                }
            )
        return sorted(output, key=lambda row: (row["avg_score"], row["max_score"]), reverse=True)

    canonical_agg = aggregate(by_canonical)
    family_agg = aggregate(by_family)
    highest_gold = sorted(all_rows, key=lambda row: row["gold"], reverse=True)[:15]
    highest_score = sorted(all_rows, key=lambda row: row["score"], reverse=True)[:15]
    highest_material = sorted(all_rows, key=lambda row: row["material"], reverse=True)[:15]

    lines = [
        "# Shizhen Replay Collation",
        "",
        f"Analyzed {len(summaries)} replay files from `{args.replay_dir}`.",
        "Score formula: `bases*10000 + buildings*300 + units*100 + hp + gold`.",
        "Material score excludes banked gold.",
        f"Malformed lines skipped: {sum(s['bad_lines'] for s in summaries)}.",
        "",
        "## Consistent Top Scorers",
        "",
        md_table(
            ["bot", "top-score wins"],
            [[name, count] for name, count in top_score_counter.most_common()],
        ),
        "",
        "## Sole Survivors",
        "",
        md_table(
            ["bot", "sole-survivor wins"],
            [[name, count] for name, count in sole_counter.most_common()],
        )
        if sole_counter
        else "No replay ended with exactly one survivor.",
        "",
        "## Top-Gold Finishers",
        "",
        md_table(
            ["bot", "highest-gold finishes"],
            [[name, count] for name, count in top_gold_counter.most_common()],
        ),
        "",
        "## Family Summary",
        "",
        md_table(
            [
                "family",
                "entries",
                "top-score wins",
                "sole-survivor wins",
                "top-gold finishes",
                "avg score",
                "avg gold",
                "alive rate",
            ],
            [
                [
                    row["name"],
                    row["entries"],
                    family_score_counter[row["name"]],
                    family_sole_counter[row["name"]],
                    family_gold_counter[row["name"]],
                    round(row["avg_score"], 1),
                    round(row["avg_gold"], 1),
                    f"{row['alive_rate']:.0%}",
                ]
                for row in family_agg
            ],
        ),
        "",
        "## Per-Replay Final Leaders",
        "",
        md_table(
            [
                "match",
                "turn",
                "players",
                "alive",
                "top score",
                "score",
                "top gold",
                "gold",
                "top material",
                "material",
                "sole survivor",
            ],
            [
                [
                    s["match_id"],
                    s["turn"],
                    s["players"],
                    s["alive_count"],
                    s["top_score"]["name"],
                    s["top_score"]["score"],
                    s["top_gold"]["name"],
                    s["top_gold"]["gold"],
                    s["top_material"]["name"],
                    s["top_material"]["material"],
                    s["sole_survivor"]["name"] if s["sole_survivor"] else "",
                ]
                for s in summaries
            ],
        ),
        "",
        "## Highest End-State Gold",
        "",
        md_table(
            ["rank", "match", "bot", "gold", "score", "material", "bases", "buildings", "units", "hp", "alive"],
            [
                [
                    index,
                    row["match_id"],
                    row["name"],
                    row["gold"],
                    row["score"],
                    row["material"],
                    row["bases"],
                    row["buildings"],
                    row["units"],
                    row["hp"],
                    row["alive"],
                ]
                for index, row in enumerate(highest_gold, start=1)
            ],
        ),
        "",
        "## Highest End-State Score",
        "",
        md_table(
            ["rank", "match", "bot", "score", "gold", "material", "bases", "buildings", "units", "hp", "alive"],
            [
                [
                    index,
                    row["match_id"],
                    row["name"],
                    row["score"],
                    row["gold"],
                    row["material"],
                    row["bases"],
                    row["buildings"],
                    row["units"],
                    row["hp"],
                    row["alive"],
                ]
                for index, row in enumerate(highest_score, start=1)
            ],
        ),
        "",
        "## Highest End-State Material",
        "",
        md_table(
            ["rank", "match", "bot", "material", "score", "gold", "bases", "buildings", "units", "hp", "alive"],
            [
                [
                    index,
                    row["match_id"],
                    row["name"],
                    row["material"],
                    row["score"],
                    row["gold"],
                    row["bases"],
                    row["buildings"],
                    row["units"],
                    row["hp"],
                    row["alive"],
                ]
                for index, row in enumerate(highest_material, start=1)
            ],
        ),
        "",
        "## Bot Aggregate By Name",
        "",
        md_table(
            ["bot", "entries", "avg score", "avg gold", "max score", "max gold", "max material", "alive rate"],
            [
                [
                    row["name"],
                    row["entries"],
                    round(row["avg_score"], 1),
                    round(row["avg_gold"], 1),
                    row["max_score"],
                    row["max_gold"],
                    row["max_material"],
                    f"{row['alive_rate']:.0%}",
                ]
                for row in canonical_agg
            ],
        ),
        "",
    ]

    report = "\n".join(lines)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(report, encoding="utf-8")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "match_id",
                    "turn",
                    "records",
                    "players",
                    "alive_count",
                    "rank",
                    "name",
                    "canonical",
                    "family",
                    "alive",
                    "score",
                    "material",
                    "gold",
                    "bases",
                    "buildings",
                    "units",
                    "hp",
                ],
            )
            writer.writeheader()
            for summary in summaries:
                for rank, row in enumerate(summary["rows"], start=1):
                    writer.writerow(
                        {
                            "match_id": summary["match_id"],
                            "turn": summary["turn"],
                            "records": summary["records"],
                            "players": summary["players"],
                            "alive_count": summary["alive_count"],
                            "rank": rank,
                            "name": row["name"],
                            "canonical": row["canonical"],
                            "family": row["family"],
                            "alive": row["alive"],
                            "score": row["score"],
                            "material": row["material"],
                            "gold": row["gold"],
                            "bases": row["bases"],
                            "buildings": row["buildings"],
                            "units": row["units"],
                            "hp": row["hp"],
                        }
                    )

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

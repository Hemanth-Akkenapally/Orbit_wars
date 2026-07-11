"""Mine BioHub_Project replay JSON files for daily strategy notes.

This script is intentionally dependency-free. It summarizes replays you already
downloaded and extracts the opening patterns we care about most:

- winner, agents, rewards, status
- production / planets / ships at checkpoints
- first actions and inferred first target
- aggregate opening records for top agents
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CHECKPOINTS = (0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 499)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Replay JSON files or folders containing replay JSON files.")
    parser.add_argument("--out", default="daily_runs/latest", help="Output folder for CSV/Markdown summaries.")
    parser.add_argument("--first-turn-limit", type=int, default=45)
    args = parser.parse_args()

    replay_files = collect_replays(args.paths)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    replay_rows = []
    checkpoint_rows = []
    opening_rows = []

    for replay_file in replay_files:
        data = json.loads(replay_file.read_text(encoding="utf-8"))
        replay_rows.append(replay_summary(replay_file, data))
        checkpoint_rows.extend(checkpoint_summary(replay_file, data))
        opening_rows.extend(opening_summary(replay_file, data, args.first_turn_limit))

    write_csv(out / "replays.csv", replay_rows)
    write_csv(out / "checkpoints.csv", checkpoint_rows)
    write_csv(out / "openings.csv", opening_rows)
    write_markdown(out / "summary.md", replay_rows, checkpoint_rows, opening_rows)

    print(f"Analyzed {len(replay_files)} replay(s)")
    print(out.resolve())
    return 0


def collect_replays(paths):
    files = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("episode-*.json")))
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
    unique = []
    seen = set()
    for file in files:
        resolved = file.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(file)
    return unique


def replay_summary(path, data):
    names = [agent.get("Name", f"P{i}") for i, agent in enumerate(data.get("info", {}).get("Agents", []))]
    rewards = data.get("rewards", [])
    winner = None
    if rewards:
        best = max(range(len(rewards)), key=lambda i: rewards[i] if rewards[i] is not None else -10**9)
        winner = best if rewards[best] is not None else None
    return {
        "file": path.name,
        "episode_id": data.get("info", {}).get("EpisodeId"),
        "seed": data.get("info", {}).get("seed"),
        "steps": len(data.get("steps", [])),
        "agents": " | ".join(names),
        "winner": winner,
        "winner_name": names[winner] if winner is not None and winner < len(names) else "",
        "rewards": json.dumps(rewards),
        "statuses": json.dumps(data.get("statuses", [])),
    }


def checkpoint_summary(path, data):
    rows = []
    steps = data.get("steps", [])
    for step_index in CHECKPOINTS:
        if step_index >= len(steps):
            continue
        for player_index, player_state in enumerate(steps[step_index]):
            obs = player_state.get("observation", {})
            planets = obs.get("planets", [])
            fleets = obs.get("fleets", [])
            owned = [p for p in planets if len(p) >= 7 and p[1] == player_index]
            fleet_ships = sum(f[6] for f in fleets if len(f) >= 7 and f[1] == player_index)
            rows.append({
                "file": path.name,
                "step": step_index,
                "player": player_index,
                "owned_planets": len(owned),
                "production": sum(p[6] for p in owned),
                "planet_ships": sum(p[5] for p in owned),
                "fleet_ships": fleet_ships,
                "total_ships": sum(p[5] for p in owned) + fleet_ships,
                "actions": len(player_state.get("action") or []),
            })
    return rows


def opening_summary(path, data, first_turn_limit):
    rows = []
    steps = data.get("steps", [])
    if not steps:
        return rows
    agents = [agent.get("Name", f"P{i}") for i, agent in enumerate(data.get("info", {}).get("Agents", []))]
    for player_index in range(len(steps[0])):
        for step_index, step in enumerate(steps[:first_turn_limit]):
            action = step[player_index].get("action")
            if not action:
                continue
            obs = step[player_index].get("observation", {})
            planets = obs.get("planets", [])
            planet_by_id = {p[0]: p for p in planets if len(p) >= 7}
            for move_index, move in enumerate(action):
                if len(move) < 3:
                    continue
                source = planet_by_id.get(move[0])
                target = infer_target(source, move[1], planets) if source else None
                rows.append({
                    "file": path.name,
                    "agent": agents[player_index] if player_index < len(agents) else f"P{player_index}",
                    "player": player_index,
                    "step": step_index,
                    "move_index": move_index,
                    "from_planet": move[0],
                    "angle": move[1],
                    "ships": move[2],
                    "source_ships": source[5] if source else "",
                    "source_production": source[6] if source else "",
                    "target_planet": target[0] if target else "",
                    "target_owner": target[1] if target else "",
                    "target_ships": target[5] if target else "",
                    "target_production": target[6] if target else "",
                    "target_distance": round(distance(source, target), 3) if source and target else "",
                })
            break
    return rows


def infer_target(source, angle, planets):
    sx, sy = source[2], source[3]
    dx, dy = math.cos(angle), math.sin(angle)
    best = None
    for planet in planets:
        if len(planet) < 7 or planet[0] == source[0]:
            continue
        fx, fy = planet[2] - sx, planet[3] - sy
        projection = fx * dx + fy * dy
        if projection <= 0:
            continue
        perpendicular_sq = fx * fx + fy * fy - projection * projection
        hit_radius = planet[4] + 0.75
        if perpendicular_sq <= hit_radius * hit_radius:
            if best is None or projection < best[0]:
                best = (projection, planet)
    return best[1] if best else None


def distance(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, replay_rows, checkpoint_rows, opening_rows):
    lines = ["# BioHub_Project Replay Summary", ""]
    lines.append(f"Replays analyzed: {len(replay_rows)}")
    lines.append("")
    lines.append("## Winners")
    for row in replay_rows:
        lines.append(f"- `{row['file']}`: winner P{row['winner']} {row['winner_name']} in {row['steps']} steps")
    lines.append("")
    lines.append("## Opening Patterns")
    for row in opening_rows[:80]:
        target = f" -> {row['target_planet']} ({row['target_ships']} ships, prod {row['target_production']})" if row["target_planet"] != "" else ""
        lines.append(
            f"- `{row['file']}` {row['agent']} P{row['player']} step {row['step']}: "
            f"{row['ships']} ships from {row['from_planet']}{target}"
        )
    lines.append("")
    lines.append("## Production At Turn 75")
    for row in checkpoint_rows:
        if row["step"] == 75:
            lines.append(
                f"- `{row['file']}` P{row['player']}: {row['owned_planets']} planets, "
                f"prod {row['production']}, ships {row['total_ships']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

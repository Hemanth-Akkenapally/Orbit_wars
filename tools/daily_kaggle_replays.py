"""Daily Orbit Wars replay collection helper.

This script automates the parts the Kaggle CLI exposes. The public leaderboard
does not always expose submission IDs, so keep a text file of known top
submission IDs when you find them. The script will:

1. create a dated output folder
2. save the public leaderboard text
3. for each submission ID, fetch episode listings
4. download replay JSON files for the newest episodes
5. run tools/replay_miner.py on the downloaded replays
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="orbit-wars")
    parser.add_argument("--out-root", default="daily_runs")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--submission-ids", default="top_submission_ids.txt")
    parser.add_argument("--episodes-per-submission", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.out_root) / args.date
    replay_dir = out / "replays"
    out.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    kaggle = find_kaggle()
    if kaggle is None:
        print("Kaggle CLI was not found. Install/authenticate it, then rerun this script.")
        print("For now, place downloaded replay JSON files in:", replay_dir.resolve())
        return 0

    run([kaggle, "competitions", "leaderboard", args.competition, "-s"], out / "leaderboard.txt")

    ids_path = Path(args.submission_ids)
    if not ids_path.exists():
        ids_path.write_text(
            "# Add one top submission id per line.\n"
            "# Get IDs from Kaggle episode pages or `kaggle competitions submissions orbit-wars` for your own.\n",
            encoding="utf-8",
        )
        print("Created", ids_path.resolve())
        print("Add top submission IDs, then rerun to download their episodes.")
        return 0

    submission_ids = [
        line.strip()
        for line in ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    episode_ids = []
    for submission_id in submission_ids:
        episodes_file = out / f"episodes-{submission_id}.csv"
        run([kaggle, "competitions", "episodes", submission_id, "-v"], episodes_file)
        episode_ids.extend(read_episode_ids(episodes_file, args.episodes_per_submission))

    for episode_id in sorted(set(episode_ids)):
        run([kaggle, "competitions", "replay", str(episode_id), "-p", str(replay_dir)], None)

    miner = Path(__file__).with_name("replay_miner.py")
    subprocess.run(
        [sys.executable, str(miner), str(replay_dir), "--out", str(out / "summary")],
        check=False,
    )
    print(out.resolve())
    return 0


def run(command, output_path):
    print(" ".join(command))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if output_path is not None:
        output_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    elif result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode


def find_kaggle():
    found = shutil.which("kaggle")
    if found:
        return found
    scripts = Path(os.environ.get("APPDATA", "")) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "kaggle.exe"
    if scripts.exists():
        return str(scripts)
    return None


def read_episode_ids(path, limit):
    text = path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    ids = []
    for row in rows:
        for key in ("EpisodeId", "episodeId", "episode_id", "Id", "id"):
            if key in row and row[key]:
                ids.append(row[key])
                break
        if len(ids) >= limit:
            break
    return ids


if __name__ == "__main__":
    raise SystemExit(main())

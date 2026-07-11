"""Daily BioHub_Project replay collection helper.

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
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="biohub-cell-tracking-during-development")
    parser.add_argument("--out-root", default="daily_runs")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--submission-ids", default="top_submission_ids.txt")
    parser.add_argument("--urls", nargs="*", default=[], help="Kaggle URLs containing submissionId and/or episodeId.")
    parser.add_argument("--url-file", default=None, help="Text file with one Kaggle URL per line.")
    parser.add_argument("--episode-ids", nargs="*", default=[], help="Known episode IDs to download directly.")
    parser.add_argument("--top-n", type=int, default=20)
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

    leaderboard_txt = out / "top20_leaderboard.txt"
    run([kaggle, "competitions", "leaderboard", args.competition, "-s", "--page-size", str(args.top_n)], leaderboard_txt)
    leaderboard_zip_dir = out / "leaderboard_download"
    leaderboard_zip_dir.mkdir(exist_ok=True)
    run([kaggle, "competitions", "leaderboard", args.competition, "-d", "-p", str(leaderboard_zip_dir)], out / "leaderboard_download.log")
    extract_zips(leaderboard_zip_dir)
    leaderboard_rows = extract_leaderboard_rows(leaderboard_zip_dir, args.top_n)
    write_csv(out / "top20_leaderboard.csv", leaderboard_rows)

    url_values = list(args.urls)
    if args.url_file and Path(args.url_file).exists():
        url_values.extend(
            line.strip()
            for line in Path(args.url_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    parsed_urls = [parse_kaggle_url(url) for url in url_values]

    candidate_ids = [str(row["TeamId"]) for row in leaderboard_rows if row.get("TeamId")]
    candidate_ids.extend(item["submission_id"] for item in parsed_urls if item["submission_id"])
    ids_path = Path(args.submission_ids)
    if ids_path.exists():
        candidate_ids.extend(
            line.strip()
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    else:
        ids_path.write_text(
            "# Optional fallback: add known submission IDs here if Kaggle exposes them.\n",
            encoding="utf-8",
        )

    episode_ids = list(args.episode_ids)
    episode_ids.extend(item["episode_id"] for item in parsed_urls if item["episode_id"])
    attempts = []
    for candidate_id in unique(candidate_ids):
        episodes_file = out / f"episodes-{candidate_id}.csv"
        code = run([kaggle, "competitions", "episodes", candidate_id, "-v"], episodes_file)
        ids = read_episode_ids(episodes_file, args.episodes_per_submission) if code == 0 else []
        attempts.append({"candidate_id": candidate_id, "return_code": code, "episodes_found": len(ids)})
        episode_ids.extend(ids)
    write_csv(out / "episode_lookup_attempts.csv", attempts)

    replay_attempts = []
    for episode_id in sorted(set(episode_ids)):
        code = run([kaggle, "competitions", "replay", str(episode_id), "-p", str(replay_dir)], out / f"replay-{episode_id}.log")
        replay_attempts.append({"episode_id": episode_id, "return_code": code})
    write_csv(out / "replay_download_attempts.csv", replay_attempts)

    miner = Path(__file__).with_name("replay_miner.py")
    subprocess.run(
        [sys.executable, str(miner), str(replay_dir), "--out", str(out / "summary")],
        check=False,
    )
    write_report(out, leaderboard_rows, attempts, episode_ids, replay_attempts)
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


def extract_leaderboard_rows(folder, top_n):
    csv_files = sorted(folder.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not csv_files:
        return []
    rows = []
    with csv_files[0].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if len(rows) >= top_n:
                break
    return rows


def extract_zips(folder):
    for archive in folder.glob("*.zip"):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(folder)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def unique(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_kaggle_url(url):
    submission = re.search(r"[?&]submissionId=(\d+)", url)
    episode = re.search(r"[?&]episodeId=(\d+)", url)
    return {
        "url": url,
        "submission_id": submission.group(1) if submission else "",
        "episode_id": episode.group(1) if episode else "",
    }


def write_report(out, leaderboard_rows, attempts, episode_ids, replay_attempts):
    lines = [
        "# Daily BioHub_Project Replay Download",
        "",
        f"Top leaderboard rows captured: {len(leaderboard_rows)}",
        f"Episodes discovered: {len(set(episode_ids))}",
        "",
        "## Top Teams",
    ]
    for row in leaderboard_rows:
        lines.append(f"- #{row.get('Rank')} {row.get('TeamName')} score {row.get('Score')} teamId {row.get('TeamId')}")
    lines.append("")
    lines.append("## Episode Lookup")
    if any(attempt["episodes_found"] for attempt in attempts):
        for attempt in attempts:
            if attempt["episodes_found"]:
                lines.append(f"- `{attempt['candidate_id']}`: {attempt['episodes_found']} episode(s)")
    else:
        lines.extend([
            "No replay episodes were discovered automatically.",
            "",
            "Kaggle exposes `TeamId` in the public leaderboard, but the replay API expects `submission_id`.",
            "When `TeamId` is passed to `kaggle competitions episodes`, Kaggle currently returns 403 Forbidden.",
            "So the script captures the daily top 20 automatically and will download replays if Kaggle later exposes compatible IDs.",
        ])
    if episode_ids:
        lines.append("")
        lines.append("## Direct Replay Downloads")
        for attempt in replay_attempts:
            status = "ok" if attempt["return_code"] == 0 else f"failed ({attempt['return_code']})"
            lines.append(f"- episode `{attempt['episode_id']}`: {status}")
    (out / "DOWNLOAD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

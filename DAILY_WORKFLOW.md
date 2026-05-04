# Daily Orbit Wars Workflow

Goal: use top-20 replays every day to improve `main.py` with controlled patches.

## 1. Collect Replays

If Kaggle CLI is installed and `top_submission_ids.txt` has IDs:

```powershell
python tools/daily_kaggle_replays.py --competition orbit-wars
```

If you manually download replays, put them in a dated folder, then run:

```powershell
python tools/replay_miner.py "C:\Users\akken\Desktop\Deep_learning\Orbit_wars\May_3\Top_replays" --out daily_runs\2026-05-03\summary
```

## 2. Read The Summary

Open:

```text
daily_runs\YYYY-MM-DD\summary\summary.md
daily_runs\YYYY-MM-DD\summary\openings.csv
daily_runs\YYYY-MM-DD\summary\checkpoints.csv
```

What we compare:

- first action turn
- first target ships / production
- ships sent
- production by turn 40, 75, 100
- whether top bots wait for a bigger opening packet
- how often they reinforce captured hubs

## 3. Patch One Behavior

Patch only one behavior per submission:

- opening hold timing
- first target scoring
- hub reinforcement
- sun-safe aiming
- enemy steal timing
- comet capture logic

## 4. Submit

```powershell
kaggle competitions submit -c orbit-wars -f main.py -m "short version note"
```

## 5. Bring Back Results

For each submission, save:

- public score
- validation status
- our replay JSON
- our logs if there is an error or timeout
- one or more top replays from the same day

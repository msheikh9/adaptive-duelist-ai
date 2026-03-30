# Adaptive Duelist AI

A 1v1 combat sandbox where an AI opponent learns your playstyle, predicts your next move, and adapts its strategy across matches — all running locally.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [AI Tier System](#ai-tier-system)
4. [Session Adaptation](#session-adaptation)
5. [Requirements](#requirements)
6. [Setup](#setup)
7. [Running the Game](#running-the-game)
8. [How to Play](#how-to-play)
9. [Controls](#controls)
10. [Local API & Dashboard](#local-api--dashboard)
11. [CLI Tools](#cli-tools)
12. [Training Pipeline](#training-pipeline)
13. [Evaluation & Regression Gates](#evaluation--regression-gates)
14. [Database Maintenance](#database-maintenance)
15. [Project Structure](#project-structure)
16. [Contributing](#contributing)

---

## Overview

Adaptive Duelist AI is a headless-capable Python combat simulation with a layered AI system. The AI builds a persistent behavioral profile of the human player, uses a Markov/sklearn ensemble to predict the player's next commitment, and selects a tactical mode (exploit, bait, defend, probe, punish) via weighted scoring. Across matches within a session, it further adapts via a lightweight in-process session memory without any retraining.

**Key properties:**
- Fully deterministic by seed — every match is replayable
- No network calls — runs entirely on `localhost`
- Modular tier system: swap between random (T0), Markov-only (T1), and fully adaptive (T2)
- A local REST API + single-page dashboard for observability and control

---

## Architecture

The AI operates through four distinct layers:

| Layer | Module | Description |
|-------|--------|-------------|
| Behavior Modeling | `ai/layers/behavior_model.py` | Builds a persistent player profile from observed combat commitments |
| Real-Time Prediction | `ai/layers/prediction_engine.py` | Predicts the player's next commitment via a Markov/sklearn ensemble |
| Tactical Planning | `ai/layers/tactical_planner.py` | Selects strategic modes via weighted scoring |
| Explainability | `analytics/` | Logs reasoning, surfaces patterns, supports replay inspection |

---

## AI Tier System

| Tier | Name | Behavior |
|------|------|----------|
| T0 | Baseline | Random AI — no planner, no prediction |
| T1 | Markov-only | Prediction + strategy scoring, no memory updates |
| T2 | Full Adaptive | Full planner with memory, outcome tracking, session adaptation, shift detection |

---

## Session Adaptation

The AI adapts its tactical strategy across matches **within a session** without retraining. After each match, per-mode outcomes are recorded in a `SessionMemory` with exponential decay — recent matches carry more weight. A player archetype is inferred from the behavioral profile and used to bias mode selection toward strategies effective against that archetype.

**Archetypes:** `AGGRESSIVE`, `DEFENSIVE`, `PATTERNED`, `EVASIVE`, `BALANCED`

Session memory is in-process only (not persisted to the database). It resets when the game restarts. Configuration is in `config/ai_config.yaml` under `session_adaptation` and `archetype_mode_alignment`.

```bash
# View the current player archetype and session stats
python3 scripts/cli.py session-status
```

---

## Requirements

- Python 3.11+
- See `requirements.txt` for all dependencies

Core dependencies: `pygame`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `fastapi`, `uvicorn`, `httpx`

---

## Setup

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate.bat   # Windows

pip install -r requirements.txt
```

---

## Running the Game

```bash
python main.py
# or
python3 scripts/cli.py play
```

---

## How to Play

1. **Launch the game** — a title / controls screen is shown. Review the bindings, then press **Space** or **Enter** to start.
2. **Fight** — you are the blue fighter (left). The red fighter is the AI. Deplete the AI's HP to win.
3. **After the match** — press **R** to restart or **Esc** to quit.
4. **Help at any time** — press **H** during a match (or on the end screen) to open the controls overlay. The match pauses while the overlay is open. Press **H** again (or **Esc**) to close it.

The AI gets smarter the longer you play. It builds a behavioral profile from your moves and adapts its tactics across matches within a session.

---

## Controls

| Key | Action |
|-----|--------|
| `A` / `←` | Move Left |
| `D` / `→` | Move Right |
| `J` / `Z` | Light Attack (fast, low damage) |
| `K` / `X` | Heavy Attack (slow, high damage) |
| `Space` | Dodge Backward |
| `L` / `C` | Block |
| `H` | Open / close controls overlay |
| `R` | Restart match (end screen) |
| `Esc` | Quit (or close help overlay) |

The controls overlay is also shown on the title screen before each session.

---

## Local API & Dashboard

A local FastAPI server exposes the full backend over HTTP. A single-page dashboard is served at `http://localhost:8000/ui/`.

**Start the server:**

```bash
python3 scripts/run_api.py
# or
python3 scripts/cli.py api
```

Options: `--host`, `--port`, `--reload` (dev mode).

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server health and version |
| GET | `/api/config` | Active game and AI config |
| GET | `/api/stats` | Match counts and active model |
| POST | `/api/matches/self-play` | Generate self-play training data |
| POST | `/api/matches/evaluate` | Run a batch evaluation |
| GET | `/api/matches/recent` | List recent matches |
| GET | `/api/matches/{id}/report` | Full explainability report for a match |
| GET | `/api/training/status` | Retrain threshold and delta |
| POST | `/api/training/run` | Run training pipeline |
| POST | `/api/training/curriculum` | Run curriculum-driven training cycle |
| GET | `/api/models/status` | Model registry |
| POST | `/api/models/baseline` | Create evaluation baseline |
| POST | `/api/models/check-regression` | Run regression check vs baseline |

The database path can be overridden via the `ADAPTIVE_DUELIST_DB` environment variable — useful for isolated test environments.

---

## CLI Tools

All tools are accessible via the unified CLI:

```bash
python3 scripts/cli.py <command> [args...]
```

| Command | Description |
|---------|-------------|
| `play` | Launch the game |
| `api` | Start local API server |
| `simulate` | Run headless AI vs AI matches |
| `bulk` | Large batch simulation with CSV/JSONL export |
| `train` | Train sklearn model from match data |
| `explain <match_id>` | Post-match AI explanation report |
| `analyze` | Player tendency analysis across matches |
| `export <match_id>` | Export match report to JSON |
| `replay-audit` | Batch verify all replay files |
| `verify <file>` | Verify a single replay file |
| `db stats\|integrity\|prune\|vacuum` | Database maintenance |
| `profile` | Performance profiler |
| `benchmark` | Repeatable benchmark suite |
| `player-profile` | Show profile evolution across matches |
| `evaluate` | Run AI evaluation suite |
| `check-regression` | Check for regressions vs baseline |
| `create-baseline` | Create frozen baseline artifact |
| `self-play` | Generate self-play training data |
| `train-promote` | Retrain model and promote if gates pass |
| `model-status` | Show model versions and training data stats |
| `curriculum` | Curriculum-driven training cycle |
| `session-status` | Show session adaptation state |

---

## Training Pipeline

### Self-Play Data Generation

```bash
# Generate 50 self-play matches cycling all 5 scripted profiles
python3 scripts/cli.py self-play --matches 50 --profiles RANDOM AGGRESSIVE DEFENSIVE PATTERNED MIXED

# Check training status: model versions, match counts, retrain threshold
python3 scripts/cli.py model-status
```

Scripted profiles: `RANDOM`, `AGGRESSIVE`, `DEFENSIVE`, `PATTERNED`, `MIXED`. All use only Phase 1 commitments. Self-play data is tagged `source='self_play'` and can be filtered separately from human data.

### Retraining

```bash
# Full pipeline: check → retrain → evaluate → promote (if gates pass)
python3 scripts/cli.py train-promote --auto-promote

# Retrain from human matches only
python3 scripts/cli.py train-promote --auto-promote --source-filter human
```

**Exit codes** for `train_and_promote.py`: `0` = promoted / not needed, `1` = regression detected (not promoted), `2` = insufficient data.

### Curriculum Training

The curriculum system allocates self-play matches toward the player's weakest areas using a greedy interleaved strategy:

```bash
python3 scripts/cli.py curriculum --matches 50 --auto-promote
```

---

## Evaluation & Regression Gates

```bash
# Evaluate AI at T2
python3 scripts/cli.py evaluate --tier 2 --matches 50 --seed 0

# Create a baseline snapshot
python3 scripts/cli.py create-baseline --tier 2 --matches 50 --tag v1.0

# Check for regressions (exit 0=pass, 1=fail)
python3 scripts/cli.py check-regression --tier 2 --matches 50
```

Baselines are frozen JSON snapshots stored in `baselines/`. Regression thresholds are configured in `config/eval_config.yaml`.

### Benchmark Workflow

```bash
# Profile core system latencies
python3 scripts/perf_profiler.py --ticks 5000

# Run benchmark suite (1 match + 100 matches + replay verification)
python3 scripts/benchmark.py --matches 100 -o benchmark_results.json

# Bulk self-play evaluation
python3 scripts/bulk_simulate.py --count 1000 --tier 2 --format csv -o results.csv
```

### Analytics Workflow

```bash
# Explain a specific match
python3 scripts/explain_match.py <match_id> --db data/game.db

# Analyze player tendencies across recent matches
python3 scripts/analyze_player.py --last 20

# Export structured report for UI integration
python3 scripts/export_match_report.py <match_id> -o report.json
```

---

## Database Maintenance

```bash
python3 scripts/cli.py db stats
python3 scripts/cli.py db integrity
python3 scripts/cli.py db prune --before-date 2025-01-01 --dry-run
python3 scripts/cli.py db vacuum
```

---

## Project Structure

```
config/          Configuration files and loader
game/            Game engine, entities, combat mechanics
ai/              AI layers, models, training pipeline
  layers/        Behavior model, prediction engine, tactical planner
  models/        Markov, sklearn, ensemble predictors
  strategy/      Strategy selector, action resolver, planner memory
  features/      Feature extraction for ML
  training/      Dataset builder, model trainer, self-play runner, curriculum
  profile/       Player profile, archetype classifier
api/             FastAPI application (routes, schemas, dependencies)
ui/              Single-page dashboard (HTML/JS/CSS) + Pygame rendering
data/            Logging, database, serialization, migrations
evaluation/      Evaluation harness, regression detection, baselines
baselines/       Frozen evaluation snapshots for regression checks
replay/          Deterministic replay system
analytics/       Post-match analysis and explainability
rendering/       Pygame rendering
scripts/         CLI tools, benchmarks, profiling, maintenance
tests/           Unit and integration tests
```

---

## Contributing

1. Run the full test suite before opening a PR: `pytest`
2. All new functionality must include unit tests
3. Evaluation gates (`check-regression`) must pass before merging to `main`
4. Session memory and archetype classification are in-process only — do not persist them to the database

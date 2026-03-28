# Adaptive Duelist

A 1v1 combat sandbox where an AI opponent learns your behavior, predicts your actions, and adapts its strategy in real time.

## Architecture

The AI operates through four distinct layers:

1. **Behavior Modeling** - Builds a persistent player profile from observed combat commitments
2. **Real-Time Prediction** - Predicts the player's next commitment using a Markov/sklearn ensemble
3. **Tactical Planning** - Selects strategic modes (exploit, bait, punish, defend, probe) via weighted scoring
4. **Explainability** - Logs reasoning, surfaces discovered patterns, supports replay inspection

### AI Tier System

| Tier | Description | Behavior |
|------|-------------|----------|
| T0   | Baseline    | Random AI, no planner, no prediction |
| T1   | Markov-only | Prediction + strategy scoring, no memory updates |
| T2   | Full adaptive | Full planner with memory, outcome tracking, shift detection |

## Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt`

## Setup

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Test

```bash
pytest
```

## CLI Tools

All tools are accessible via the unified CLI entrypoint:

```bash
python3 scripts/cli.py <command> [args...]
```

| Command | Description |
|---------|-------------|
| `play` | Launch the game |
| `simulate` | Run headless AI vs AI matches |
| `bulk` | Large batch simulation with CSV/JSONL export |
| `train` | Train sklearn model from match data |
| `explain <match_id>` | Post-match AI explanation report |
| `analyze` | Player tendency analysis across matches |
| `export <match_id>` | Export match report to JSON |
| `replay-audit` | Batch verify all replay files |
| `verify <file>` | Verify a single replay file |
| `db stats\|integrity\|prune\|vacuum` | Database maintenance |
| `profile` | Performance profiler for core systems |
| `benchmark` | Repeatable benchmark suite |
| `player-profile` | Show profile evolution across matches |
| `evaluate` | Run AI evaluation suite |
| `check-regression` | Check for AI regressions vs baseline |
| `create-baseline` | Create frozen baseline artifact |

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

# Export simulation/analytics summaries
python3 scripts/export_results.py simulation-summary --db data/game.db
python3 scripts/export_results.py analytics-summary --last 10
```

### Evaluation and Release Gates

```bash
# Evaluate AI at a specific tier
python3 scripts/evaluate_ai.py --tier 2 --matches 50 --seed 0

# Create a baseline from the current AI
python3 scripts/create_baseline.py --tier 2 --matches 50 --tag v1.0

# Check for regressions against the stored baseline (exit 0=pass, 1=fail)
python3 scripts/check_regression.py --tier 2 --matches 50
```

Baselines are frozen JSON snapshots stored in `baselines/`. Regression thresholds are configured in `config/eval_config.yaml`. See `baselines/README.md` for the schema and update policy.

### Database Maintenance

```bash
python3 scripts/db_maintenance.py stats --db data/game.db
python3 scripts/db_maintenance.py integrity
python3 scripts/db_maintenance.py prune --before-date 2025-01-01 --dry-run
python3 scripts/db_maintenance.py vacuum
```

## Project Structure

```
config/          Configuration files and loader
game/            Game engine, entities, combat mechanics
ai/              AI layers, models, training pipeline
  layers/        Behavior model, prediction engine, tactical planner
  models/        Markov, sklearn, ensemble predictors
  strategy/      Strategy selector, action resolver, planner memory
  features/      Feature extraction for ML
  training/      Dataset builder, model trainer
  profile/       Player profile and profile updater
data/            Logging, database, serialization, migrations
evaluation/      Evaluation harness, regression detection, baselines
baselines/       Frozen evaluation snapshots for regression checks
replay/          Deterministic replay system
analytics/       Post-match analysis and explainability
rendering/       Pygame rendering
scripts/         CLI tools, benchmarks, profiling, maintenance
tests/           Unit and integration tests
```

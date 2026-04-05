# Adaptive Duelist AI

A 1v1 combat sandbox where an AI opponent learns your playstyle, predicts your next move, and adapts its strategy — all running locally with a full ML training pipeline and REST API.

---

## Why This Project Is Interesting

Most "adaptive AI" in games is scripted difficulty scaling. This one is different:

- **Real machine learning in the game loop.** A Random Forest ensemble (Markov + sklearn) predicts your next combat commitment every tick. Prediction confidence drives tactical mode selection — the AI doesn't just react, it tries to anticipate.
- **Full self-contained pipeline.** The same project that runs the game also handles data collection, model training, holdout evaluation, baseline snapshots, regression detection, and model promotion — all from the CLI.
- **Three auditable tiers.** Swap between a random baseline (T0), Markov-only prediction (T1), and the full adaptive system (T2) at the title screen. Useful for isolating how much each layer contributes.
- **Deterministic by seed.** Every match is replayable. Replay files record per-tick checksums; a replay audit tool verifies bit-for-bit consistency.
- **1 033 passing tests.** Unit tests down to individual FSM transitions and stamina accumulator math. Integration tests covering full headless match runs, training pipelines, and API endpoints.
- **Sub-pixel integer physics.** No floating-point position math in the simulation. Positions are stored as `pixels × 100`; physics is exact and reproducible.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Controls](#controls)
3. [Features](#features)
4. [AI Tier System](#ai-tier-system)
5. [Session Adaptation](#session-adaptation)
6. [Local API & Dashboard](#local-api--dashboard)
7. [CLI Tools](#cli-tools)
8. [Training Pipeline](#training-pipeline)
9. [Evaluation & Regression Gates](#evaluation--regression-gates)
10. [Database Maintenance](#database-maintenance)
11. [Project Architecture](#project-architecture)
12. [Roadmap / Future Ideas](#roadmap--future-ideas)
13. [Contributing](#contributing)

---

## Quick Start

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/mutwalli/adaptive-duelist-ai.git
cd adaptive-duelist-ai
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the game
python3 main.py

# 4. Run the local API + dashboard (separate terminal)
python3 scripts/run_api.py
# then open http://localhost:8000/ui/

# 5. Run all tests
python3 -m pytest
```

> **Python 3.11+ required.**  
> Core runtime dependencies: `pygame`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `fastapi`, `uvicorn`, `httpx`.

---

## Controls

<!-- Screenshot placeholder: docs/images/gameplay.png -->

| Action | Keys | Notes |
|--------|------|-------|
| Move Left | `A` / `←` | |
| Move Right | `D` / `→` | |
| Jump | `W` / `↑` | |
| Light Attack | `J` / `Z` | Fast, low damage |
| Heavy Attack | `K` / `X` | Slow, high damage; has cooldown |
| Charged Shot | `E` / `Right Ctrl` | Hold to charge, release to fire |
| Dodge Backward | `Space` | Has inter-dodge cooldown |
| Block | `L` / `C` | Hold; has guard meter — repeated blocks can be guard-broken |
| Controls overlay | `H` | Pauses simulation while open |
| Restart match | `R` | End screen only |
| Quit | `Esc` | |
| Hitbox debug | `F1` | Toggles attack hitbox / hurtbox overlay |

---

## Features

### Combat & Gameplay

- Fixed-timestep simulation at 60 ticks/sec with sub-pixel integer physics (`position = pixels × 100`)
- Full FSM per fighter: IDLE → ATTACK_STARTUP → ATTACK_ACTIVE → ATTACK_RECOVERY, DODGING, HITSTUN, BLOCKING, BLOCKSTUN, PARRY_STUNNED, AIRBORNE, LANDING, EXHAUSTED, KO
- **Light attack** — fast startup, low damage, no cooldown
- **Heavy attack** — slow startup, high damage, inter-attack cooldown
- **Dodge** — invincibility window with inter-dodge cooldown; dodge-avoid detection
- **Block** — guard meter drains on absorbed hits; guard break → PARRY_STUNNED
- **Jump** — startup → airborne → landing with gravity physics
- **Charged ranged weapon** — hold to charge (up to 60 ticks); release fires a projectile; damage scales with charge fraction; block chip-damage on impact
- Combo counter with center-screen display, scale animation, and combo ring VFX
- Attack trails, impact rings, floating damage numbers, screen shake, hitstop
- F1 hitbox debug overlay (red = attack hitbox, cyan = hurtbox)

### Adaptive AI

- **T0 Baseline:** weighted random with seeded RNG — deterministic, good for regression baselines
- **T1 Markov-only:** real-time commitment prediction + strategy scoring, no learning
- **T2 Full Adaptive:** Markov/sklearn ensemble + tactical planner + in-session memory
- Planner modes: `exploit`, `bait`, `defend`, `probe`, `punish`
- Player archetypes: `AGGRESSIVE`, `DEFENSIVE`, `PATTERNED`, `EVASIVE`, `BALANCED`
- Shift detection — probes alternate tactics when player patterns change
- Configurable softmax temperature, exploration budget, and staleness thresholds

### Training Pipeline

- Scripted self-play runner (profiles: `RANDOM`, `AGGRESSIVE`, `DEFENSIVE`, `PATTERNED`, `MIXED`)
- Curriculum trainer — allocates self-play matches toward the AI's weakest zones
- Sklearn Random Forest training with configurable holdout split
- Train-and-promote pipeline with exit codes suitable for CI automation

### Evaluation & Regression Gates

- Headless evaluation harness (configurable tier, seed range, match count)
- Frozen JSON baseline snapshots (win rate, prediction accuracy, damage, latency)
- Regression detection with pass/fail exit codes (`0` = pass, `1` = regression, `2` = insufficient data)
- Per-tick performance profiler and repeatable benchmark suite

### Local API & Dashboard

- FastAPI REST API at `http://localhost:8000`
- Single-page dashboard at `http://localhost:8000/ui/`
- Endpoints for match data, training, model registry, evaluation, and explainability reports
- Database path overrideable via `ADAPTIVE_DUELIST_DB` environment variable

### UX & Readability

- Title screen with live AI tier selector
- In-game controls overlay (H key, pauses simulation)
- Persistent combo counter with ×N scale-pop animation (≥3 hit streak)
- Guard meter bar per fighter (teal → orange-red → broken)
- Dodge and heavy attack cooldown indicator pips in the HUD
- FSM-state tinting per fighter state

---

## AI Tier System

| Tier | Name | Behavior |
|------|------|----------|
| T0 | Baseline | Random AI — no planner, no prediction |
| T1 | Markov-only | Prediction + strategy scoring; profile not updated mid-session |
| T2 | Full Adaptive | Markov/sklearn ensemble + memory + session adaptation + shift detection |

Select the tier at the title screen using `←` / `→`, then press **Space** or **Enter**.

---

## Session Adaptation

At T2, the AI adapts its tactical mode selection *across matches within a session* without retraining. After each match, per-mode outcomes are recorded in a `SessionMemory` with exponential decay (configurable via `config/ai_config.yaml`). A player archetype is inferred from the persistent behavioral profile and used to bias mode selection.

**Archetypes:** `AGGRESSIVE`, `DEFENSIVE`, `PATTERNED`, `EVASIVE`, `BALANCED`

Session memory is in-process only — it resets when the game restarts.

```bash
# View current archetype and session stats
python3 scripts/cli.py session-status
```

---

## Local API & Dashboard

```bash
python3 scripts/run_api.py
# Options: --host, --port, --reload (dev mode)
```

<!-- Screenshot placeholder: docs/images/dashboard.png -->

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server health and version |
| GET | `/api/config` | Active game and AI config |
| GET | `/api/stats` | Match counts and active model |
| POST | `/api/matches/self-play` | Generate self-play training data |
| POST | `/api/matches/evaluate` | Run a batch evaluation |
| GET | `/api/matches/recent` | List recent matches |
| GET | `/api/matches/{id}/report` | Full explainability report for a match |
| GET | `/api/training/status` | Retrain threshold and data delta |
| POST | `/api/training/run` | Run training pipeline |
| POST | `/api/training/curriculum` | Run curriculum-driven training cycle |
| GET | `/api/models/status` | Model registry |
| POST | `/api/models/baseline` | Create evaluation baseline |
| POST | `/api/models/check-regression` | Run regression check vs baseline |

---

## CLI Tools

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

Exit codes for `train_and_promote.py`: `0` = promoted / not needed, `1` = regression detected, `2` = insufficient data.

### Curriculum Training

Allocates self-play matches toward the player's weakest areas using a greedy interleaved strategy:

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

# Check for regressions (exit 0 = pass, 1 = fail)
python3 scripts/cli.py check-regression --tier 2 --matches 50
```

Baselines are frozen JSON snapshots committed to `baselines/`. Regression thresholds are configured in `config/eval_config.yaml`.

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

## Project Architecture

```
adaptive-duelist-ai/
├── main.py               Entry point — launches the game
├── config/               YAML configs + typed dataclass loader (GameConfig, AIConfig, DisplayConfig)
├── game/                 Simulation core
│   ├── engine.py         Main loop, match lifecycle, tick orchestration
│   ├── combat/           FSM, state machine, physics, collision, damage, guard, projectiles
│   ├── entities/         Player controller, AI baseline controller, fighter helpers
│   ├── input/            Input actions (Vocab A), input map, keybind registry
│   └── state.py          Authoritative mutable simulation state with phase-lock enforcement
├── ai/                   AI pipeline
│   ├── layers/           Behavior model, prediction engine, tactical planner
│   ├── models/           Markov, sklearn, and ensemble predictors
│   ├── strategy/         Strategy selector, action resolver, planner memory
│   ├── features/         Feature extraction for ML
│   ├── training/         Dataset builder, model trainer, self-play runner, curriculum
│   └── profile/          Player profile persistence, archetype classifier
├── api/                  FastAPI application (routes, schemas, dependency injection)
├── ui/                   Dashboard HTML/JS/CSS (served by FastAPI) + Pygame renderer
├── rendering/            Pygame renderer (particles, rings, trails, HUD, overlays)
├── data/                 SQLite DB wrapper, migrations, event logging, tick snapshots
├── evaluation/           Evaluation harness, regression detector, baseline management
├── replay/               Deterministic replay recorder and verifier
├── analytics/            Post-match explainability, pattern mining, planner metrics
├── baselines/            Committed frozen baseline JSON snapshots
├── scripts/              CLI tools, benchmarks, performance profiler, maintenance scripts
├── tests/                1 033-test suite (unit + integration)
└── docs/                 Screenshots and supplementary documentation
```

**Key design decisions:**
- **Two-vocabulary input model** — `InputAction` (raw key events) never crosses into the simulation; fighters only see `CombatCommitment` (resolved intent). The AI operates entirely in Vocab B.
- **Phase-locked simulation state** — `SimulationState` mutations are only allowed during the `SIMULATE` phase. Debug mode enforces this at runtime via `PhaseLockError`.
- **Headless-capable by default** — pass `headless=True` to `Engine` to run without pygame display. All tests use this path.

---

## Roadmap / Future Ideas

These are directions worth exploring — not promises or committed work:

- **Online learning** — update the Markov model within a session without a full retrain cycle
- **Networked play** — separate the simulation tick loop from the renderer so both sides can run on separate hosts
- **Richer action space** — air attacks, attack cancels, specials; each adds a new commitment type that the prediction layer can model
- **ONNX export** — export the trained RF classifier to ONNX for faster inference or mobile targets
- **Replay UI** — scrubable web-based replay viewer using the existing replay file format and dashboard infrastructure
- **Human benchmark mode** — structured sessions with automated skill assessment across multiple matches

---

## Contributing

1. Run the full test suite before opening a PR: `python3 -m pytest`
2. All new functionality must include unit tests
3. Evaluation gates (`check-regression`) must pass before merging to `main`
4. Session memory and archetype classification are in-process only — do not persist them to the database

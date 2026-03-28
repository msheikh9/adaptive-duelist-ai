# Baselines

Frozen evaluation snapshots used for regression detection.

## Naming Convention

```
baseline_<TIER>_<tag>.json
```

- **TIER**: `t0_baseline`, `t1_markov_only`, `t2_full_adaptive`
- **tag**: optional version or commit label (e.g., `v1.0`, `pre_refactor`)

## Creating a Baseline

```bash
python3 scripts/create_baseline.py --tier 2 --matches 50 --seed 0
python3 scripts/create_baseline.py --tier 0 --matches 100 --tag v1.0
```

## Schema (v1)

```json
{
  "schema_version": 1,
  "created_at": "2025-01-01T00:00:00+00:00",
  "git_sha": "abc1234",
  "config_hash": "sha256_game:sha256_ai",
  "tier": "T2_FULL_ADAPTIVE",
  "match_count": 50,
  "seed_start": 0,
  "win_rate": { "ai_win_rate": 0.72, ... },
  "match_length": { "avg_ticks": 3500, ... },
  "damage": { "avg_hp_differential": 45.2, ... },
  "prediction": { "top1_accuracy": 0.35, ... },
  "planner": { "overall_success_rate": 0.52, ... },
  "performance": { "p95_tick_ms": 0.15, ... },
  "replay_verification": { "pass_rate": 1.0, ... }
}
```

## Updating Baselines

1. Run a fresh evaluation at the same parameters as the existing baseline.
2. If the new results are intentionally different (due to AI improvements), create a new baseline to replace the old one.
3. Commit baseline files to version control — they are the source of truth for regression checks.

## Trust Model

- Baselines are committed artifacts, not auto-generated. A human decides when to update them.
- Regression checks compare current performance against the committed baseline.
- If a regression is intentional (e.g., trading win rate for diversity), update the baseline and document why in the commit message.

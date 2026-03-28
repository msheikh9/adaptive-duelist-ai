"""Canonical evaluation metrics for AI quality assessment.

All metrics return typed dataclasses. Functions accept match results
(from batch evaluation) or database handles for DB-backed metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WinRateMetrics:
    """Win/loss/draw rates across a batch of matches."""

    total_matches: int
    ai_wins: int
    player_wins: int
    draws: int
    ai_win_rate: float
    player_win_rate: float
    draw_rate: float


@dataclass
class MatchLengthMetrics:
    """Match duration statistics."""

    avg_ticks: float
    min_ticks: int
    max_ticks: int
    median_ticks: float


@dataclass
class DamageMetrics:
    """Aggregate damage dealt/taken by the AI."""

    avg_ai_hp_remaining: float
    avg_player_hp_remaining: float
    avg_hp_differential: float  # positive = AI advantage


@dataclass
class PredictionAccuracyMetrics:
    """Prediction quality from ai_decisions table."""

    total_predictions: int
    top1_correct: int
    top1_accuracy: float
    top2_correct: int
    top2_accuracy: float


@dataclass
class PlannerSuccessMetrics:
    """Planner mode outcome rates."""

    total_decisions_with_outcome: int
    mode_success_rates: dict[str, float]
    overall_success_rate: float


@dataclass
class ReplayVerificationMetrics:
    """Replay verification pass rate."""

    total_replays: int
    passed: int
    failed: int
    pass_rate: float


@dataclass
class PerformanceMetrics:
    """Latency / throughput metrics."""

    p95_tick_ms: float
    p95_planner_ms: float
    avg_ticks_per_sec: float


@dataclass
class EvaluationResult:
    """Complete evaluation result combining all metric categories."""

    tier: str
    match_count: int
    seed_start: int
    win_rate: WinRateMetrics
    match_length: MatchLengthMetrics
    damage: DamageMetrics
    prediction: PredictionAccuracyMetrics | None
    planner: PlannerSuccessMetrics | None
    performance: PerformanceMetrics
    replay_verification: ReplayVerificationMetrics | None = None
    raw_results: list[dict] = field(default_factory=list)


def compute_win_rate(results: list[dict]) -> WinRateMetrics:
    """Compute win/loss/draw rates from match result dicts."""
    total = len(results)
    ai = sum(1 for r in results if r["winner"] == "AI")
    player = sum(1 for r in results if r["winner"] == "PLAYER")
    draws = total - ai - player
    return WinRateMetrics(
        total_matches=total,
        ai_wins=ai,
        player_wins=player,
        draws=draws,
        ai_win_rate=ai / total if total else 0.0,
        player_win_rate=player / total if total else 0.0,
        draw_rate=draws / total if total else 0.0,
    )


def compute_match_length(results: list[dict]) -> MatchLengthMetrics:
    """Compute match duration statistics from result dicts."""
    import statistics

    ticks = [r["ticks"] for r in results]
    return MatchLengthMetrics(
        avg_ticks=statistics.mean(ticks) if ticks else 0.0,
        min_ticks=min(ticks) if ticks else 0,
        max_ticks=max(ticks) if ticks else 0,
        median_ticks=statistics.median(ticks) if ticks else 0.0,
    )


def compute_damage(results: list[dict]) -> DamageMetrics:
    """Compute damage metrics from result dicts."""
    import statistics

    ai_hp = [r["ai_hp"] for r in results]
    player_hp = [r["player_hp"] for r in results]
    diffs = [a - p for a, p in zip(ai_hp, player_hp)]
    return DamageMetrics(
        avg_ai_hp_remaining=statistics.mean(ai_hp) if ai_hp else 0.0,
        avg_player_hp_remaining=statistics.mean(player_hp) if player_hp else 0.0,
        avg_hp_differential=statistics.mean(diffs) if diffs else 0.0,
    )


def compute_prediction_accuracy(db, match_ids: list[str]) -> PredictionAccuracyMetrics | None:
    """Compute prediction accuracy by joining ai_decisions with semantic_events.

    Matches each AI prediction to the next player commitment event,
    mirroring the approach in analytics/match_analyzer.py.
    """
    import json

    if not match_ids or not db.table_exists("ai_decisions"):
        return None

    placeholders = ",".join("?" for _ in match_ids)

    decisions = db.fetchall(
        f"SELECT tick_id, predicted_top, pred_probs, match_id "
        f"FROM ai_decisions WHERE match_id IN ({placeholders}) "
        f"AND predicted_top IS NOT NULL "
        f"ORDER BY match_id, tick_id;",
        tuple(match_ids),
    )
    if not decisions:
        return PredictionAccuracyMetrics(
            total_predictions=0, top1_correct=0, top1_accuracy=0.0,
            top2_correct=0, top2_accuracy=0.0,
        )

    commits = db.fetchall(
        f"SELECT tick_id, commitment, match_id "
        f"FROM semantic_events WHERE match_id IN ({placeholders}) "
        f"AND actor = 'player' AND commitment IS NOT NULL "
        f"ORDER BY match_id, tick_id;",
        tuple(match_ids),
    )

    # Group commits by match for efficient lookup
    commits_by_match: dict[str, list] = {}
    for c in commits:
        commits_by_match.setdefault(c["match_id"], []).append(c)

    top1_correct = 0
    top2_correct = 0
    total = 0

    for d in decisions:
        mid = d["match_id"]
        match_commits = commits_by_match.get(mid, [])
        # Find first player commit at or after this decision tick
        actual = None
        for c in match_commits:
            if c["tick_id"] >= d["tick_id"]:
                actual = c["commitment"]
                break
        if actual is None:
            continue

        total += 1
        if d["predicted_top"] == actual:
            top1_correct += 1

        # Top-2 from pred_probs
        top2_labels = [d["predicted_top"]]
        if d["pred_probs"]:
            try:
                probs = json.loads(d["pred_probs"])
                top2_labels = sorted(probs.keys(), key=lambda k: -probs[k])[:2]
            except (json.JSONDecodeError, TypeError):
                pass
        if actual in top2_labels:
            top2_correct += 1

    return PredictionAccuracyMetrics(
        total_predictions=total,
        top1_correct=top1_correct,
        top1_accuracy=top1_correct / total if total else 0.0,
        top2_correct=top2_correct,
        top2_accuracy=top2_correct / total if total else 0.0,
    )


def compute_planner_success(db, match_ids: list[str]) -> PlannerSuccessMetrics | None:
    """Compute planner mode success rates from ai_decisions table."""
    if not match_ids or not db.table_exists("ai_decisions"):
        return None

    placeholders = ",".join("?" for _ in match_ids)
    rows = db.fetchall(
        f"SELECT tactical_mode, outcome FROM ai_decisions "
        f"WHERE match_id IN ({placeholders}) AND outcome IS NOT NULL;",
        tuple(match_ids),
    )

    if not rows:
        return PlannerSuccessMetrics(
            total_decisions_with_outcome=0,
            mode_success_rates={},
            overall_success_rate=0.0,
        )

    total = len(rows)
    successes = sum(1 for r in rows if r["outcome"] == "success")

    mode_counts: dict[str, list] = {}
    for r in rows:
        mode = r["tactical_mode"]
        if mode not in mode_counts:
            mode_counts[mode] = [0, 0]
        mode_counts[mode][1] += 1
        if r["outcome"] == "success":
            mode_counts[mode][0] += 1

    mode_rates = {
        m: s / t if t > 0 else 0.0
        for m, (s, t) in mode_counts.items()
    }

    return PlannerSuccessMetrics(
        total_decisions_with_outcome=total,
        mode_success_rates=mode_rates,
        overall_success_rate=successes / total if total else 0.0,
    )


def compute_performance(results: list[dict]) -> PerformanceMetrics:
    """Compute performance metrics from match result dicts.

    Expects each result to have 'elapsed_s' and 'ticks' keys.
    Uses 'tick_latencies_ms' list for real p95 tick time when available,
    otherwise falls back to per-match average estimates. Same for
    'planner_latencies_ms'.
    """
    total_ticks = sum(r["ticks"] for r in results)
    total_time = sum(r["elapsed_s"] for r in results)
    tps = total_ticks / total_time if total_time > 0 else 0.0

    # Collect real per-tick latencies if available
    all_tick_ms: list[float] = []
    all_planner_ms: list[float] = []
    for r in results:
        if "tick_latencies_ms" in r:
            all_tick_ms.extend(r["tick_latencies_ms"])
        if "planner_latencies_ms" in r:
            all_planner_ms.extend(r["planner_latencies_ms"])

    if all_tick_ms:
        all_tick_ms.sort()
        n = len(all_tick_ms)
        p95_tick = all_tick_ms[min(int(n * 0.95), n - 1)]
    else:
        # Fallback: per-match average tick time
        per_match_ms = [
            (r["elapsed_s"] / r["ticks"] * 1000) if r["ticks"] > 0 else 0.0
            for r in results
        ]
        per_match_ms.sort()
        n = len(per_match_ms)
        p95_tick = per_match_ms[min(int(n * 0.95), n - 1)] if n > 0 else 0.0

    if all_planner_ms:
        all_planner_ms.sort()
        n = len(all_planner_ms)
        p95_planner = all_planner_ms[min(int(n * 0.95), n - 1)]
    else:
        p95_planner = p95_tick  # proxy when no planner data

    return PerformanceMetrics(
        p95_tick_ms=p95_tick,
        p95_planner_ms=p95_planner,
        avg_ticks_per_sec=tps,
    )


def compute_replay_verification(replay_dir) -> ReplayVerificationMetrics | None:
    """Compute replay verification pass rate from a replay directory.

    Returns None if the directory does not exist or has no replays.
    """
    from pathlib import Path

    replay_dir = Path(replay_dir)
    if not replay_dir.exists():
        return None

    files = sorted(replay_dir.glob("*.replay"))
    if not files:
        return None

    try:
        from config.config_loader import load_config
        from replay.replay_player import load_replay, verify_replay
    except ImportError:
        return None

    game_cfg, _, _ = load_config()
    passed = 0
    failed = 0
    for f in files:
        try:
            replay = load_replay(f)
            result = verify_replay(replay, game_cfg)
            if result.passed:
                passed += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    total = passed + failed
    return ReplayVerificationMetrics(
        total_replays=total,
        passed=passed,
        failed=failed,
        pass_rate=passed / total if total > 0 else 0.0,
    )

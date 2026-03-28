"""Train and evaluate a RandomForest classifier for commitment prediction.

Uses chronological train/test split (no shuffle) to prevent cross-match
leakage. Produces a TrainingResult with accuracy metrics and the fitted
model. Persists the model via joblib and registers it in the model_registry
table.

Label space includes HOLD — the model must explicitly predict "player
does not commit within the window".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import numpy as np

from ai.models.base_predictor import ALL_LABELS

if TYPE_CHECKING:
    from data.db import Database

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models"


@dataclass
class TrainingResult:
    """Output of a training run."""
    version: str
    model_path: Path
    accuracy: float
    top2_accuracy: float
    dataset_size: int
    train_size: int
    test_size: int
    elapsed_seconds: float
    label_counts: dict[str, int]


def train_model(
    X: list[list[float]],
    y: list[str],
    *,
    n_estimators: int = 100,
    max_depth: int = 8,
    min_samples_leaf: int = 10,
    holdout_fraction: float = 0.15,
    version: str | None = None,
) -> TrainingResult:
    """Train a RandomForest on (X, y) with chronological split.

    Parameters
    ----------
    X, y:              Feature matrix and label vector (chronological order).
    n_estimators:      Number of trees.
    max_depth:         Maximum tree depth.
    min_samples_leaf:  Minimum samples per leaf.
    holdout_fraction:  Fraction of tail data used for evaluation.
    version:           Model version string. Auto-generated if None.

    Returns
    -------
    TrainingResult with the fitted model saved to disk.
    """
    t0 = time.monotonic()

    n = len(X)
    split_idx = max(1, int(n * (1.0 - holdout_fraction)))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train_np = np.asarray(X_train, dtype=np.float32)
    X_test_np = np.asarray(X_test, dtype=np.float32) if X_test else np.empty((0, X_train_np.shape[1]), dtype=np.float32)
    y_train_np = np.asarray(y_train)
    y_test_np = np.asarray(y_test) if y_test else np.empty(0)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_np, y_train_np)

    # Evaluate on holdout
    if len(X_test) > 0:
        y_pred = clf.predict(X_test_np)
        acc = accuracy_score(y_test_np, y_pred)

        # Top-2 accuracy: need predict_proba
        proba = clf.predict_proba(X_test_np)
        classes = list(clf.classes_)
        k = min(2, len(classes))
        if k >= 2:
            top2_acc = top_k_accuracy_score(
                y_test_np, proba, k=2, labels=classes,
            )
        else:
            top2_acc = acc
    else:
        acc = 0.0
        top2_acc = 0.0

    # Label distribution
    from collections import Counter
    label_counts = dict(Counter(y))

    # Persist model
    if version is None:
        version = f"rf_{int(time.time())}"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{version}.joblib"
    joblib.dump(clf, model_path)

    elapsed = time.monotonic() - t0

    log.info(
        "Model trained: v=%s acc=%.3f top2=%.3f train=%d test=%d (%.1fs)",
        version, acc, top2_acc, len(X_train), len(X_test), elapsed,
    )

    return TrainingResult(
        version=version,
        model_path=model_path,
        accuracy=acc,
        top2_accuracy=top2_acc,
        dataset_size=n,
        train_size=len(X_train),
        test_size=len(X_test),
        elapsed_seconds=elapsed,
        label_counts=label_counts,
    )


def register_model(db: Database, result: TrainingResult) -> None:
    """Insert a TrainingResult into the model_registry table."""
    import json
    db.execute_safe(
        """INSERT INTO model_registry
           (version, model_path, model_type, eval_accuracy, eval_top2_acc,
            dataset_size, is_active, metadata)
           VALUES (?, ?, ?, ?, ?, ?, 1, ?);""",
        (
            result.version,
            str(result.model_path),
            "random_forest",
            result.accuracy,
            result.top2_accuracy,
            result.dataset_size,
            json.dumps({
                "train_size": result.train_size,
                "test_size": result.test_size,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "label_counts": result.label_counts,
            }),
        ),
    )
    # Deactivate previous models
    db.execute_safe(
        "UPDATE model_registry SET is_active = 0 WHERE version != ?;",
        (result.version,),
    )


def load_latest_model(db: Database) -> tuple[object | None, str | None]:
    """Load the active model from registry. Returns (clf, version) or (None, None)."""
    row = db.fetchone(
        "SELECT version, model_path FROM model_registry WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1;"
    )
    if row is None:
        return None, None

    model_path = Path(row["model_path"])
    if not model_path.exists():
        log.warning("Model file not found: %s", model_path)
        return None, None

    clf = joblib.load(model_path)
    return clf, row["version"]

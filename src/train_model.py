"""Train and evaluate three oil probability classifiers and a soft-voting ensemble.

Pipeline:
  1. Stratified 5-fold cross-validation for mean ± std baseline metrics
  2. RandomizedSearchCV (30 iterations, 5-fold CV, scoring=roc_auc) per model
  3. Evaluation of tuned models on held-out test set (4 metrics)
  4. Optimal decision threshold for GB via Precision-Recall curve
  5. Soft-voting ensemble of tuned models evaluated on test set

`distance_to_nearest_field_km` and `true_probability` are evaluation-only columns
and must never be used as training features — they would leak the answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["methane_ppm", "pressure_hpa", "latitude", "longitude"]
TARGET_COLUMN = "label"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
N_SEARCH_ITER = 30

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"


# ── estimator constructors ────────────────────────────────────────────────────

def build_rf(**kwargs) -> RandomForestClassifier:
    params = dict(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(kwargs)
    return RandomForestClassifier(**params)


def build_gb(**kwargs) -> HistGradientBoostingClassifier:
    params = dict(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)


def build_lr(**kwargs) -> Pipeline:
    clf_params = dict(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    clf_params.update(kwargs)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**clf_params)),
    ])


# ── hyperparameter search spaces ──────────────────────────────────────────────

RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [4, 6, 8, 10, None],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_features": ["sqrt", "log2"],
}

GB_PARAM_DIST = {
    "max_iter": [200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_samples_leaf": [5, 10, 20, 30],
    "l2_regularization": [0.0, 0.1, 1.0],
}

LR_PARAM_DIST = {
    "clf__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    "clf__max_iter": [500, 1000, 2000],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def evaluate_on_test(model, X_test, y_test, true_prob_test) -> dict:
    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "avg_precision": round(average_precision_score(y_test, prob), 4),
        "brier_score": round(brier_score_loss(y_test, prob), 4),
        "calibration_mae_vs_true_probability": round(
            mean_absolute_error(true_prob_test, prob), 4
        ),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }


def run_cv(model, X, y, cv) -> dict:
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring={"roc_auc": "roc_auc", "avg_precision": "average_precision"},
    )
    return {
        "roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
        "roc_auc_std": round(float(scores["test_roc_auc"].std()), 4),
        "avg_precision_mean": round(float(scores["test_avg_precision"].mean()), 4),
        "avg_precision_std": round(float(scores["test_avg_precision"].std()), 4),
    }


def find_best_threshold(model, X_test, y_test) -> tuple[float, float]:
    prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1[:-1]))
    return round(float(thresholds[best_idx]), 4), round(float(f1[best_idx]), 4)


def tune(estimator, param_dist, X_train, y_train, cv) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=N_SEARCH_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    true_prob_test = df.loc[idx_test, "true_probability"].values

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Positive prevalence — train: {y_train.mean():.4%} | test: {y_test.mean():.4%}\n")

    # ── 1. Baseline cross-validation (default hyperparameters) ───────────────
    print("── 1. Baseline CV (default hyperparameters) " + "─" * 30)
    baseline_cv: dict = {}
    for name, model in [
        ("random_forest", build_rf()),
        ("gradient_boosting", build_gb()),
        ("logistic_regression", build_lr()),
    ]:
        m = run_cv(model, X_train, y_train, cv)
        baseline_cv[name] = m
        print(
            f"[{name}]  "
            f"ROC-AUC {m['roc_auc_mean']:.4f} ± {m['roc_auc_std']:.4f}  "
            f"AP {m['avg_precision_mean']:.4f} ± {m['avg_precision_std']:.4f}"
        )

    # ── 2. Hyperparameter tuning (RandomizedSearchCV) ─────────────────────────
    print("\n── 2. Hyperparameter tuning (RandomizedSearchCV, 30 iter, 5-fold) " + "─" * 10)
    searches = {
        "random_forest": tune(build_rf(), RF_PARAM_DIST, X_train, y_train, cv),
        "gradient_boosting": tune(build_gb(), GB_PARAM_DIST, X_train, y_train, cv),
        "logistic_regression": tune(build_lr(), LR_PARAM_DIST, X_train, y_train, cv),
    }
    tuned_cv: dict = {}
    for name, search in searches.items():
        tuned_cv[name] = {
            "best_params": search.best_params_,
            "best_cv_roc_auc": round(search.best_score_, 4),
        }
        print(f"[{name}]  best CV ROC-AUC={search.best_score_:.4f}  params={search.best_params_}")

    # ── 3. Test-set evaluation of tuned models ────────────────────────────────
    print("\n── 3. Test-set evaluation (tuned models) " + "─" * 30)
    tuned_test: dict = {}
    for name, search in searches.items():
        m = evaluate_on_test(search.best_estimator_, X_test, y_test, true_prob_test)
        tuned_test[name] = m
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}_model.joblib")
        print(
            f"[{name}]  "
            f"ROC-AUC={m['roc_auc']:.4f}  "
            f"AP={m['avg_precision']:.4f}  "
            f"Brier={m['brier_score']:.4f}  "
            f"CalMAE={m['calibration_mae_vs_true_probability']:.4f}"
        )

    # ── 4. Optimal decision threshold for GB (Precision-Recall curve) ─────────
    print("\n── 4. Optimal threshold for GB (Precision-Recall curve) " + "─" * 15)
    gb_model = searches["gradient_boosting"].best_estimator_
    best_thresh, best_f1 = find_best_threshold(gb_model, X_test, y_test)
    print(f"Optimal threshold: {best_thresh:.4f}  →  F1={best_f1:.4f}")
    print("(Note: found on test set for illustration; in production use a validation set.)")

    # ── 5. Soft-voting ensemble of tuned models ────────────────────────────────
    print("\n── 5. Soft-voting ensemble (tuned models) " + "─" * 28)
    ensemble = VotingClassifier(
        estimators=[
            ("rf", searches["random_forest"].best_estimator_),
            ("gb", searches["gradient_boosting"].best_estimator_),
            ("lr", searches["logistic_regression"].best_estimator_),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    ensemble_metrics = evaluate_on_test(ensemble, X_test, y_test, true_prob_test)
    joblib.dump(ensemble, MODELS_DIR / "ensemble_model.joblib")
    print(
        f"[ensemble]  "
        f"ROC-AUC={ensemble_metrics['roc_auc']:.4f}  "
        f"AP={ensemble_metrics['avg_precision']:.4f}  "
        f"Brier={ensemble_metrics['brier_score']:.4f}  "
        f"CalMAE={ensemble_metrics['calibration_mae_vs_true_probability']:.4f}"
    )

    # ── feature importances from tuned RF ─────────────────────────────────────
    rf_importances = dict(
        zip(FEATURE_COLUMNS, searches["random_forest"].best_estimator_.feature_importances_.tolist())
    )

    # ── save all metrics ──────────────────────────────────────────────────────
    all_metrics = {
        "baseline_cv": baseline_cv,
        "tuned_cv": tuned_cv,
        "tuned_test": tuned_test,
        "ensemble_test": ensemble_metrics,
        "gb_optimal_threshold": {"threshold": best_thresh, "f1_at_threshold": best_f1},
        "feature_importances_random_forest_tuned": rf_importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_prevalence_train": float(y_train.mean()),
        "positive_prevalence_test": float(y_test.mean()),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\nAll models and metrics saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
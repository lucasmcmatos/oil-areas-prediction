"""Train and evaluate three oil probability classifiers and a soft-voting ensemble.

Three models with complementary inductive biases are trained on sensor-observable
features only (methane_ppm, pressure_hpa, latitude, longitude). Individual metrics
are compared, then a soft-voting ensemble combines their probability estimates into
the final model used by the API.

`distance_to_nearest_field_km` and `true_probability` are kept in the dataset for
evaluation purposes only — they must never be used as training features, since they
would leak the answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
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
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["methane_ppm", "pressure_hpa", "latitude", "longitude"]
TARGET_COLUMN = "label"

RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_gb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate(model, X_test, y_test, true_prob_test) -> dict:
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


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    true_prob_test = df.loc[idx_test, "true_probability"].values

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"Positive prevalence — train: {y_train.mean():.4%} | test: {y_test.mean():.4%}\n")

    model_builders = {
        "random_forest": build_rf,
        "gradient_boosting": build_gb,
        "logistic_regression": build_lr,
    }

    fitted_models: dict = {}
    individual_metrics: dict = {}

    for name, builder in model_builders.items():
        model = builder()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, true_prob_test)
        fitted_models[name] = model
        individual_metrics[name] = metrics
        joblib.dump(model, MODELS_DIR / f"{name}_model.joblib")
        print(
            f"[{name}]  ROC-AUC={metrics['roc_auc']:.4f}  "
            f"AP={metrics['avg_precision']:.4f}  "
            f"Brier={metrics['brier_score']:.4f}  "
            f"CalMAE={metrics['calibration_mae_vs_true_probability']:.4f}"
        )

    print("\nTraining soft-voting ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ("rf", build_rf()),
            ("gb", build_gb()),
            ("lr", build_lr()),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    ensemble_metrics = evaluate(ensemble, X_test, y_test, true_prob_test)

    # Ensemble is the main model used by the API.
    joblib.dump(ensemble, MODELS_DIR / "ensemble_model.joblib")
    joblib.dump(ensemble, MODELS_DIR / "oil_probability_model.joblib")

    print(
        f"[ensemble]  ROC-AUC={ensemble_metrics['roc_auc']:.4f}  "
        f"AP={ensemble_metrics['avg_precision']:.4f}  "
        f"Brier={ensemble_metrics['brier_score']:.4f}  "
        f"CalMAE={ensemble_metrics['calibration_mae_vs_true_probability']:.4f}"
    )

    rf_importances = dict(
        zip(FEATURE_COLUMNS, fitted_models["random_forest"].feature_importances_.tolist())
    )

    all_metrics = {
        "individual_models": individual_metrics,
        "ensemble": ensemble_metrics,
        "feature_importances_random_forest": rf_importances,
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
"""Reproducible regression comparison for the worker burnout dataset.

Run: python code.py --data ai_worker_burnout_attrition_2026.csv
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

SEED = 42
TARGET = "burnout_score"
EXCLUDED = ("employee_id", TARGET, "attrition_risk")
DEFAULT_DATA = Path(__file__).resolve().parent / "ai_worker_burnout_attrition_2026.csv"


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    """Apply deterministic cleanup only; do not learn outlier thresholds."""
    df = pd.read_csv(path)
    required = set(EXCLUDED)
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    audit = {"original_rows": len(df), "original_columns": len(df.columns)}
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].str.strip().replace("", np.nan)
    audit["exact_duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    if df["employee_id"].isna().any():
        raise ValueError("Missing employee IDs require an explicit identity policy.")
    if df["employee_id"].duplicated().any():
        raise ValueError("Conflicting/repeated employee IDs require review before splitting.")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    valid = df[TARGET].notna() & np.isfinite(df[TARGET])
    audit["invalid_targets_removed"] = int((~valid).sum())
    df = df.loc[valid].reset_index(drop=True)
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    audit["retained_rows"] = len(df)
    audit["missing_values"] = {c: int(v) for c, v in df.isna().sum().items()}
    audit["outlier_policy"] = "Retain extremes; no IQR-based row removal."
    return df, audit


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=list(EXCLUDED)).copy(), df[TARGET].copy()


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if len(numeric) + len(categorical) != X.shape[1]:
        raise ValueError("Unsupported feature dtype; review the input schema.")
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]), categorical),
    ])


def make_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    estimators = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, n_jobs=1, random_state=SEED),
    }
    return {name: Pipeline([("preprocessor", make_preprocessor(X)), ("model", model)])
            for name, model in estimators.items()}


def regression_metrics(y: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction))}


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, pd.DataFrame, str]:
    """Select using training-only CV; the test set is not accepted here."""
    models = make_models(X_train)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring,
                                n_jobs=1, error_score="raise")
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"] * (-1 if metric != "r2" else 1)
            row[f"cv_{metric}_mean"] = float(values.mean())
            row[f"cv_{metric}_std"] = float(values.std(ddof=1))
        row["cv_fit_seconds_mean"] = float(scores["fit_time"].mean())
        rows.append(row)
    cv_results = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    selected = str(cv_results.iloc[0]["model"])
    for pipeline in models.values():
        pipeline.fit(X_train, y_train)
    return models, cv_results, selected


def evaluate_models(models: dict, X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series) -> tuple[pd.DataFrame, dict]:
    rows, predictions = [], {}
    for name, pipeline in models.items():
        train_scores = regression_metrics(y_train, pipeline.predict(X_train))
        start = perf_counter()
        predictions[name] = pipeline.predict(X_test)
        elapsed = perf_counter() - start
        test_scores = regression_metrics(y_test, predictions[name])
        rows.append({"model": name, **{f"train_{k}": v for k, v in train_scores.items()},
                     **{f"test_{k}": v for k, v in test_scores.items()},
                     "test_predict_seconds": elapsed})
    return pd.DataFrame(rows), predictions


def export_figures(X_train: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                   predictions: dict, selected: str, importance: pd.DataFrame,
                   comparison: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_train, bins=25, color="#3667b0", edgecolor="white")
    ax.set(xlabel="Burnout score", ylabel="Training records", title="Target distribution — training split")
    save(fig, "target_distribution.png")
    fig, ax = plt.subplots(figsize=(10, 7))
    numeric = X_train.select_dtypes(include=[np.number]).assign(burnout_score=y_train)
    corr = numeric.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=65, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.set_title("Numeric correlations — training split")
    fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, "correlations.png")
    names = list(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, name in zip(axes.flat, names):
        ax.scatter(y_test, predictions[name], s=14, alpha=.5, color="#3667b0")
        lower = min(y_test.min(), predictions[name].min())
        upper = max(y_test.max(), predictions[name].max())
        ax.plot([lower, upper], [lower, upper], "--", color="#c64b49")
        ax.set(title=name, xlabel="Actual burnout score", ylabel="Predicted burnout score")
    save(fig, "predictions.png")
    residual = y_test.to_numpy() - predictions[selected]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(predictions[selected], residual, s=16, alpha=.5, color="#3667b0")
    axes[0].axhline(0, color="#c64b49", linestyle="--")
    axes[0].set(xlabel="Predicted score", ylabel="Actual − predicted", title=f"{selected}: test residuals")
    axes[1].hist(residual, bins=20, color="#3667b0", edgecolor="white")
    axes[1].set(xlabel="Residual", ylabel="Records", title="Residual distribution")
    save(fig, "residuals.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    top = importance.nlargest(10, "importance_mean").sort_values("importance_mean")
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3667b0")
    ax.set(xlabel="Decrease in test R² after permutation", title=f"{selected}: permutation importance")
    save(fig, "feature_importance.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = comparison.sort_values("cv_rmse_mean")
    positions = np.arange(len(ordered))
    ax.bar(positions-.18, ordered["cv_rmse_mean"], width=.36, label="Training CV", color="#3667b0")
    ax.bar(positions+.18, ordered["test_rmse"], width=.36, label="Held-out test", color="#d8953e")
    ax.set_xticks(positions, ordered["model"], rotation=15, ha="right")
    ax.set(ylabel="RMSE (lower is better)", title="Model comparison")
    ax.legend()
    save(fig, "model_comparison.png")


def run_experiment(data_path: Path, output_dir: Path) -> dict:
    df, audit = load_data(data_path)
    X, y = make_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
    models, cv_results, selected = compare_models(X_train, y_train)
    metrics, predictions = evaluate_models(models, X_train, X_test, y_train, y_test)
    comparison = cv_results.merge(metrics, on="model", validate="one_to_one")
    perm = permutation_importance(models[selected], X_test, y_test, scoring="r2",
                                  n_repeats=10, random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean,
                               "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_metrics.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    fold_rows = [{"partition": label, "employee_id": df.loc[index, "employee_id"]}
                 for label, indexes in [("train", X_train.index), ("test", X_test.index)]
                 for index in indexes]
    split_bytes = pd.DataFrame(fold_rows).to_csv(index=False).encode()
    metadata = {"seed": SEED, "test_fraction": .2, "train_rows": len(X_train), "test_rows": len(X_test),
                "cv": "5-fold shuffled KFold on training split only; seed 42",
                "selection_metric": "Lowest mean training-CV RMSE", "selected_model": selected,
                "excluded_features": list(EXCLUDED), "features": X.columns.tolist(), "audit": audit,
                "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
                "python": platform.python_version(),
                "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib"]},
                "timing_note": "Single-machine timings; not portable cost benchmarks.",
                "metrics": comparison.to_dict(orient="records")}
    (output_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    export_figures(X_train, y_train, y_test, predictions, selected, importance, comparison, output_dir / "figures")
    print(comparison[["model", "cv_rmse_mean", "cv_r2_mean", "test_mae", "test_rmse", "test_r2"]].to_string(index=False))
    print(f"\nSelected by training CV: {selected}")
    return {"comparison": comparison, "metadata": metadata, "importance": importance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "reports")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}. See data/README.md.")
    run_experiment(args.data, args.output_dir)


if __name__ == "__main__":
    main()

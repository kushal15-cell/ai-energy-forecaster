"""CPU training, untouched external evaluation, and a minimal model audit trail."""

import hashlib
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from final_feature_engineering import CATEGORICAL, FEATURES, NUMERIC, ROOT


def metrics(y, pred):
    actual, estimate = 10.0 ** np.asarray(y), 10.0 ** np.asarray(pred)
    return {
        "n": len(y),
        "r2_log10": float(r2_score(y, pred)) if len(y) > 1 else None,
        "rmse_log10": float(np.sqrt(mean_squared_error(y, pred))),
        "mae_kwh": float(mean_absolute_error(actual, estimate)),
        "r2_kwh": float(r2_score(actual, estimate)) if len(y) > 1 else None,
    }


def main():
    df = pd.read_csv(ROOT / "data/final_training_dataset.csv")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    df.groupby("source")[NUMERIC].agg(lambda s: s.isna().mean()).to_csv(
        ROOT / "reports/missing_feature_rates.csv"
    )
    syn, external = df[df.source == "synthetic"], df[df.source != "synthetic"]
    train, remainder = train_test_split(syn, test_size=0.30, random_state=42)
    calibration, test = train_test_split(remainder, test_size=2 / 3, random_state=42)
    preprocess = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), NUMERIC),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=2,
                    tree_method="hist",
                    device="cpu",
                ),
            ),
        ]
    )
    model.fit(train[FEATURES], train.log_energy_kwh)
    residuals = np.abs(
        calibration.log_energy_kwh - model.predict(calibration[FEATURES])
    )
    q = float(
        np.quantile(
            residuals,
            min(1, np.ceil((len(residuals) + 1) * 0.9) / len(residuals)),
            method="higher",
        )
    )
    rows = []
    predictions = []
    # Controlled audit: identical synthetic split and estimator settings, with duration added.
    # This artifact is evaluation-only and is never used by the application.
    from sklearn.base import clone

    raw = pd.read_csv(ROOT / "data/merged_dataset.csv")
    duration = np.log10(pd.to_numeric(raw.duration_seconds, errors="coerce") + 1)
    audit_train = train[FEATURES].assign(log_duration=train.row_id.map(duration))
    audit_test = test[FEATURES].assign(log_duration=test.row_id.map(duration))
    audit = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            SimpleImputer(strategy="median"),
                            NUMERIC + ["log_duration"],
                        ),
                        (
                            "category",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            CATEGORICAL,
                        ),
                    ]
                ),
            ),
            ("regressor", clone(model["regressor"])),
        ]
    )
    audit.fit(audit_train, train.log_energy_kwh)
    rows.append(
        dict(
            version="duration_ablation",
            evaluation="synthetic_holdout",
            **metrics(test.log_energy_kwh, audit.predict(audit_test)),
        )
    )
    for name, subset in [
        ("synthetic_holdout", test),
        ("external_all", external),
    ] + list(external.groupby("source")):
        pred = model.predict(subset[FEATURES])
        result = metrics(subset.log_energy_kwh, pred)
        result["interval_coverage"] = float(
            np.mean(np.abs(subset.log_energy_kwh - pred) <= q)
        )
        rows.append(dict(version="v2_no_duration", evaluation=name, **result))
        baseline = np.full(len(subset), train.log_energy_kwh.median())
        rows.append(
            dict(
                version="median_baseline",
                evaluation=name,
                **metrics(subset.log_energy_kwh, baseline),
            )
        )
        if name in ["synthetic_holdout", "external_all"]:
            predictions.append(
                pd.DataFrame(
                    {
                        "row_id": subset.row_id,
                        "source": subset.source,
                        "actual_kwh": subset.energy_kwh,
                        "predicted_kwh": 10.0**pred,
                        "evaluation": name,
                    }
                )
            )
    legacy = ROOT / "models/legacy/final_training_dataset.csv"
    if legacy.exists():
        old = pd.read_csv(legacy)
        _, old_test = train_test_split(old, test_size=0.2, random_state=42)
        old_model = joblib.load(ROOT / "models/legacy/xgboost_energy_forecaster.pkl")
        pred = old_model.predict(old_test[old_model.get_booster().feature_names])
        rows.append(
            dict(
                version="v1_leaky_archived",
                evaluation="original_blended_holdout",
                **metrics(old_test.log_energy_kwh, pred),
            )
        )
    split = pd.concat(
        [
            part[["row_id", "source"]].assign(split=name)
            for name, part in [
                ("train", train),
                ("calibration", calibration),
                ("synthetic_test", test),
                ("external_test", external),
            ]
        ]
    )
    split.to_csv(ROOT / "reports/split_manifest.csv", index=False)
    pd.concat(predictions).to_csv(ROOT / "reports/predictions.csv", index=False)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ROOT / "reports/model_comparison.csv", index=False)
    artifact = ROOT / "models/xgboost_energy_forecaster.pkl"
    joblib.dump(model, artifact)
    metadata = {
        "version": "v2_no_duration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "seed": 42,
        "split_counts": split.split.value_counts().to_dict(),
        "source_counts": df.source.value_counts().to_dict(),
        "synthetic_interval": {
            "nominal_coverage": 0.9,
            "log10_radius": q,
            "calibration_n": len(calibration),
        },
        "training_ranges": {
            c: [float(train[c].min()), float(train[c].max())] for c in NUMERIC
        },
        "categories": {c: sorted(train[c].unique().tolist()) for c in CATEGORICAL},
        "dataset_sha256": hashlib.sha256(
            (ROOT / "data/merged_dataset.csv").read_bytes()
        ).hexdigest(),
        "model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (ROOT / "models/metadata.json").write_text(json.dumps(metadata, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

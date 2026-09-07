"""Generate exact TreeSHAP plots on a bounded synthetic test sample."""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import DMatrix

from final_feature_engineering import FEATURES, ROOT


def main():
    df = pd.read_csv(ROOT / "data/final_training_dataset.csv")
    split = pd.read_csv(ROOT / "reports/split_manifest.csv")
    held = df[df.row_id.isin(split[split.split == "synthetic_test"].row_id)].sample(
        n=500, random_state=42
    )
    model = joblib.load(ROOT / "models/xgboost_energy_forecaster.pkl")
    x = model["preprocess"].transform(held[FEATURES])
    names = model["preprocess"].get_feature_names_out()
    values = model["regressor"].get_booster().predict(DMatrix(x), pred_contribs=True)
    explanation = shap.Explanation(
        values=values[:, :-1],
        base_values=values[:, -1],
        data=x,
        feature_names=list(names),
    )
    shap.summary_plot(explanation, x, feature_names=names, show=False, max_display=15)
    plt.savefig(ROOT / "models/shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close("all")
    shap.plots.bar(explanation, show=False, max_display=15)
    plt.savefig(ROOT / "models/shap_bar.png", bbox_inches="tight", dpi=150)
    plt.close("all")
    pd.DataFrame(
        {"feature": names, "mean_absolute_shap": np.abs(values[:, :-1]).mean(axis=0)}
    ).sort_values("mean_absolute_shap", ascending=False).to_csv(
        ROOT / "reports/shap_importance.csv", index=False
    )
    (ROOT / "reports/shap_sample.json").write_text(
        json.dumps(
            {"row_ids": held.row_id.tolist(), "split": "synthetic_test", "seed": 42}
        )
    )
    print("Saved corrected TreeSHAP plots and attribution audit.")


if __name__ == "__main__":
    main()

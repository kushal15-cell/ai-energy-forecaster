import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("data/final_training_dataset.csv")

features = [col for col in df.columns if col not in [
    "energy_kwh",
    "log_energy_kwh"
]]

X = df[features]

model = joblib.load("models/xgboost_energy_forecaster.pkl")

explainer = shap.Explainer(model)
shap_values = explainer(X)

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("models/shap_summary.png", bbox_inches="tight")

plt.figure()
shap.plots.bar(shap_values, show=False)
plt.savefig("models/shap_bar.png", bbox_inches="tight")

print("Saved:")
print("models/shap_summary.png")
print("models/shap_bar.png")
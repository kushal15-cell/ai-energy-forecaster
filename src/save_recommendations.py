from src.final_feature_engineering import ROOT
from src.inference import DEFAULT, scenarios

if __name__ == "__main__":
    scenarios(DEFAULT).to_csv(ROOT / "data/recommendations.csv", index=False)
    print("Saved recomputed default scenarios to data/recommendations.csv")

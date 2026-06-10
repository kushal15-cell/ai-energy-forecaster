import pandas as pd
from recommend_energy_savings import recommend_savings, base_input

base_energy, recommendations = recommend_savings(base_input)

df = pd.DataFrame(recommendations)
df.insert(0, "baseline_energy_kwh", base_energy)

df.to_csv("data/recommendations.csv", index=False)

print("Saved data/recommendations.csv")
print(df)
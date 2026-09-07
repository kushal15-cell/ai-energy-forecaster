"""Model-derived scenario comparisons; no fixed savings multipliers."""

from src.inference import DEFAULT, forecast, scenarios


def recommend_savings(config):
    return forecast(config, False)["energy_kwh"], scenarios(config).to_dict(
        orient="records"
    )


if __name__ == "__main__":
    print(scenarios(DEFAULT).to_string(index=False))

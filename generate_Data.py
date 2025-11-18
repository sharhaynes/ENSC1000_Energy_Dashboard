import pandas as pd
import numpy as np


def generate_data():
    buildings = [
        "Sydney Martin Library",
        "FST Building",
        "Sir Philip Sherlock Hall",
        "Admin Building"
    ]

    commodities = ["Electricity", "Water", "Gas"]
    cost_per_unit = {"Electricity": 0.25, "Water": 0.01, "Gas": 0.08}

    # data range of 5 years
    date_range = pd.date_range(start="1/1/2020", periods=60, freq="M")

    rows = []

    for building in buildings:
        for commodity in commodities:

            # Base usage (randomized for each building/commodity)
            base_usage = {
                "Electricity": np.random.uniform(3500, 5000),
                "Water": np.random.uniform(1000, 2500),
                "Gas": np.random.uniform(800, 1800)
            }[commodity]

            for d in date_range:
                month = d.month

                # ----- Barbados/UWI academic calendar seasonality -----

                # Semester 2 (Jan–Apr): moderately high usage
                if month in [1, 2, 3, 4]:
                    season_factor = 1.15

                # Summer (May–July): very low usage
                elif month in [5, 6, 7]:
                    season_factor = 0.65

                # Semester 1 (Aug–Dec): highest usage
                else:  # 8,9,10,11,12
                    season_factor = 1.35

                # Occasional extreme spike (equipment failure, AC issues, water leak)
                anomaly_factor = np.random.choice(
                    [1, 1.4, 1.8],
                    p=[0.92, 0.06, 0.02]
                )

                # total usage
                total_usage = (
                    base_usage * season_factor * anomaly_factor
                )

                cost = total_usage * cost_per_unit[commodity] * np.random.uniform(0.85, 1.15)

                rows.append({
                    "Date": d,
                    "Building": building,
                    "Commodity": commodity,
                    "Usage": round(total_usage, 2),
                    "Cost": round(cost, 2)
                })

    df = pd.DataFrame(rows)
    df.to_csv("uwi_energy_seasonal.csv", index=False)
    print("✔ Saved as uwi_energy_seasonal.csv")
    return df

df = generate_data()
df.head()

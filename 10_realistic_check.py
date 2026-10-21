import numpy as np
import pandas as pd
from core import simulate

df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
solar = df["solar_kwh"].to_numpy(float)
demand = df["demand_kwh"].to_numpy(float)
hours = df.index.hour.to_numpy()
n, W = len(df), 168

week_starts = np.arange(0, n - W + 1, W)
test_weeks = week_starts[3::4]                       # same 13 held-out weeks as before

# typical-day demand profile, learned from TRAINING weeks only
train_mask = np.ones(n, dtype=bool)
for s in test_weeks:
    train_mask[s:s + W] = False
profile = pd.Series(demand[train_mask]).groupby(hours[train_mask]).mean().to_numpy()   # 24 values
typical = profile[hours]                              # forecast array aligned with the dataset

configs = {
    "No battery":               dict(policy="none"),
    "Self-use rule":            dict(policy="selfuse"),
    "MPC (perfect demand)":     dict(policy="mpc"),
    "MPC (typical-day demand)": dict(policy="mpc", demand_forecast=typical),
}

print("Running (about a minute)...")
results = {name: np.array([simulate(solar, demand, hours, start=int(s), length=W, **kw)["cost"].sum()
                           for s in test_weeks])
           for name, kw in configs.items()}

base, su = results["No battery"].mean(), results["Self-use rule"].mean()
print("\n=== Average weekly cost on HELD-OUT weeks (PKR) ===")
for name, c in results.items():
    print(f"{name:25s}: {c.mean():7.0f} | saving vs no battery: {100*(base-c.mean())/base:5.1f}% "
          f"| vs self-use: {100*(su-c.mean())/su:+5.1f}% | cheaper than self-use in "
          f"{int((c < results['Self-use rule']).sum())}/{len(c)} weeks")
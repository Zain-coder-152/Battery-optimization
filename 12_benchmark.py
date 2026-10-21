"""Phase 5 + 7: score every controller on the same 13 held-out weeks, add the load-shedding and
battery-wear scenarios, log each result to MLflow, and save data/processed/benchmark.json for the dashboard."""
import json
import time

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from core import simulate, tou_prices, outage_mask, oracle_cost
from env.battery_env import BatteryEnv
from env.wrappers import RuleResidualWrapper
from agent import LearnedController

df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
solar = df["solar_kwh"].to_numpy(float)
demand = df["demand_kwh"].to_numpy(float)
hours = df.index.hour.to_numpy()
n, W = len(df), 168
buy, sell = tou_prices(hours)

# ---------- same week-interleaved split as before ----------
week_starts = np.arange(0, n - W + 1, W)
test_weeks = week_starts[3::4]
train_mask = np.ones(n, dtype=bool)
for s in test_weeks:
    train_mask[s:s + W] = False
typical = pd.Series(demand[train_mask]).groupby(hours[train_mask]).mean().to_numpy()[hours]

OUTAGE_HOURS = (8, 9, 19, 20)      # 4 h/day of scheduled load-shedding, incl. two evening-peak hours
DEG_COST = 3.0                     # PKR per kWh through the battery (~LFP pack price / lifetime throughput)


def weekly(policy, **kw):
    return [simulate(solar, demand, hours, policy, int(s), W, **kw) for s in test_weeks]


def sb3_weeks(model):
    env = RuleResidualWrapper(BatteryEnv(df, tariff="pk_tou", use_forecast=True, random_start=False), scale=0.5)
    out = []
    for s in test_weeks:
        obs, _ = env.reset(options={"start": int(s)})
        total, done = 0.0, False
        while not done:
            obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            total += info["cost"]
            done = term or trunc
        out.append(total)
    return np.array(out)


results = {}      # scenario -> name -> dict of per-week arrays


def add(scenario, name, frames=None, costs=None, family="rule"):
    if frames is not None:
        r = {"cost": [float(f.cost.sum() + f.degradation.sum()) for f in frames],
             "unserved_kwh": [float(f.unserved.sum()) for f in frames],
             "cycles": [float(f.battery_flow.abs().sum() / 2 / 10.0) for f in frames]}
    else:
        r = {"cost": [float(c) for c in costs]}
    r["family"] = family
    results.setdefault(scenario, {})[name] = r
    print(f"  [{scenario}] {name:28s} {np.mean(r['cost']):8.0f} PKR/week")


t0 = time.time()
print("Base scenario (NEPRA TOU, no outages)...")
add("base", "No battery", weekly("none"), family="baseline")
add("base", "Self-use rule", weekly("selfuse"), family="baseline")
add("base", "Peak-aware rule", weekly("peakaware"), family="baseline")
add("base", "PPO (residual, RL)", costs=sb3_weeks(PPO.load("models/ppo_selfuse_fc_v5")), family="rl")
add("base", "SAC (residual, RL)", costs=sb3_weeks(SAC.load("models/sac_selfuse_fc_v5")), family="rl")
add("base", "Learned (imitates MPC)", weekly(LearnedController(solar, demand, hours)), family="learned")
add("base", "MPC (perfect forecast)", weekly("mpc"), family="mpc")
add("base", "MPC (typical-day demand)", weekly("mpc", demand_forecast=typical), family="mpc")
add("base", "Oracle (ceiling)", costs=[oracle_cost(solar[s:s + W], demand[s:s + W], buy[s:s + W], sell[s:s + W])
                                       for s in test_weeks], family="oracle")

print("Load-shedding scenario...")
out = outage_mask(hours, OUTAGE_HOURS)
add("outage", "No battery", weekly("none", outage=out), family="baseline")
add("outage", "Self-use rule", weekly("selfuse", outage=out), family="baseline")
add("outage", "Peak-aware rule", weekly("peakaware", outage=out), family="baseline")
add("outage", "MPC (typical-day demand)", weekly("mpc", outage=out, demand_forecast=typical), family="mpc")

print("Battery-wear scenario...")
add("wear", "Self-use rule", weekly("selfuse", deg_cost=DEG_COST), family="baseline")
add("wear", "MPC ignoring wear", weekly("mpc", demand_forecast=typical), family="mpc")
add("wear", "MPC wear-aware", weekly("mpc", deg_cost=DEG_COST, demand_forecast=typical), family="mpc")
# "ignoring wear" was planned without the cost, so charge its wear afterwards for a fair comparison
r = results["wear"]["MPC ignoring wear"]
r["cost"] = [c + DEG_COST * cyc * 2 * 10.0 for c, cyc in zip(r["cost"], r["cycles"])]

# ---------- summaries + MLflow ----------
mlflow.set_experiment("battery-benchmark")
summary = {}
for scenario, rows in results.items():
    ref = np.array(rows["Self-use rule"]["cost"])
    base = np.array(rows["No battery"]["cost"]) if "No battery" in rows else None
    summary[scenario] = []
    for name, r in rows.items():
        c = np.array(r["cost"])
        m = {"mean_weekly_cost_pkr": float(c.mean()),
             "vs_self_use_pct": float(100 * (ref.mean() - c.mean()) / ref.mean()),
             "weeks_cheaper_than_self_use": int((c < ref - 1e-6).sum())}
        if base is not None:
            m["saving_vs_no_battery_pct"] = float(100 * (base.mean() - c.mean()) / base.mean())
        if "unserved_kwh" in r:
            m["unserved_kwh_per_week"] = float(np.mean(r["unserved_kwh"]))
            m["battery_cycles_per_week"] = float(np.mean(r["cycles"]))
        summary[scenario].append({"name": name, "family": r["family"], **m, "weekly": r["cost"]})
        with mlflow.start_run(run_name=f"{scenario} | {name}"):
            mlflow.log_params({"scenario": scenario, "controller": name, "family": r["family"],
                               "test_weeks": len(test_weeks), "battery_kwh": 10, "max_power_kw": 3,
                               "outage_hours": str(OUTAGE_HOURS) if scenario == "outage" else "none",
                               "deg_cost_pkr_per_kwh": DEG_COST if scenario == "wear" else 0})
            mlflow.log_metrics(m)

payload = {
    "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "test_weeks": [df.index[s].strftime("%d %b") for s in test_weeks],
    "outage_hours": list(OUTAGE_HOURS), "deg_cost": DEG_COST,
    "scenarios": summary,
}
with open("data/processed/benchmark.json", "w") as f:
    json.dump(payload, f, indent=1)
print(f"\nSaved data/processed/benchmark.json and {sum(len(v) for v in summary.values())} MLflow runs "
      f"in {time.time() - t0:.0f}s. View them with:  mlflow ui")
for scenario, rows in summary.items():
    print(f"\n=== {scenario} ===")
    for r in rows:
        extra = f" | unserved {r['unserved_kwh_per_week']:5.1f} kWh" if "unserved_kwh_per_week" in r else ""
        print(f"{r['name']:28s}: {r['mean_weekly_cost_pkr']:7.0f} PKR/week | vs self-use {r['vs_self_use_pct']:+5.1f}% "
              f"| cheaper in {r['weeks_cheaper_than_self_use']:2d}/13{extra}")

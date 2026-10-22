"""Make the optimizer robust to forecast errors with a risk margin.

With real (imperfect) forecasts, the optimizer's thin grid-arbitrage margin (~7.7 PKR/kWh) gets eaten by
mistakes that cost ~23 PKR/kWh (energy bought at night, then exported at 11 PKR when the forecast was wrong).
A risk margin is a planning-only cost per kWh moved through the battery: storing solar (worth ~23 PKR/kWh)
still pays, marginal trades do not. It is never billed - reported costs are the real grid bill.

The margin is tuned on the TUNING homes (odd house numbers, every 4th week) and reported on the
held-out homes (even house numbers, every 2nd complete week), for both forecasts.

    python 20_robust_mpc.py
Results: data/processed/lums_benchmark_robust.parquet, data/processed/lums_robust_summary.json"""
import importlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from core import simulate, tou_prices

bench = importlib.import_module("17_benchmark_homes")
OUT = Path("data/processed")
MARGINS = [0.0, 3.0, 6.0]


def run_home(args):
    house, city, ts, demand_raw, spk, F_dem, F_sol, margins, week_step = args
    hours = ts.hour.to_numpy()
    demand = bench.fill_gaps(demand_raw, (ts.dayofweek * 24 + ts.hour).to_numpy())
    panel_kw, cap, pmax = bench.size_system(demand_raw, spk)
    solar = spk * panel_kw
    buy, sell = tou_prices(hours)
    s_p, d_p = bench.persistence(solar), bench.persistence(demand)
    forecasts = {"lgbm": lambda t, e: (F_sol[t, :e - t] * panel_kw, F_dem[t, :e - t]),
                 "persistence": lambda t, e: (s_p(t, e), d_p(t, e))}
    rows = []
    for w in range(1, len(ts) // bench.W, week_step):
        s = w * bench.W
        if np.isnan(demand_raw[s:s + bench.W]).any() or s + bench.W + bench.H > len(ts):
            continue
        rows.append({"house": house, "city": city, "week": w, "forecast": "-", "margin": np.nan, "policy": "selfuse",
                     "cost": simulate(solar, demand, hours, "selfuse", s, bench.W, cap=cap, pmax=pmax, buy=buy, sell=sell).cost.sum()})
        rows.append({"house": house, "city": city, "week": w, "forecast": "-", "margin": np.nan, "policy": "none",
                     "cost": simulate(solar, demand, hours, "none", s, bench.W, cap=cap, pmax=pmax, buy=buy, sell=sell).cost.sum()})
        for fname, f in forecasts.items():
            for m in margins:
                r = simulate(solar, demand, hours, "mpc", s, bench.W, cap=cap, pmax=pmax, buy=buy, sell=sell,
                             horizon=bench.H, forecaster=f, deg_cost=m)
                rows.append({"house": house, "city": city, "week": w, "forecast": fname, "margin": m,
                             "policy": "mpc", "cost": r.cost.sum()})        # r.cost is the grid bill only
    return rows


def jobs_for(houses, margins, week_step):
    homes = pd.read_parquet(OUT / "lums_homes.parquet")
    cities = pd.read_parquet(OUT / "lums_cities.parquet")
    fc = np.load(OUT / "lums_forecasts.npz")
    jobs = []
    for (house, city), g in homes[homes.house.isin(houses)].groupby(["house", "city"]):
        ts = pd.DatetimeIndex(g.timestamp)
        spk = cities[cities.city == city].set_index("timestamp").solar_kwh_per_kw.reindex(ts).to_numpy()
        jobs.append((house, city, ts, g.demand_kwh.to_numpy(float), spk, fc[f"demand_{house}"], fc[f"solar_{city}"],
                     margins, week_step))
    return jobs


def run(jobs, label):
    t0, rows = time.time(), []
    with Pool(8) as pool:
        for i, r in enumerate(pool.imap_unordered(run_home, jobs), 1):
            rows += r
            print(f"  [{label}] {i}/{len(jobs)} homes ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def gains(df):
    """Per-home gain over the self-use rule as % of the no-battery bill, for each forecast/margin."""
    base = df[df.policy != "mpc"].pivot_table(index="house", columns="policy", values="cost", aggfunc="sum")
    m = df[df.policy == "mpc"].pivot_table(index="house", columns=["forecast", "margin"], values="cost", aggfunc="sum")
    return m.rsub(base["selfuse"], axis=0).div(base["none"].abs(), axis=0) * 100, base


def main():
    houses = sorted(pd.read_csv(OUT / "lums_homes.csv").house)
    tune_h, test_h = [h for h in houses if h % 2], [h for h in houses if h % 2 == 0]

    print(f"Tuning the risk margin on {len(tune_h)} homes (every 4th week)...")
    g_tune, _ = gains(run(jobs_for(tune_h, MARGINS, 4), "tune"))
    med = g_tune.median().unstack(0)
    print("\nMedian gain over self-use rule (% of no-battery bill), tuning homes:\n" + med.round(2).to_string())
    best = {f: float(med[f].idxmax()) for f in med.columns}
    print(f"Chosen margins: {best}")

    print(f"\nEvaluating on {len(test_h)} held-out homes (all complete weeks)...")
    margins = sorted(set(best.values()) | {0.0})
    df = run(jobs_for(test_h, margins, 2), "test")
    df.to_parquet(OUT / "lums_benchmark_robust.parquet", index=False)
    g, base = gains(df)
    weeks = df[df.policy == "none"].groupby("house").size()
    g = g[weeks.reindex(g.index) >= 4]
    summary = {"tuning_homes": len(tune_h), "test_homes": int(len(g)), "chosen_margin": best, "results": {}}
    print(f"\nHeld-out homes ({len(g)} with >= 4 test weeks): gain over the self-use rule")
    for f in ("persistence", "lgbm"):
        for m in sorted({0.0, best[f]}):
            x = g[(f, m)].to_numpy()
            lo, hi = bench.bootstrap_ci(x, stat=np.median)
            pkr = ((base["selfuse"] - df[(df.policy == "mpc") & (df.forecast == f) & (df.margin == m)]
                    .groupby("house").cost.sum()) / weeks * 52).reindex(g.index)
            wins = int((x > 0).sum())
            key = f"{f}, margin {m:g}"
            print(f"  {key:26s} median {np.median(x):+.2f}% of bill [95% CI {lo:+.2f}, {hi:+.2f}]  "
                  f"{pkr.median():+,.0f} PKR/yr  beats rule in {wins}/{len(x)} homes")
            summary["results"][key] = {"forecast": f, "margin": m, "median_pct_of_bill": round(float(np.median(x)), 2),
                                       "ci95": [round(lo, 2), round(hi, 2)], "pkr_per_year_median": round(pkr.median()),
                                       "homes_beating_rule": wins}
    (OUT / "lums_robust_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Saved {OUT / 'lums_robust_summary.json'}")


if __name__ == "__main__":
    main()

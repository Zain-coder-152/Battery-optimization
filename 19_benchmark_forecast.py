"""Run the optimizer with the LightGBM forecasts (18_forecast.py) on the same home-weeks as
17_benchmark_homes.py, then print the combined report.

    python 19_benchmark_forecast.py
Results: data/processed/lums_benchmark_forecast.parquet"""
import importlib
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from core import simulate, tou_prices

bench = importlib.import_module("17_benchmark_homes")
OUT = Path("data/processed")


def run_home(args):
    house, city, ts, demand_raw, solar_per_kw, F_dem, F_sol = args
    hours = ts.hour.to_numpy()
    demand = bench.fill_gaps(demand_raw, (ts.dayofweek * 24 + ts.hour).to_numpy())
    panel_kw, cap, pmax = bench.size_system(demand_raw, solar_per_kw)
    solar = solar_per_kw * panel_kw
    buy, sell = tou_prices(hours)
    forecast = lambda t, e: (F_sol[t, :e - t] * panel_kw, F_dem[t, :e - t])
    rows = []
    for w in range(1, len(ts) // bench.W):
        s = w * bench.W
        if np.isnan(demand_raw[s:s + bench.W]).any() or s + bench.W + bench.H > len(ts):
            continue
        r = simulate(solar, demand, hours, "mpc", s, bench.W, cap=cap, pmax=pmax, buy=buy, sell=sell,
                     horizon=bench.H, forecaster=forecast)
        rows.append({"house": house, "city": city, "week": w, "policy": "mpc_lgbm", "cost": r.cost.sum(),
                     "import_kwh": r.grid_import.sum(), "export_kwh": r.grid_export.sum()})
    return rows


def main():
    homes = pd.read_parquet(OUT / "lums_homes.parquet")
    cities = pd.read_parquet(OUT / "lums_cities.parquet")
    fc = np.load(OUT / "lums_forecasts.npz")
    jobs = []
    for (house, city), g in homes.groupby(["house", "city"]):
        ts = pd.DatetimeIndex(g.timestamp)
        spk = cities[cities.city == city].set_index("timestamp").solar_kwh_per_kw.reindex(ts).to_numpy()
        jobs.append((house, city, ts, g.demand_kwh.to_numpy(float), spk, fc[f"demand_{house}"], fc[f"solar_{city}"]))
    t0, rows = time.time(), []
    with Pool(8) as pool:
        for i, r in enumerate(pool.imap_unordered(run_home, jobs), 1):
            rows += r
            print(f"  {i}/{len(jobs)} homes done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_parquet(OUT / "lums_benchmark_forecast.parquet", index=False)
    bench.report(pd.read_parquet(bench.RESULTS))


if __name__ == "__main__":
    main()

"""Benchmark the controllers on real Pakistani homes (LUMS REWD-P, built by 16_build_lums_dataset.py).

Every home gets a system sized the way installers size them: panels to cover the home's yearly use,
battery to about half a day's use. Every week in which all hours of the home's meter data exist is
simulated with each controller, NEPRA time-of-use prices, battery starting half full.

Controllers: no battery, self-use rule, peak-aware rule, optimizer (MPC) with a perfect 48 h forecast,
optimizer with a persistence forecast ("the next 48 h repeat the last 24 h", past data only), oracle.

    python 17_benchmark_homes.py            # simulate (about 30 min on 8 cores) and report
    python 17_benchmark_homes.py --report   # report from the saved results only
Results: data/processed/lums_benchmark.parquet (per home-week) and data/processed/lums_benchmark_summary.json."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from core import simulate, tou_prices, oracle_cost

OUT = Path("data/processed")
RESULTS = OUT / "lums_benchmark.parquet"
W, H = 168, 48
EFF = 0.95


def size_system(demand, solar_per_kw):
    """Panels cover yearly use; battery ~ half a day's use; inverter can empty it in ~2 h."""
    yearly = np.nanmean(demand) * 8760
    panel_kw = float(np.clip(np.round(2 * yearly / solar_per_kw.sum()) / 2, 1.0, 15.0))
    cap = float(np.clip(np.round(0.5 * np.nanmean(demand) * 24), 3.0, 20.0))
    return panel_kw, cap, float(np.clip(cap / 2, 1.5, 10.0))


def fill_gaps(demand, hours_of_week):
    """Missing hours -> that home's median for the same hour of the week (only used for look-ahead/forecasts)."""
    s = pd.Series(demand)
    med = s.groupby(hours_of_week).transform("median")
    return s.fillna(med).fillna(s.median()).to_numpy()


def persistence(series):
    """Forecast for hour t+k issued at t: the value one or two days earlier (never in the future)."""
    def f(t, e):
        k = np.arange(e - t)
        lag = 24 * np.ceil(np.maximum(k, 1) / 24).astype(int)
        return series[t + k - lag]
    return f


def run_home(args):
    house, city, ts, demand_raw, solar_per_kw = args
    hours = ts.hour.to_numpy()
    how = (ts.dayofweek * 24 + ts.hour).to_numpy()
    demand = fill_gaps(demand_raw, how)
    panel_kw, cap, pmax = size_system(demand_raw, solar_per_kw)
    solar = solar_per_kw * panel_kw
    buy, sell = tou_prices(hours)
    s_pers, d_pers = persistence(solar), persistence(demand)
    forecast = lambda t, e: (s_pers(t, e), d_pers(t, e))
    kw = dict(cap=cap, pmax=pmax, buy=buy, sell=sell, horizon=H)
    rows = []
    for w in range(1, len(ts) // W):                       # week 0 has no history for the forecast
        s = w * W
        if np.isnan(demand_raw[s:s + W]).any() or s + W + H > len(ts):
            continue
        base = {"house": house, "city": city, "week": w, "panel_kw": panel_kw, "battery_kwh": cap,
                "demand_kwh": demand[s:s + W].sum(), "solar_kwh": solar[s:s + W].sum()}
        for name, pol, extra in [("none", "none", {}), ("selfuse", "selfuse", {}), ("peakaware", "peakaware", {}),
                                 ("mpc_perfect", "mpc", {}), ("mpc_persistence", "mpc", {"forecaster": forecast})]:
            r = simulate(solar, demand, hours, pol, s, W, **kw, **extra)
            rows.append({**base, "policy": name, "cost": r.cost.sum(),
                         "import_kwh": r.grid_import.sum(), "export_kwh": r.grid_export.sum()})
        rows.append({**base, "policy": "oracle", "cost": oracle_cost(solar[s:s + W], demand[s:s + W], buy[s:s + W],
                                                                      sell[s:s + W], cap=cap, pmax=pmax)})
    return rows


def bootstrap_ci(x, n=5000, seed=0, stat=np.mean):
    """95% interval of `stat` over homes, resampling homes with replacement."""
    rng = np.random.default_rng(seed)
    draws = stat(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return np.percentile(draws, [2.5, 97.5])


def report(df):
    extra = OUT / "lums_benchmark_forecast.parquet"            # optimizer with LightGBM forecasts (19_*.py)
    if extra.exists():
        df = pd.concat([df, pd.read_parquet(extra)], ignore_index=True)
    homes = pd.read_csv(OUT / "lums_homes.csv").set_index("house")
    per_home = df.pivot_table(index=["house", "city"], columns="policy", values="cost", aggfunc="sum")
    weeks = df[df.policy == "none"].groupby("house").size()
    per_home = per_home.join(weeks.rename("weeks"), on="house")
    per_home = per_home[per_home.weeks >= 8]                       # enough data to say anything
    names = {"none": "No battery", "selfuse": "Self-use rule", "peakaware": "Peak-aware rule",
             "mpc_persistence": "Optimizer, persistence forecast", "mpc_lgbm": "Optimizer, LightGBM forecast",
             "mpc_perfect": "Optimizer, perfect forecast", "oracle": "Oracle (ceiling)"}
    names = {p: label for p, label in names.items() if p in per_home.columns}
    print(f"\n{len(per_home)} homes with >= 8 complete weeks, {int(per_home.weeks.sum())} home-weeks in total")
    # Gains over the self-use rule are measured in PKR per year and as a share of the home's NO-BATTERY bill:
    # dividing by the self-use bill is unstable because solar covers almost all of some homes' bills.
    bill = per_home["none"].abs()
    print(f"\n{'':32s} {'saves vs no battery':>20s} | {'extra saving vs self-use rule, median home [95% CI]':>52s} | beats rule")
    summary = {"homes": int(len(per_home)), "home_weeks": int(per_home.weeks.sum()), "policies": {}}
    for p, label in names.items():
        vs_none = 100 * (per_home["none"] - per_home[p]) / bill
        gain_pts = 100 * (per_home["selfuse"] - per_home[p]) / bill
        pkr_year = (per_home["selfuse"] - per_home[p]) / per_home.weeks * 52
        lo, hi = bootstrap_ci(gain_pts.to_numpy(), stat=np.median)
        plo, phi = bootstrap_ci(pkr_year.to_numpy(), stat=np.median)
        wins = int((per_home[p] < per_home["selfuse"] - 1e-6).sum())
        print(f"{label:32s} {vs_none.median():19.1f}% | {gain_pts.median():+5.2f}% of bill [{lo:+.2f}, {hi:+.2f}]  "
              f"{pkr_year.median():+7,.0f} PKR/yr [{plo:+,.0f}, {phi:+,.0f}] | {wins:2d}/{len(per_home)}")
        summary["policies"][p] = {"label": label, "saving_vs_none_median_pct": round(vs_none.median(), 1),
                                  "gain_vs_selfuse_pct_of_bill_median": round(gain_pts.median(), 2),
                                  "gain_pct_ci95": [round(lo, 2), round(hi, 2)],
                                  "gain_pkr_per_year_median": round(pkr_year.median()),
                                  "gain_pkr_ci95": [round(plo), round(phi)], "homes_beating_rule": wins}

    best = "mpc_lgbm" if "mpc_lgbm" in per_home else "mpc_persistence"
    gain = 100 * (per_home["selfuse"] - per_home[best]) / bill
    by_city = gain.groupby(level="city").agg(["median", "count"]).round(2)
    print(f"\n{names[best]} vs self-use rule by city (% of no-battery bill):\n" + by_city.to_string())
    summary["by_city"] = {c: {"median_pct_of_bill": float(r["median"]), "homes": int(r["count"])} for c, r in by_city.iterrows()}

    headroom = (per_home["selfuse"] - per_home["oracle"]).sum()
    captured = {p: round(100 * (per_home["selfuse"] - per_home[p]).sum() / headroom)
                for p in names if p.startswith("mpc")}
    print(f"\nHeadroom over the rule (oracle): median {100 * ((per_home['selfuse'] - per_home['oracle']) / bill).median():.1f}% "
          f"of the no-battery bill. Share of it captured, all homes together: {captured}")
    ac = homes["ac_share_pct"].reindex(per_home.index.get_level_values("house"))
    ok = ac.notna().to_numpy()
    rho = pd.Series(gain.to_numpy()[ok]).corr(pd.Series(ac.to_numpy()[ok]), method="spearman")
    print(f"Spearman correlation of the gain with the home's AC share: {rho:.2f} ({ok.sum()} homes with an AC meter)")
    summary.update({"best_forecast_policy": best, "captured_pct_of_headroom": captured,
                    "gain_vs_ac_share_spearman": round(rho, 2)})
    (OUT / "lums_benchmark_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nSaved {OUT / 'lums_benchmark_summary.json'}")


def main():
    if "--report" not in sys.argv:
        homes = pd.read_parquet(OUT / "lums_homes.parquet")
        cities = pd.read_parquet(OUT / "lums_cities.parquet")
        jobs = []
        for (house, city), g in homes.groupby(["house", "city"]):
            ts = pd.DatetimeIndex(g.timestamp)
            spk = cities[cities.city == city].set_index("timestamp").solar_kwh_per_kw.reindex(ts).to_numpy()
            jobs.append((house, city, ts, g.demand_kwh.to_numpy(float), spk))
        t0 = time.time()
        rows = []
        with Pool(8) as pool:
            for i, r in enumerate(pool.imap_unordered(run_home, jobs), 1):
                rows += r
                print(f"  {i}/{len(jobs)} homes done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_parquet(RESULTS, index=False)
    report(pd.read_parquet(RESULTS))


if __name__ == "__main__":
    main()

"""Phase 3: forecast the next 48 h of household demand and solar output with LightGBM, then
check whether better forecasts make the optimizer save more (19_benchmark_forecast.py).

Every forecast is out-of-sample:
  demand: one model for all homes, cross-fitted over 5 folds of HOMES (a home is never in its own training set)
  solar:  one model for all cities, cross-fitted over 6 two-month blocks (a month is never in its own training set)
Inputs only use what is known when the forecast is issued at hour t: the home's past usage, clear-sky solar
(computable in advance) and a weather forecast. We only have measured weather, so the "forecast" is the measured
value plus noise of typical 1-2 day forecast error (temperature +-1.5 C, humidity +-8 points).

Output: data/processed/lums_forecasts.npz with demand[house] and solar[city] arrays of shape (hours, 48)
where row t, column k is the forecast for hour t+k issued at hour t (column 0 = measured)."""
import importlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pvlib

from lums import COORDS, solar_per_kw

bench = importlib.import_module("17_benchmark_homes")
OUT = Path("data/processed")
H, STRIDE = 48, 6                     # horizon; train on every 6th issue hour to keep it fast
RNG = np.random.default_rng(0)
PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
ROUNDS = 400


def lagged(x, t, k):
    """x at the same hour one or two days before t+k (always <= t)."""
    lag = 24 * np.ceil(np.maximum(k, 1) / 24).astype(int)
    return x[np.clip(t + k - lag, 0, len(x) - 1)]


def grid(n, issue_hours):
    t = np.repeat(issue_hours, H - 1)
    k = np.tile(np.arange(1, H), len(issue_hours))
    keep = t + k < n
    return t[keep], k[keep]


# ---------------------------------------------------------------- demand
def demand_features(d, ts, temp_fc, city_code, t, k):
    roll24 = pd.Series(d).rolling(24, min_periods=1).mean().to_numpy()
    scale = np.maximum(pd.Series(d).rolling(168, min_periods=24).mean().bfill().to_numpy(), 0.05)
    tgt = t + k
    X = np.column_stack([
        k, ts.hour.to_numpy()[tgt], ts.dayofweek.to_numpy()[tgt], ts.month.to_numpy()[tgt],
        lagged(d, t, k) / scale[t], d[np.clip(tgt - 168, 0, None)] / scale[t], d[t] / scale[t], roll24[t] / scale[t],
        temp_fc[tgt], temp_fc[t], np.full(len(t), city_code),
    ])
    return X, scale[t]


def recalibrate(F, d, window=168, lead=24):
    """Online bias correction using past data only: at issue hour t, scale the forecast by
    (actual use over the last week) / (what was forecast 24 h ahead for those same hours).
    A biased forecast misleads the optimizer more than a noisy one, so this matters for control."""
    n = len(d)
    past_fc = np.full(n, np.nan)
    past_fc[lead:] = F[np.arange(n - lead), lead]                 # forecast for hour j, issued at j - 24
    act = pd.Series(d).rolling(window, min_periods=48).sum().to_numpy()
    fc = pd.Series(past_fc).rolling(window, min_periods=48).sum().to_numpy()
    ratio = np.clip(np.nan_to_num(act / np.maximum(fc, 1e-6), nan=1.0), 0.5, 2.0)
    out = F.copy()
    out[:, 1:] *= ratio[:, None]
    return out


def forecast_demand(homes, cities):
    city_codes = {c: i for i, c in enumerate(sorted(homes.city.unique()))}
    data = {}
    for (house, city), g in homes.groupby(["house", "city"]):
        ts = pd.DatetimeIndex(g.timestamp)
        raw = g.demand_kwh.to_numpy(float)
        d = bench.fill_gaps(raw, (ts.dayofweek * 24 + ts.hour).to_numpy())
        temp = cities[cities.city == city].set_index("timestamp").temperature.reindex(ts).interpolate().bfill().to_numpy()
        data[house] = dict(city=city, ts=ts, raw=raw, d=d, temp_fc=temp + RNG.normal(0, 1.5, len(temp)))
    houses = np.array(sorted(data))
    folds = np.arange(len(houses)) % 5
    RNG.shuffle(folds)
    out = {}
    for f in range(5):
        Xs, ys = [], []
        for house in houses[folds != f]:
            h = data[house]
            t, k = grid(len(h["d"]), np.arange(168, len(h["d"]), STRIDE))
            ok = ~np.isnan(h["raw"][t + k])                        # learn only from real readings
            X, scale = demand_features(h["d"], h["ts"], h["temp_fc"], city_codes[h["city"]], t[ok], k[ok])
            Xs.append(X); ys.append(h["raw"][(t + k)[ok]] / scale)
        model = lgb.train(PARAMS, lgb.Dataset(np.vstack(Xs), np.concatenate(ys), categorical_feature=[10]), ROUNDS)
        for house in houses[folds == f]:
            h = data[house]
            n = len(h["d"])
            t, k = grid(n, np.arange(n))
            X, scale = demand_features(h["d"], h["ts"], h["temp_fc"], city_codes[h["city"]], t, k)
            F = np.repeat(h["d"][:, None], H, axis=1).astype(float)            # fallback: measured value
            F[t, k] = np.maximum(model.predict(X) * scale, 0)
            F[:, 0] = h["d"]
            out[house] = recalibrate(F, h["d"])
        print(f"  demand fold {f + 1}/5 done ({(folds == f).sum()} homes)", flush=True)
    return out, data


# ---------------------------------------------------------------- solar
def clear_sky_per_kw(city, ts):
    lat, lon = COORDS[city]
    loc = pvlib.location.Location(lat, lon, tz="Asia/Karachi")
    cs = loc.get_clearsky(ts.tz_localize("Asia/Karachi") + pd.Timedelta(minutes=30))
    w = pd.DataFrame({"Solar Radiation": cs["ghi"].to_numpy(), "Temperature": 25.0, "Wind Speed": 7.0}, index=ts)
    return solar_per_kw(city, w).to_numpy()


def solar_features(s, cs, hum_fc, t, k, city_code):
    tgt = t + k
    day = pd.Series(s).rolling(24, min_periods=1).sum().to_numpy()
    day_cs = pd.Series(cs).rolling(24, min_periods=1).sum().to_numpy()
    return np.column_stack([
        k, cs[tgt], lagged(s, t, k), lagged(cs, t, k), day[t] / np.maximum(day_cs[t], 1e-3),
        hum_fc[tgt], hum_fc[t], np.full(len(t), city_code),
    ])


def forecast_solar(cities):
    data = {}
    for i, (city, g) in enumerate(cities.groupby("city")):
        ts = pd.DatetimeIndex(g.timestamp)
        hum = g.humidity.interpolate().bfill().to_numpy()
        data[city] = dict(code=i, ts=ts, s=g.solar_kwh_per_kw.to_numpy(), cs=clear_sky_per_kw(city, ts),
                          hum_fc=np.clip(hum + RNG.normal(0, 8, len(hum)), 0, 100))
    n = len(next(iter(data.values()))["s"])
    block = (pd.DatetimeIndex(next(iter(data.values()))["ts"]).month.to_numpy() - 1) // 2      # 6 two-month blocks
    out = {c: np.repeat(v["s"][:, None], H, axis=1).astype(float) for c, v in data.items()}
    for b in range(6):
        Xs, ys = [], []
        for c, v in data.items():
            t, k = grid(n, np.arange(24, n, 3))
            m = block[t + k] != b
            Xs.append(solar_features(v["s"], v["cs"], v["hum_fc"], t[m], k[m], v["code"])); ys.append(v["s"][(t + k)[m]])
        model = lgb.train({**PARAMS, "objective": "l2", "min_data_in_leaf": 50}, lgb.Dataset(np.vstack(Xs), np.concatenate(ys),
                                                                                         categorical_feature=[7]), ROUNDS)
        for c, v in data.items():
            t, k = grid(n, np.arange(n))
            m = block[t + k] == b
            out[c][t[m], k[m]] = np.clip(model.predict(solar_features(v["s"], v["cs"], v["hum_fc"], t[m], k[m], v["code"])), 0, 1)
        print(f"  solar block {b + 1}/6 done", flush=True)
    for c in out:
        out[c][:, 0] = data[c]["s"]
        # no sun when clear-sky is zero at the target hour
        n_ = len(data[c]["s"])
        tt = np.arange(n_)[:, None] + np.arange(H)[None, :]
        out[c][(data[c]["cs"][np.clip(tt, 0, n_ - 1)] == 0)] = 0
    return out, data


def skill(name, F, actual, valid=None):
    """Mean absolute error by horizon band for the model vs the persistence forecast, out of sample."""
    n = len(actual)
    t, k = grid(n, np.arange(168, n - H))
    a = actual[t + k]
    ok = ~np.isnan(a) if valid is None else valid[t + k]
    mae_m = np.abs(F[t, k] - a)[ok]
    mae_p = np.abs(lagged(np.nan_to_num(actual), t, k) - a)[ok]
    return mae_m.mean(), mae_p.mean()


def main():
    homes = pd.read_parquet(OUT / "lums_homes.parquet")
    cities = pd.read_parquet(OUT / "lums_cities.parquet")
    print("Training demand forecasters...")
    dem, ddata = forecast_demand(homes, cities)
    print("Training solar forecasters...")
    sol, sdata = forecast_solar(cities)

    bias = np.median([np.nanmean((dem[h][168:-H, 24] - ddata[h]["raw"][192:-H + 24]) / np.nanmean(ddata[h]["raw"])) for h in dem])
    print(f"Demand forecast bias at 24 h, median home: {100 * bias:+.1f}% of mean demand")
    m, p = zip(*[skill(h, dem[h], ddata[h]["raw"]) for h in dem])
    print(f"\nDemand, out-of-sample MAE (kWh/h) averaged over {len(dem)} homes: "
          f"LightGBM {np.mean(m):.3f} vs persistence {np.mean(p):.3f} ({100 * (1 - np.mean(m) / np.mean(p)):.0f}% better)")
    m, p = zip(*[skill(c, sol[c], sdata[c]["s"], valid=sdata[c]["cs"] > 0) for c in sol])
    print(f"Solar (per kW, daylight hours), out-of-sample MAE: LightGBM {np.mean(m):.3f} vs persistence {np.mean(p):.3f} "
          f"({100 * (1 - np.mean(m) / np.mean(p)):.0f}% better)")
    np.savez_compressed(OUT / "lums_forecasts.npz", **{f"demand_{h}": v.astype(np.float32) for h, v in dem.items()},
                        **{f"solar_{c}": v.astype(np.float32) for c, v in sol.items()})
    print(f"Saved {OUT / 'lums_forecasts.npz'}")


if __name__ == "__main__":
    main()

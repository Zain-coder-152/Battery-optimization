"""Build the hourly dataset from the LUMS REWD-P data (run 14_fetch_lums.py first).

Outputs
  data/processed/lums_homes.parquet   one row per home-hour: demand, AC demand, measured rooftop generation
  data/processed/lums_cities.parquet  one row per city-hour: weather + modelled output of 1 kW of panels
  data/processed/lums_homes.csv       one row per home: city, data quality, consumption summary

Solar output is modelled with pvlib from each city's measured hourly radiation (fixed panels, 30 deg tilt,
facing south, 14% system losses, PVWatts DC model with cell temperature). Two Lahore homes have
rooftop panels with a generation meter; they are used to check the timing and the model."""
from pathlib import Path

import numpy as np
import pandas as pd

from lums import CITIES, house_files, hourly, is_ac, load_house_raw, load_weather, solar_per_kw

OUT = Path("data/processed")
START, END = "2023-11-01 00:00", "2024-10-30 23:00"        # the window almost every home covers
IDX = pd.date_range(START, END, freq="h")
WEATHER_SHIFT_H = -1          # weather is stamped ~1 h late; chosen by best match with measured rooftop output


# ---------- cities: weather + solar ----------
cities = []
for city in CITIES:
    w = load_weather(city)
    w.index = w.index + pd.Timedelta(hours=WEATHER_SHIFT_H)
    w = w.reindex(IDX).interpolate(limit=3)
    c = pd.DataFrame({"city": city, "timestamp": IDX,
                      "solar_kwh_per_kw": solar_per_kw(city, w).to_numpy(),
                      "temperature": w["Temperature"].to_numpy(), "humidity": w["Humidity"].to_numpy(),
                      "precipitation": w["Precipitation"].to_numpy(), "radiation": w["Solar Radiation"].to_numpy(),
                      "wind_speed": w["Wind Speed"].to_numpy()})
    cities.append(c)
    print(f"{city:9s} modelled output of 5 kW: {5 * c.solar_kwh_per_kw.sum():,.0f} kWh/yr")
cities = pd.concat(cities, ignore_index=True)
cities.to_parquet(OUT / "lums_cities.parquet", index=False)

# ---------- homes: hourly demand ----------
homes, summary = [], []
for city, house, path in house_files():
    h = hourly(load_house_raw(path)).reindex(IDX)
    ac_cols = [c for c in h.columns if is_ac(c)]
    gen_col = next((c for c in h.columns if c.lower().startswith("generation")), None)
    demand = h["Usage"].clip(lower=0)
    demand = demand.interpolate(limit=2, limit_area="inside")          # fill gaps of up to 2 hours
    frame = pd.DataFrame({
        "city": city, "house": house, "timestamp": IDX, "demand_kwh": demand.to_numpy(),
        "ac_kwh": h[ac_cols].sum(axis=1, min_count=1).to_numpy() if ac_cols else np.nan,
        "generation_kwh": (-h[gen_col]).clip(lower=0).to_numpy() if gen_col else np.nan,
    })
    homes.append(frame)
    valid = demand.notna()
    summary.append({"house": house, "city": city, "valid_pct": round(100 * valid.mean(), 1),
                    "kwh_per_day": round(demand.mean() * 24, 1), "peak_kw": round(demand.max(), 1),
                    "has_ac_meter": bool(ac_cols), "has_rooftop_solar": gen_col is not None,
                    "ac_share_pct": round(100 * frame.ac_kwh.sum() / demand.sum(), 1) if ac_cols else None})
homes = pd.concat(homes, ignore_index=True)
homes.to_parquet(OUT / "lums_homes.parquet", index=False)
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "lums_homes.csv", index=False)

print(f"\n{len(summary)} homes, {len(IDX)} hours each; median valid hours {summary.valid_pct.median()}%")
print(f"homes with >= 90% valid hours: {(summary.valid_pct >= 90).sum()}")

# ---------- check the solar model against the two homes with rooftop panels ----------
print("\nSolar model vs measured rooftop generation (Lahore):")
lahore = cities[cities.city == "Lahore"].set_index("timestamp")["solar_kwh_per_kw"]
for house in summary.loc[summary.has_rooftop_solar, "house"]:
    g = homes[homes.house == house].set_index("timestamp")["generation_kwh"]
    j = pd.concat([g.rename("measured"), lahore.rename("model")], axis=1).dropna()
    j = j[j.measured.resample("D").transform("sum") > 1]            # days the meter was working
    kw = (j.measured @ j.model) / (j.model @ j.model)                 # best-fit system size
    daily = j.resample("D").sum()
    daily = daily[daily.measured > 1]
    print(f"  House{house}: fitted size {kw:.1f} kW | hourly r = {j.measured.corr(j.model):.2f} | "
          f"daily r = {daily.measured.corr(daily.model):.2f} | daily error {100 * (daily.measured - kw * daily.model).abs().mean() / daily.measured.mean():.0f}% "
          f"| {len(daily)} days")

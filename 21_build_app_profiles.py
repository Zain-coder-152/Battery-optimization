"""Build the data the web app ships with: one typical-home demand profile and one solar profile per city.

The LUMS dataset states no license, so the app never ships a home's own readings. For each city it uses
the hourly mean, across that city's homes, of each home's demand divided by its own yearly mean (a
"shape" that averages 1.0) - real Pakistani patterns like heat-wave AC load, no individual home's data.
Solar is the pvlib output per kW from the city's measured 2023-24 weather (16_build_lums_dataset.py).

Cities with fewer than 5 usable homes are left out.

Output: data/app/city_profiles.csv (city, timestamp, solar_kwh_per_kw, demand_shape)
        data/app/cities.json       (city, coordinates, homes averaged, typical kWh/day, yearly solar per 5 kW)"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lums import COORDS

SRC, OUT = Path("data/processed"), Path("data/app")
OUT.mkdir(parents=True, exist_ok=True)
homes = pd.read_parquet(SRC / "lums_homes.parquet")
cities = pd.read_parquet(SRC / "lums_cities.parquet")
quality = pd.read_csv(SRC / "lums_homes.csv").set_index("house")

MIN_HOMES = 5
good = quality.index[quality.valid_pct >= 80]                     # skip homes with mostly missing data
h = homes[homes.house.isin(good)].copy()
h["shape"] = h.demand_kwh / h.groupby("house").demand_kwh.transform("mean")

rows, meta = [], []
for city, g in h.groupby("city"):
    if g.house.nunique() < MIN_HOMES:                              # an "average" of 1-2 homes is not typical
        print(f"{city:9s} skipped: only {g.house.nunique()} homes with usable data")
        continue
    shape = g.pivot_table(index="timestamp", columns="house", values="shape").mean(axis=1)
    shape = shape.interpolate(limit_direction="both")
    shape /= shape.mean()
    solar = cities[cities.city == city].set_index("timestamp").solar_kwh_per_kw.reindex(shape.index)
    rows.append(pd.DataFrame({"city": city, "timestamp": shape.index, "solar_kwh_per_kw": solar.round(4).to_numpy(),
                              "demand_shape": shape.round(4).to_numpy()}))
    lat, lon = COORDS[city]
    n_homes = g.house.nunique()
    meta.append({"id": city.lower(), "name": city, "lat": lat, "lon": lon, "homes": int(n_homes),
                 "typical_kwh_per_day": round(float(quality.loc[g.house.unique(), "kwh_per_day"].median()), 1),
                 "annual_kwh_per_5kw": round(5 * float(solar.sum()))})
    print(f"{city:9s} {n_homes:2d} homes averaged | typical home {meta[-1]['typical_kwh_per_day']} kWh/day | "
          f"5 kW of panels: {meta[-1]['annual_kwh_per_5kw']:,} kWh/yr")

pd.concat(rows).to_csv(OUT / "city_profiles.csv", index=False)
(OUT / "cities.json").write_text(json.dumps(meta, indent=2))
print(f"Saved {OUT / 'city_profiles.csv'} ({(OUT / 'city_profiles.csv').stat().st_size // 1024} KB) and {OUT / 'cities.json'}")

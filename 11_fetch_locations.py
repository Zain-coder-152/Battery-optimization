"""Fetch a PVWatts typical-year hourly solar profile (5 kW system) for several Pakistani cities.
Same system settings as 02a_fetch_solar.py, so every city is directly comparable.
Writes data/raw/solar_<slug>.csv and data/locations.json (used by the API)."""
import json
import os
import time
from pathlib import Path

import requests
import pandas as pd

API_KEY = os.environ.get("NREL_API_KEY", "DEMO_KEY")
HOSTS = ["https://developer.nrel.gov/api/pvwatts/v8.json",
         "https://developer.nlr.gov/api/pvwatts/v8.json"]
RAW = Path("data/raw")

CITIES = {
    "faisalabad": ("Faisalabad", 31.42, 73.08),
    "lahore":     ("Lahore",     31.55, 74.34),
    "karachi":    ("Karachi",    24.86, 67.01),
    "islamabad":  ("Islamabad",  33.68, 73.05),
    "peshawar":   ("Peshawar",   34.01, 71.58),
    "quetta":     ("Quetta",     30.18, 66.98),
    "multan":     ("Multan",     30.16, 71.52),
}
SYSTEM = dict(system_capacity=5, module_type=0, array_type=1, losses=14, tilt=30, azimuth=180,
              timeframe="hourly")


def fetch(lat, lon):
    last = None
    for url in HOSTS:
        for extra in ({}, {"dataset": "intl"}):
            try:
                r = requests.get(url, params={"api_key": API_KEY, "lat": lat, "lon": lon, **SYSTEM, **extra},
                                 timeout=60)
                data = r.json()
                if not data.get("errors") and "outputs" in data:
                    return data
                last = data.get("errors")
            except Exception as e:   # host unreachable -> try the next one
                last = e
    raise RuntimeError(f"PVWatts failed: {last}")


meta = []
for slug, (name, lat, lon) in CITIES.items():
    path = RAW / f"solar_{slug}.csv"
    if not path.exists():
        print(f"Fetching {name}...")
        ac = fetch(lat, lon)["outputs"]["ac"]
        assert len(ac) == 8760, len(ac)
        pd.DataFrame({"solar_wh": ac}).to_csv(path, index=False)
        time.sleep(1.5)               # DEMO_KEY is rate limited
    kwh = pd.read_csv(path)["solar_wh"].sum() / 1000
    print(f"  {name:11s} annual output of a 5 kW system: {kwh:,.0f} kWh")
    meta.append({"id": slug, "name": name, "lat": lat, "lon": lon, "file": path.name,
                 "annual_kwh_per_5kw": round(kwh)})

Path("data/locations.json").write_text(json.dumps(meta, indent=2))
print("Saved data/locations.json")

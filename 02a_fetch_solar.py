import os
import requests
import pandas as pd

API_KEY = os.environ.get("NREL_API_KEY", "DEMO_KEY")
HOSTS = [
    "https://developer.nrel.gov/api/pvwatts/v8.json",
    "https://developer.nlr.gov/api/pvwatts/v8.json",
]

params = dict(
    api_key=API_KEY,
    lat=31.42, lon=73.08,        # Faisalabad
    system_capacity=5,           # kW of panels
    module_type=0, array_type=1,
    losses=14, tilt=30, azimuth=180,
    timeframe="hourly",
)

def fetch(extra=None):
    last_err = None
    for url in HOSTS:
        try:
            r = requests.get(url, params={**params, **(extra or {})}, timeout=60)
            data = r.json()
            print(f"Host answered: {url}")
            return data
        except Exception as e:
            last_err = e
            print(f"Host failed: {url} -> {e}")
    raise SystemExit(f"All hosts failed: {last_err}")

data = fetch()
if data.get("errors"):
    print("First try failed:", data["errors"], "-> retrying with international dataset")
    data = fetch({"dataset": "intl"})

if data.get("errors"):
    raise SystemExit(f"PVWatts error: {data['errors']}")

ac_wh = data["outputs"]["ac"]            # 8760 hourly values in Wh
print("Hours returned:", len(ac_wh))
print("Annual energy (kWh):", round(sum(ac_wh) / 1000))

pd.DataFrame({"solar_wh": ac_wh}).to_csv("data/raw/solar_faisalabad.csv", index=False)
print("Saved data/raw/solar_faisalabad.csv")
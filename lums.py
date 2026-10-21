"""Loading helpers for the LUMS Energy Institute REWD-P dataset (59 Pakistani homes, 1-minute data).
The raw files are inconsistent (column names like "Usage (kW)", "Usage..kW.", "Usage_kW";
duplicate timestamps; Peshawar homes stored twice), so every script reads them through here."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent / "data/raw/lums"
ENERGY = ROOT / "rewdp_dataset/rewdp_dataset"
WEATHER = ROOT / "weather_dataset/weather_dataset"
CITIES = ["Lahore", "Multan", "Peshawar", "Karachi", "Islamabad", "Skardu"]
COORDS = {"Lahore": (31.55, 74.34), "Multan": (30.16, 71.52), "Peshawar": (34.01, 71.58),
          "Karachi": (24.86, 67.01), "Islamabad": (33.68, 73.05), "Skardu": (35.30, 75.63)}


def clean_column(c):
    """'Usage..kW.' / 'Usage_kW' / 'Usage (kW)' -> 'Usage'; 'Washroom_1...Geyser_1..kW.' -> 'Washroom_1 + Geyser_1'."""
    base = re.sub(r"[\s._]*\(?kW\)?\.?$", "", c.strip(), flags=re.IGNORECASE)
    return base.replace("...", " + ").strip()


def is_ac(name):
    return re.search(r"(^|[_\s+])AC($|[_\s+])", name) is not None


def house_files():
    """[(city, house_number, path)] for the 59 homes, skipping Peshawar's duplicate copies."""
    out = []
    for city in CITIES:
        for p in (ENERGY / city).glob(f"{city.lower()}_House*.csv"):
            out.append((city, int(p.stem.split("House")[1]), p))
    return sorted(out, key=lambda x: x[1])


def load_house(path):
    """1-minute kW readings, cleaned column names, sorted, one row per timestamp.
    Returns (frame, number_of_duplicate_timestamps_dropped)."""
    df = load_house_raw(path)
    dup = int(df.index.duplicated().sum())
    return df[~df.index.duplicated(keep="first")], dup


def load_house_raw(path):
    """All rows with cleaned column names, including repeated timestamps (see hourly())."""
    df = pd.read_csv(path)
    df.columns = [clean_column(c) for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    return df.set_index("datetime").sort_index()


def hourly(df, min_samples=45):
    """1-minute kW -> hourly kWh (mean kW over the hour), with the file's timestamp problems handled:
    - fully identical rows are true duplicates and are dropped;
    - rows whose time was truncated to the hour still belong to that hour, so they are kept;
    - rows whose time was truncated to the whole day (many rows stamped exactly 00:00) cannot be
      placed in an hour and are dropped.
    Hours with fewer than `min_samples` readings are NaN."""
    df = df[~(df.reset_index().duplicated().to_numpy())]
    stamp_count = df.groupby(level=0).size()
    day_truncated = stamp_count[(stamp_count > 60) & (stamp_count.index == stamp_count.index.normalize())].index
    df = df[~df.index.isin(day_truncated)]
    grouped = df.resample("h")
    out = grouped.mean()
    n = grouped.count()
    return out.where(n >= min_samples)


def load_weather(city):
    df = pd.read_csv(WEATHER / f"{city}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    df = df.set_index("datetime").sort_index()
    return df[~df.index.duplicated(keep="first")].apply(pd.to_numeric, errors="coerce")

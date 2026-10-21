"""Audit the LUMS REWD-P smart-meter data before using it: coverage, gaps, impossible values,
consumption level, AC share and zero-usage stretches (possible grid outages), plus the city weather.
Writes data/processed/lums_audit.csv and prints a per-city summary."""
from pathlib import Path

import numpy as np
import pandas as pd

from lums import CITIES, ENERGY, house_files, is_ac, load_house, load_weather

OUT = Path("data/processed/lums_audit.csv")


def runs(mask):
    """Lengths (in samples) of consecutive True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


# ---------- duplicate Peshawar files: identical or not? ----------
same = []
for p in sorted((ENERGY / "Peshawar").glob("House*.csv")):
    a, _ = load_house(p)
    b, _ = load_house(p.with_name(f"peshawar_{p.name}"))
    same.append(a.shape == b.shape and np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True))
print(f"Peshawar duplicate copies identical: {sum(same)}/{len(same)}")

rows = []
for city, house, path in house_files():
    df, n_dup = load_house(path)
    usage = df["Usage"]
    ac_cols = [c for c in df.columns if is_ac(c)]
    full = pd.date_range(df.index.min().floor("D"), df.index.max().ceil("D") - pd.Timedelta(minutes=1), freq="min")
    zero = usage.reindex(full).fillna(-1).to_numpy()
    zr = runs(np.abs(zero) <= 0.005)          # real zero readings (gaps are -1, not counted)
    gap = runs(zero == -1)
    rows.append({
        "city": city, "house": house, "start": df.index.min().date(), "end": df.index.max().date(),
        "days": round(len(full) / 1440), "coverage_pct": round(100 * usage.reindex(full).notna().mean(), 1),
        "longest_gap_days": round(gap.max() / 1440, 1) if len(gap) else 0, "duplicate_ts": n_dup,
        "negative": int((df.select_dtypes("number") < 0).sum().sum()),
        "max_kw": round(usage.max(), 2), "kwh_per_day": round(usage.mean() * 24, 1),
        "ac_circuits": len(ac_cols),
        "ac_share_pct": round(100 * df[ac_cols].sum(axis=1).mean() / usage.mean(), 1) if ac_cols else np.nan,
        "zero_runs_ge_30min": int((zr >= 30).sum()), "longest_zero_h": round(zr.max() / 60, 1) if len(zr) else 0,
        "submeters": ", ".join(c for c in df.columns if c != "Usage"),
    })
    r = rows[-1]
    print(f"  {city:9s} House{house:<3d} {r['start']} to {r['end']}  cov {r['coverage_pct']:5.1f}%  "
          f"{r['kwh_per_day']:5.1f} kWh/day  peak {r['max_kw']:5.1f} kW  AC {r['ac_share_pct']}%  "
          f"zero-runs>=30min {r['zero_runs_ge_30min']}  dup {n_dup}  neg {r['negative']}")

audit = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUT, index=False)
print(f"\nSaved {OUT}\n")
print(audit.groupby("city", sort=False).agg(
    homes=("house", "count"), kwh_day_median=("kwh_per_day", "median"), coverage_median=("coverage_pct", "median"),
    ac_share_median=("ac_share_pct", "median"), homes_with_ac_meter=("ac_circuits", lambda x: int((x > 0).sum())),
    outage_like_runs_median=("zero_runs_ge_30min", "median")).round(1).to_string())

print("\nWeather files:")
for city in CITIES:
    w = load_weather(city)
    print(f"  {city:9s} {w.index.min()} to {w.index.max()}  rows {len(w)}  "
          f"missing radiation {100 * w['Solar Radiation'].isna().mean():.1f}%  max radiation {w['Solar Radiation'].max():.0f}")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW, OUT = Path("data/raw"), Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

PRICE_YEAR = 2019                      # non-leap year, full data
DEMAND_YEAR = 2009                     # non-leap year in UCI data
PRICE_COL = "DE_LU_price_day_ahead"    # check the printed list below if this errors

# ---------- DEMAND (UCI, minute-level -> hourly) ----------
uci = pd.read_csv(
    RAW / "household_power_consumption.txt",
    sep=";", na_values=["?"],
    usecols=["Date", "Time", "Global_active_power"], low_memory=False,
)
uci["ts"] = pd.to_datetime(uci["Date"] + " " + uci["Time"], format="%d/%m/%Y %H:%M:%S")
# Average kW over one hour == kWh used in that hour
demand = uci.set_index("ts")["Global_active_power"].resample("h").mean()
demand = demand[demand.index.year == DEMAND_YEAR].interpolate(limit_direction="both")

# ---------- PRICE (OPSD, EUR/MWh -> EUR/kWh) ----------
all_cols = pd.read_csv(RAW / "time_series_60min_singleindex.csv", nrows=0).columns
print("Price columns available:", [c for c in all_cols if "price" in c])

ops = pd.read_csv(
    RAW / "time_series_60min_singleindex.csv",
    usecols=["utc_timestamp", PRICE_COL], parse_dates=["utc_timestamp"],
).set_index("utc_timestamp")
ops.index = ops.index.tz_convert("UTC").tz_localize(None)
price = ops[PRICE_COL]
price = price[price.index.year == PRICE_YEAR].interpolate(limit_direction="both") / 1000

# ---------- SOLAR (PVWatts, Wh -> kWh) ----------
solar = pd.read_csv(RAW / "solar_faisalabad.csv")["solar_wh"] / 1000

# ---------- COMBINE on one hourly clock ----------
assert len(demand) == len(price) == len(solar) == 8760, (len(demand), len(price), len(solar))

idx = pd.date_range(f"{PRICE_YEAR}-01-01", periods=8760, freq="h")
df = pd.DataFrame({
    "price_eur_per_kwh": price.values,
    "demand_kwh": demand.values,
    "solar_kwh": solar.values,
}, index=idx)
df.index.name = "timestamp"

assert df.isna().sum().sum() == 0, "NaNs left in dataset"
df.to_csv(OUT / "hourly_dataset.csv")

print(df.describe().round(4))
print("Annual totals (kWh): demand =", round(df.demand_kwh.sum()),
      "| solar =", round(df.solar_kwh.sum()))

# ---------- SANITY PLOT: one summer week ----------
week = df.loc["2019-07-01":"2019-07-07"]
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
ax[0].plot(week.index, week.price_eur_per_kwh);  ax[0].set_ylabel("Price (EUR/kWh)")
ax[1].plot(week.index, week.demand_kwh);         ax[1].set_ylabel("Demand (kWh)")
ax[2].plot(week.index, week.solar_kwh);          ax[2].set_ylabel("Solar (kWh)")
plt.tight_layout()
plt.savefig(OUT / "sanity_week.png")
plt.show()
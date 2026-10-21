import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import agent
from core import simulate, tou_prices, outage_mask, PEAK_HOURS

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
DEMAND = df["demand_kwh"].to_numpy(float)
HOURS = df.index.hour.to_numpy()
TIMES = df.index
PROFILE = pd.Series(DEMAND).groupby(HOURS).mean().to_numpy()   # typical-day demand (24 values)
TYPICAL = PROFILE[HOURS]
W = 168
LOCATIONS = {loc["id"]: loc for loc in json.loads((ROOT / "data/locations.json").read_text())}
DEFAULT_TARIFF = {"peak": 46.85, "offpeak": 34.53, "export": 11.0, "peak_hours": list(PEAK_HOURS)}
STRATEGIES = {
    "none": "No battery",
    "selfuse": "Self-use rule",
    "peakaware": "Peak-aware rule",
    "learned": "Learned controller",
    "mpc": "Optimizer (MPC)",
}

app = FastAPI(title="Smart Battery Optimizer", version="1.0",
              description="Hour-by-hour home battery scheduling for Pakistani solar homes.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=None)
def solar_5kw(location):
    """Hourly output of a 5 kW system at this location (PVWatts typical year), kWh."""
    return pd.read_csv(ROOT / "data/raw" / LOCATIONS[location]["file"])["solar_wh"].to_numpy(float) / 1000


class Tariff(BaseModel):
    peak: float = Field(DEFAULT_TARIFF["peak"], ge=0, le=500)
    offpeak: float = Field(DEFAULT_TARIFF["offpeak"], ge=0, le=500)
    export: float = Field(DEFAULT_TARIFF["export"], ge=0, le=500)
    peak_hours: list[int] = Field(default_factory=lambda: list(PEAK_HOURS))

    @field_validator("peak_hours")
    @classmethod
    def valid_hours(cls, v):
        if any(h < 0 or h > 23 for h in v):
            raise ValueError("hours must be 0-23")
        return sorted(set(v))


class OptimizeRequest(BaseModel):
    location: str = "faisalabad"
    panel_kw: float = Field(5.0, gt=0, le=50)
    battery_kwh: float = Field(10.0, gt=0, le=100)
    max_power_kw: float = Field(3.0, gt=0, le=50)
    week: int = Field(20, ge=0, le=51)
    demand_scale: float = Field(1.0, gt=0, le=10)
    tariff: Tariff = Field(default_factory=Tariff)
    outage_hours: list[int] = Field(default_factory=list)
    deg_cost: float = Field(0.0, ge=0, le=100)

    @field_validator("outage_hours")
    @classmethod
    def valid_outages(cls, v):
        if any(h < 0 or h > 23 for h in v):
            raise ValueError("hours must be 0-23")
        return sorted(set(v))


def learned_status(req):
    if not agent.available():
        return False, "Model file or PyTorch not available on this server."
    default = (req.location == "faisalabad" and req.panel_kw == 5 and req.battery_kwh == 10
               and req.max_power_kw == 3 and req.demand_scale == 1 and not req.outage_hours
               and req.tariff.model_dump() == DEFAULT_TARIFF)
    if not default:
        return False, ("The learned controller was trained for one system only (Faisalabad, 5 kW panels, "
                       "10 kWh / 3 kW battery, default tariff, no outages). Reset those settings to see it.")
    return True, ""


def summarize(frame, solar_w, demand_w, cap):
    grid, wear = float(frame.cost.sum()), float(frame.degradation.sum())
    demand_total = float(demand_w.sum())
    return {
        "grid_cost": round(grid, 1), "wear_cost": round(wear, 1), "total_cost": round(grid + wear, 1),
        "import_kwh": round(float(frame.grid_import.sum()), 2),
        "export_kwh": round(float(frame.grid_export.sum()), 2),
        "unserved_kwh": round(float(frame.unserved.sum()), 2),
        # share of demand met by own solar + battery (unmet demand during outages does not count as met)
        "self_sufficiency_pct": round(100 * (1 - (frame.grid_import.sum() + frame.unserved.sum()) / demand_total), 1)
        if demand_total else 0,
        "cycles": round(float(frame.battery_flow.abs().sum() / 2 / cap), 2),
    }


def action_label(flow, solar, demand, outage):
    surplus = solar - demand
    if flow > 0.05:
        return "Charge from grid" if flow > surplus + 0.05 and not outage else "Charge from solar"
    if flow < -0.05:
        return "Discharge to home"
    return "Hold"


@app.get("/api/health")
def health():
    return {"status": "ok", "learned_model": agent.available()}


@app.get("/api/config")
def config():
    starts = np.arange(0, 52) * W
    return {
        "locations": list(LOCATIONS.values()),
        "default_tariff": DEFAULT_TARIFF,
        "strategies": STRATEGIES,
        "weeks": [{"index": int(i), "start": TIMES[s].strftime("%Y-%m-%d"),
                   "label": f"{TIMES[s]:%d %b} – {TIMES[min(s + W - 1, len(TIMES) - 1)]:%d %b}"}
                  for i, s in enumerate(starts)],
        "defaults": OptimizeRequest().model_dump(),
        "outage_presets": {"none": [], "light": [19, 20], "heavy": [8, 9, 14, 15, 19, 20]},
    }


@lru_cache(maxsize=256)
def _optimize(key):
    req = OptimizeRequest.model_validate_json(key)
    if req.location not in LOCATIONS:
        raise HTTPException(404, f"Unknown location '{req.location}'")
    t0 = time.perf_counter()
    solar = solar_5kw(req.location) * req.panel_kw / 5.0
    demand = DEMAND * req.demand_scale
    typical = TYPICAL * req.demand_scale
    buy, sell = tou_prices(HOURS, req.tariff.peak, req.tariff.offpeak, req.tariff.export, req.tariff.peak_hours)
    outage = outage_mask(HOURS, req.outage_hours)
    start = req.week * W
    length = min(W, len(solar) - start)
    sl = slice(start, start + length)
    kw = dict(cap=req.battery_kwh, pmax=req.max_power_kw, buy=buy, sell=sell, outage=outage,
              deg_cost=req.deg_cost)

    learned_ok, learned_reason = learned_status(req)
    policies = {"none": "none", "selfuse": "selfuse", "peakaware": "peakaware", "mpc": "mpc"}
    if learned_ok:
        policies["learned"] = agent.LearnedController(solar, demand, HOURS, demand_forecast=typical)
    frames = {k: simulate(solar, demand, HOURS, p, start, length,
                          demand_forecast=typical if k == "mpc" else None, **kw)
              for k, p in policies.items()}

    summary = {k: {"id": k, "name": STRATEGIES[k], **summarize(f, solar[sl], demand[sl], req.battery_kwh)}
               for k, f in frames.items()}
    none_cost = summary["none"]["total_cost"]
    for s in summary.values():
        s["saving_vs_none_pct"] = round(100 * (none_cost - s["total_cost"]) / none_cost, 1) if none_cost > 0 else None

    m = frames["mpc"]
    schedule = [{"time": TIMES[t].strftime("%Y-%m-%dT%H:%M"), "hour": int(HOURS[t]),
                 "action": action_label(f, solar[t], demand[t], outage[t]),
                 "battery_kwh": round(float(f), 2), "soc_kwh": round(float(s), 2),
                 "grid_kwh": round(float(i - e), 2), "price": float(buy[t]), "outage": bool(outage[t])}
                for t, f, s, i, e in zip(m.t[:24], m.battery_flow[:24], m.soc[:24], m.grid_import[:24],
                                         m.grid_export[:24])]

    days = pd.Series(TIMES[sl].normalize())
    daily = []
    for d, idx in days.groupby(days).groups.items():
        row = {"date": d.strftime("%a %d %b")}
        for k, f in frames.items():
            row[k] = round(float((f.cost + f.degradation).to_numpy()[idx].sum()), 1)
        daily.append(row)

    r3 = lambda a: np.round(np.asarray(a, float), 3).tolist()
    return {
        "meta": {"location": LOCATIONS[req.location]["name"],
                 "period": f"{TIMES[start]:%d %b} – {TIMES[start + length - 1]:%d %b}",
                 "hours": length, "learned_available": learned_ok, "learned_reason": learned_reason,
                 "recommended": "mpc", "runtime_ms": round(1000 * (time.perf_counter() - t0))},
        "request": req.model_dump(),
        "summary": summary,
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in TIMES[sl]],
            "solar": r3(solar[sl]), "demand": r3(demand[sl]), "price": r3(buy[sl]),
            "outage": outage[sl].tolist(),
            "series": {k: {"soc": r3(f.soc), "flow": r3(f.battery_flow), "import": r3(f.grid_import),
                           "export": r3(f.grid_export), "unserved": r3(f.unserved)}
                       for k, f in frames.items()},
        },
        "schedule": schedule,
        "daily": daily,
    }


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    return _optimize(req.model_dump_json())


# ---------- the built React dashboard (frontend/dist), served from the same origin ----------
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")

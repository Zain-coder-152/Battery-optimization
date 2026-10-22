import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

PEAK_HOURS = (18, 19, 20, 21)
UNSERVED_PENALTY = 150.0     # PKR per kWh of demand the home could not supply during an outage (planning only)


def tou_prices(hours, peak=46.85, offpeak=34.53, export=11.0, peak_hours=PEAK_HOURS):
    hours = np.asarray(hours)
    buy = np.where(np.isin(hours, peak_hours), peak, offpeak).astype(float)
    sell = np.full(len(hours), float(export))
    return buy, sell


def outage_mask(hours, outage_hours=()):
    """Load-shedding schedule: True where the grid is down (same hours every day)."""
    return np.isin(np.asarray(hours), list(outage_hours))


def mpc_first_action(solar_f, demand_f, buy_f, sell_f, soc, cap, pmax, eff, min_soc,
                     outage_f=None, deg_cost=0.0, terminal_factor=1.0):
    """Solve a linear program over the forecast window.
    Returns net battery charge for the first hour (kWh): + charge, - discharge."""
    x = solve_lp(solar_f, demand_f, buy_f, sell_f, soc, cap, pmax, eff, min_soc,
                 outage_f, deg_cost, terminal_factor)
    if x is None:
        return 0.0
    H = len(solar_f)
    return float(x[0] - x[H])


def oracle_cost(solar, demand, buy, sell, cap=10.0, pmax=3.0, eff=0.95, min_frac=0.1, soc0_frac=0.5,
                outage=None, deg_cost=0.0):
    """Best possible grid + wear cost for a period with perfect knowledge of all of it (the ceiling)."""
    H = len(solar)
    x = solve_lp(solar, demand, np.asarray(buy, float), sell, soc0_frac * cap, cap, pmax, eff,
                 min_frac * cap, outage, deg_cost, terminal_factor=0.0)
    return float(buy @ x[2 * H:3 * H] - np.asarray(sell) @ x[3 * H:4 * H] + deg_cost * x[:2 * H].sum())


def solve_lp(solar_f, demand_f, buy_f, sell_f, soc, cap, pmax, eff, min_soc,
             outage_f=None, deg_cost=0.0, terminal_factor=1.0):
    """Variables per hour: charge c, discharge d, import i, export e, soc s, spill p, unserved u.
    Returns the solution vector, or None if the solver failed."""
    H = len(solar_f)
    out = np.zeros(H, bool) if outage_f is None else np.asarray(outage_f, bool)
    cost = np.concatenate([np.full(2 * H, deg_cost), buy_f, -np.asarray(sell_f, float),
                           np.zeros(H), np.full(H, 0.01), np.full(H, UNSERVED_PENALTY)])
    # value of charge left at the end of the window (it will offset a future off-peak import)
    cost[5 * H - 1] -= terminal_factor * eff * max(buy_f.min() - deg_cost, 0.0)

    I = sp.identity(H, format="csr")
    Z = sp.csr_matrix((H, H))
    # energy balance: c - d - i + e + p - u = solar - demand
    bal = sp.hstack([I, -I, -I, I, Z, I, -I])
    # battery: s_t - s_(t-1) - eff*c + d/eff = 0
    shift = sp.eye(H, k=-1, format="csr")
    bat = sp.hstack([-eff * I, I / eff, Z, Z, I - shift, Z, Z])
    A = sp.vstack([bal, bat], format="csr")
    b = np.concatenate([np.asarray(solar_f) - np.asarray(demand_f), np.zeros(H)])
    b[H] = soc

    grid = [(0, 0) if o else (0, None) for o in out]           # no import/export during an outage
    bounds = ([(0, pmax)] * (2 * H) + grid + grid + [(min_soc, cap)] * H
              + [(0, None)] * H + [(0, None) if o else (0, 0) for o in out])
    res = linprog(cost, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    return res.x if res.status == 0 else None


def simulate(solar, demand, hours, policy, start, length=168, cap=10.0, pmax=3.0, eff=0.95,
             min_frac=0.1, soc0_frac=0.5, horizon=48, demand_forecast=None,
             buy=None, sell=None, outage=None, deg_cost=0.0, terminal_factor=1.0, forecaster=None,
             plan_margin=0.0):
    """Simulate hours [start, start+length).
    policy: 'none' | 'selfuse' | 'peakaware' | 'mpc' | callable(t, soc) -> net charge kWh.
    solar/demand/hours are full-year arrays (MPC looks ahead past the end of the week).
    demand_forecast: optional array used for FUTURE demand in MPC (current hour is always real).
    forecaster: optional callable(t, end) -> (solar_f, demand_f) giving MPC its forecast for hours
    [t, end) as issued at hour t; overrides demand_forecast. The current hour is always measured.
    plan_margin: extra PKR per kWh moved through the battery that MPC adds when planning, never billed.
    It stops the optimizer from trading on thin margins that forecast errors would wipe out.
    buy/sell: price arrays (default: Pakistan TOU). outage: bool array, grid down when True.
    deg_cost: battery wear cost in PKR per kWh charged or discharged."""
    n = len(solar)
    if buy is None or sell is None:
        buy, sell = tou_prices(hours)
    outage = np.zeros(n, bool) if outage is None else np.asarray(outage, bool)
    peak_mask = buy >= buy.max() - 1e-9 if buy.max() > buy.min() + 1e-9 else np.zeros(n, bool)
    min_soc = min_frac * cap
    soc = soc0_frac * cap
    rows = []
    for t in range(start, start + length):
        surplus = solar[t] - demand[t]
        if policy == "none":
            nc = 0.0
        elif policy == "selfuse":
            nc = float(np.clip(surplus, -pmax, pmax))
        elif policy == "peakaware":
            # store surplus; discharge only during the peak price window (or to keep the lights on)
            nc = float(np.clip(surplus, -pmax, pmax)) if surplus >= 0 or peak_mask[t] or outage[t] else 0.0
        elif policy == "mpc":
            e = min(t + horizon, n)
            if forecaster is not None:
                s_f, d_f = (np.array(a, dtype=float) for a in forecaster(t, e))
            else:
                s_f = np.array(solar[t:e], dtype=float)
                d_f = np.array((demand if demand_forecast is None else demand_forecast)[t:e], dtype=float)
            s_f[0], d_f[0] = solar[t], demand[t]
            nc = mpc_first_action(s_f, d_f, buy[t:e], sell[t:e], soc, cap, pmax, eff, min_soc,
                                  outage[t:e], deg_cost + plan_margin, terminal_factor)
        elif callable(policy):
            nc = float(policy(t, soc))
        else:
            raise ValueError("policy must be 'none', 'selfuse', 'peakaware', 'mpc' or a callable")

        charge_in, discharge_out = 0.0, 0.0
        if nc > 0:
            charge_in = min(nc, pmax, (cap - soc) / eff)
            if outage[t]:
                charge_in = min(charge_in, max(surplus, 0.0))     # grid is down: charge from solar only
            soc += charge_in * eff
        else:
            discharge_out = max(0.0, min(-nc, pmax, (soc - min_soc) * eff))
            if outage[t]:
                discharge_out = min(discharge_out, max(-surplus, 0.0))
            soc -= discharge_out / eff

        net = demand[t] - solar[t] + charge_in - discharge_out
        if outage[t]:
            imp, exp, unserved, spill = 0.0, 0.0, max(net, 0.0), max(-net, 0.0)
        else:
            imp, exp, unserved, spill = max(net, 0.0), max(-net, 0.0), 0.0, 0.0
        rows.append({"t": t, "hour": int(hours[t]), "soc": soc, "battery_flow": charge_in - discharge_out,
                     "grid_import": imp, "grid_export": exp, "unserved": unserved, "spill": spill,
                     "cost": imp * buy[t] - exp * sell[t],
                     "degradation": deg_cost * (charge_in + discharge_out)})
    return pd.DataFrame(rows)


def persistence_forecaster(solar, demand):
    """Forecast for hour t+k issued at t: the value 24 h (or 48 h) earlier, so only past data is used.
    Before the first day of data, the earliest available hour is used."""
    def lagged(x, t, e):
        k = np.arange(e - t)
        lag = 24 * np.ceil(np.maximum(k, 1) / 24).astype(int)
        return x[np.maximum(t + k - lag, 0)]
    return lambda t, e: (lagged(solar, t, e), lagged(demand, t, e))

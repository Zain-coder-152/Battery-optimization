import numpy as np
import pytest

from core import simulate, tou_prices, outage_mask, oracle_cost

H = 72
hours = np.arange(H) % 24
solar = np.where((hours >= 7) & (hours <= 17), 2.0 * np.sin(np.pi * (hours - 7) / 10), 0.0)
demand = np.full(H, 0.6) + np.where(np.isin(hours, (19, 20, 21)), 0.8, 0.0)


@pytest.mark.parametrize("policy", ["none", "selfuse", "peakaware", "mpc"])
def test_battery_stays_within_limits(policy):
    r = simulate(solar, demand, hours, policy, 0, 48, cap=10, pmax=3)
    assert r.soc.min() >= 1.0 - 1e-6 and r.soc.max() <= 10.0 + 1e-6
    assert r.battery_flow.abs().max() <= 3.0 + 1e-6


def test_energy_balance_holds_every_hour():
    r = simulate(solar, demand, hours, "mpc", 0, 48)
    lhs = r.grid_import - r.grid_export + r.unserved - r.spill
    rhs = demand[:48] - solar[:48] + r.battery_flow
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_no_battery_means_no_flow():
    r = simulate(solar, demand, hours, "none", 0, 48)
    assert r.battery_flow.abs().sum() == 0
    assert r.soc.nunique() == 1


def test_optimizer_is_not_worse_than_self_use_with_perfect_forecast():
    # credit energy left in the battery at the end (worth an avoided off-peak import later)
    def net_cost(r):
        return r.cost.sum() - (r.soc.iloc[-1] - 5.0) * 0.95 * 34.53
    mpc = net_cost(simulate(solar, demand, hours, "mpc", 0, 48))
    selfuse = net_cost(simulate(solar, demand, hours, "selfuse", 0, 48))
    assert mpc <= selfuse + 1.0


def test_oracle_is_a_lower_bound():
    buy, sell = tou_prices(hours)
    best = oracle_cost(solar[:48], demand[:48], buy[:48], sell[:48])
    for p in ["none", "selfuse", "peakaware"]:
        assert best <= simulate(solar, demand, hours, p, 0, 48).cost.sum() + 1e-6


def test_no_grid_trade_during_outage():
    out = outage_mask(hours, (19, 20))
    for p in ["selfuse", "mpc"]:
        r = simulate(solar, demand, hours, p, 0, 48, outage=out)
        off = out[:48]
        assert r.grid_import[off].sum() == 0 and r.grid_export[off].sum() == 0


def test_optimizer_reserves_charge_for_outages():
    out = outage_mask(hours, (19, 20, 21))
    mpc = simulate(solar, demand, hours, "mpc", 0, 48, outage=out).unserved.sum()
    none = simulate(solar, demand, hours, "none", 0, 48, outage=out).unserved.sum()
    assert mpc < none


def test_wear_cost_is_charged_per_kwh_moved():
    r = simulate(solar, demand, hours, "selfuse", 0, 48, deg_cost=2.0)
    moved = r.battery_flow.abs().sum()
    assert r.degradation.sum() == pytest.approx(2.0 * moved)


def test_tou_prices():
    buy, sell = tou_prices([0, 18, 21, 22], peak=50, offpeak=30, export=10)
    assert buy.tolist() == [30, 50, 50, 30] and sell.tolist() == [10] * 4

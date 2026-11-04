import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_config_lists_cities_and_weeks():
    j = client.get("/api/config").json()
    ids = {loc["id"] for loc in j["locations"]}
    assert {"lahore", "karachi", "islamabad"} <= ids
    assert all(loc["typical_kwh_per_day"] > 0 for loc in j["locations"])
    assert len(j["weeks"]) == 52
    assert j["defaults"]["battery_kwh"] == 10


@pytest.fixture(scope="module")
def default_run():
    r = client.post("/api/optimize", json={})
    assert r.status_code == 200
    return r.json()


def test_optimize_returns_every_strategy(default_run):
    assert set(default_run["summary"]) == {"none", "selfuse", "peakaware", "mpc"}
    assert len(default_run["hourly"]["time"]) == 168
    assert len(default_run["schedule"]) == 24
    assert len(default_run["daily"]) == 7


def test_battery_saves_money(default_run):
    s = default_run["summary"]
    assert s["mpc"]["total_cost"] < s["none"]["total_cost"]
    assert s["selfuse"]["total_cost"] < s["none"]["total_cost"]


def test_daily_consumption_scales_demand():
    small = client.post("/api/optimize", json={"daily_kwh": 10, "week": 5}).json()
    big = client.post("/api/optimize", json={"daily_kwh": 30, "week": 5}).json()
    assert sum(big["hourly"]["demand"]) == pytest.approx(3 * sum(small["hourly"]["demand"]), rel=0.01)


def test_first_week_works_without_history():
    assert client.post("/api/optimize", json={"week": 0}).status_code == 200


def test_outages_reported():
    r = client.post("/api/optimize", json={"outage_hours": [19, 20], "week": 3}).json()
    assert r["summary"]["none"]["unserved_kwh"] > 0
    assert r["summary"]["mpc"]["unserved_kwh"] <= r["summary"]["none"]["unserved_kwh"]


@pytest.mark.parametrize("body", [{"location": "atlantis"}, {"outage_hours": [24]}, {"battery_kwh": -1},
                                  {"tariff": {"peak_hours": [-1]}}, {"week": 60}, {"daily_kwh": 0}])
def test_bad_input_rejected(body):
    assert client.post("/api/optimize", json=body).status_code in (404, 422)


@pytest.mark.parametrize("tariff", [{"export": 40}, {"export": 100}, {"peak": 10, "offpeak": 50, "export": 11},
                                    {"peak": 0, "offpeak": 0, "export": 1, "peak_hours": []}])
def test_export_above_grid_price_is_refused(tariff):
    r = client.post("/api/optimize", json={"tariff": tariff})
    assert r.status_code == 422
    assert "Export price" in r.json()["detail"][0]["msg"]


def test_export_equal_to_grid_price_is_allowed():
    assert client.post("/api/optimize", json={"tariff": {"export": 34.53}, "week": 2}).status_code == 200


@pytest.mark.parametrize("raw", ['{"panel_kw": 1e999}', '{"panel_kw": NaN}', '{"daily_kwh": Infinity}'])
def test_non_finite_numbers_are_refused_cleanly(raw):
    r = client.post("/api/optimize", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


@pytest.mark.parametrize("body", [
    {"panel_kw": 50, "battery_kwh": 100, "max_power_kw": 50, "daily_kwh": 300},
    {"panel_kw": 0.01, "battery_kwh": 0.01, "max_power_kw": 0.01, "daily_kwh": 0.01},
    {"battery_kwh": 100, "max_power_kw": 0.1},
    {"outage_hours": list(range(24))},
    {"tariff": {"peak": 0, "offpeak": 0, "export": 0, "peak_hours": []}},
    {"deg_cost": 100, "outage_hours": [19, 20]},
])
def test_physics_holds_in_extreme_scenarios(body):
    j = client.post("/api/optimize", json={"week": 7, **body}).json()
    q, h = j["request"], j["hourly"]
    cap, pmax = q["battery_kwh"], q["max_power_kw"]
    for name, s in h["series"].items():
        soc = [0.5 * cap] + s["soc"]
        for i in range(168):
            flow = s["flow"][i]
            assert 0.1 * cap - 0.01 <= s["soc"][i] <= cap + 0.01
            assert abs(flow) <= pmax + 0.01
            assert s["soc"][i] == pytest.approx(soc[i] + (flow * 0.95 if flow > 0 else flow / 0.95), abs=0.01 + 0.001 * cap)
            assert min(s["import"][i], s["export"][i]) <= 0.01
            if h["outage"][i]:
                assert s["import"][i] == s["export"][i] == 0
            else:
                net = h["demand"][i] - h["solar"][i] + flow
                assert s["import"][i] - s["export"][i] == pytest.approx(net, abs=0.01)
        assert sum(d[name] for d in j["daily"]) == pytest.approx(j["summary"][name]["total_cost"], abs=1)

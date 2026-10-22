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

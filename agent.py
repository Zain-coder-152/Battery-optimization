"""The learned controller from 09_mpc_and_imitation.py (residual on top of self-use, trained with DAgger).
It was trained for one system: 5 kW panels in Faisalabad, 10 kWh / 3 kW battery, NEPRA TOU tariff, no outages."""
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "imitation_mpc_v7.pt"
CAP, PMAX = 10.0, 3.0
PEAK = (18, 19, 20, 21)

try:
    import torch
    import torch.nn as nn
except ImportError:          # the API still works without torch, just without this controller
    torch = None


def _build(n_in):
    return nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 1), nn.Tanh())


class LearnedController:
    def __init__(self, solar, demand, hours, demand_forecast=None):
        self.solar, self.demand, self.hours = solar, demand, hours
        self.forecast = demand if demand_forecast is None else demand_forecast
        self.n = len(solar)
        self.model = _build(len(self.feat(0, CAP / 2)))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

    def rule_at(self, t):
        return float(np.clip((self.solar[t] - self.demand[t]) / PMAX, -1.0, 1.0))

    def feat(self, t, soc):
        h = self.hours[t]
        idx = np.minimum(np.arange(t, t + 48), self.n - 1)
        dem = np.array(self.forecast[idx], dtype=float)
        dem[0] = self.demand[t]                       # current hour is measured, the rest is forecast
        return np.concatenate([
            [soc / CAP, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), float(h in PEAK),
             ((PEAK[0] - h) % 24) / 24, self.rule_at(t)],
            (self.solar[idx] - dem) / 3.0, self.solar[idx[:24]] / 3.6,
        ]).astype(np.float32)

    def __call__(self, t, soc):
        """Net battery charge for hour t in kWh (+ charge, - discharge)."""
        with torch.no_grad():
            res = 2.0 * float(self.model(torch.from_numpy(self.feat(t, soc)).unsqueeze(0))[0, 0])
        return float(np.clip(self.rule_at(t) + res, -1.0, 1.0)) * PMAX


def available():
    return torch is not None and MODEL_PATH.exists()

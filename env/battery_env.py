import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BatteryEnv(gym.Env):
    """Home solar + battery + grid. One step = one hour.
    tariff="wholesale": old EUR setup. tariff="pk_tou": Pakistan time-of-use in PKR.
    use_forecast=True adds next-24h solar to the state.
    allowed_starts: optional array of episode start hours (used for train/test splits)."""

    def __init__(self, df, capacity_kwh=10.0, max_power_kw=3.0, efficiency=0.95,
                 min_soc_frac=0.10, tariff="wholesale",
                 grid_fee=0.10, sell_ratio=0.5,
                 peak_price=46.85, offpeak_price=34.53, export_price=11.0,
                 peak_hours=(18, 19, 20, 21),
                 episode_hours=168, random_start=True, reward_scale=None,
                 use_forecast=False, allowed_starts=None):
        super().__init__()
        self.demand = df["demand_kwh"].to_numpy(dtype=np.float32)
        self.solar = df["solar_kwh"].to_numpy(dtype=np.float32)
        self.hours = df.index.hour.to_numpy()
        self.n = len(df)
        wholesale = df["price_eur_per_kwh"].to_numpy(dtype=np.float32)

        if tariff == "wholesale":
            self.buy = wholesale + grid_fee
            self.sell = wholesale * sell_ratio
            self.obs_price = wholesale / 0.1
            default_scale = 10.0
        elif tariff == "pk_tou":
            is_peak = np.isin(self.hours, peak_hours)
            self.buy = np.where(is_peak, peak_price, offpeak_price).astype(np.float32)
            self.sell = np.full(self.n, export_price, dtype=np.float32)
            self.obs_price = self.buy / peak_price
            default_scale = 0.01
        else:
            raise ValueError("tariff must be 'wholesale' or 'pk_tou'")
        self.reward_scale = default_scale if reward_scale is None else reward_scale

        self.cap = capacity_kwh
        self.max_power = max_power_kw
        self.eff = efficiency
        self.min_soc = min_soc_frac * capacity_kwh
        self.episode_hours = episode_hours
        self.random_start = random_start
        self.use_forecast = use_forecast
        self.allowed_starts = None if allowed_starts is None else np.asarray(allowed_starts)
        self.solar_cum = np.concatenate([[0.0], np.cumsum(self.solar)]).astype(np.float64)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        obs_dim = 7 if use_forecast else 6
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    def _obs(self):
        t = min(self.t, self.n - 1)
        h = self.hours[t]
        o = [
            self.soc / self.cap,
            self.obs_price[t],
            self.demand[t] / 3.0,
            self.solar[t] / 3.6,
            np.sin(2 * np.pi * h / 24),
            np.cos(2 * np.pi * h / 24),
        ]
        if self.use_forecast:
            nxt = self.solar_cum[min(t + 24, self.n)] - self.solar_cum[t]
            o.append(nxt / 25.0)
        return np.array(o, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = self.n - self.episode_hours
        if options and "start" in options:
            start = int(options["start"])
        elif self.allowed_starts is not None:
            start = int(self.np_random.choice(self.allowed_starts))
        elif self.random_start:
            start = int(self.np_random.integers(0, max_start + 1))
        else:
            start = 0
        self.t = start
        self.end = start + self.episode_hours
        self.soc = 0.5 * self.cap
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        p = a * self.max_power

        charge_in, discharge_out = 0.0, 0.0
        if p > 0:
            charge_in = min(p, (self.cap - self.soc) / self.eff)
            self.soc += charge_in * self.eff
        else:
            discharge_out = max(0.0, min(-p, (self.soc - self.min_soc) * self.eff))
            self.soc -= discharge_out / self.eff

        net = self.demand[self.t] - self.solar[self.t] + charge_in - discharge_out
        imp, exp = max(net, 0.0), max(-net, 0.0)

        cost = imp * self.buy[self.t] - exp * self.sell[self.t]
        reward = -cost * self.reward_scale

        self.t += 1
        truncated = self.t >= self.end
        info = {"cost": float(cost), "soc": float(self.soc),
                "import": float(imp), "export": float(exp)}
        return self._obs(), float(reward), False, truncated, info
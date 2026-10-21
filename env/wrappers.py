import numpy as np
import gymnasium as gym

PEAK_HOURS = (18, 19, 20, 21)


def rule_action(obs, max_power):
    """Self-consumption: charge from solar surplus, discharge on any shortfall."""
    demand, solar = obs[2] * 3.0, obs[3] * 3.6     # undo BatteryEnv scaling
    return float(np.clip((solar - demand) / max_power, -1.0, 1.0))


def _hour_from_obs(obs):
    ang = np.arctan2(obs[4], obs[5]) % (2 * np.pi)  # obs[4]=sin, obs[5]=cos
    return int(round(ang * 24 / (2 * np.pi))) % 24


def peak_aware_action(obs, max_power):
    """Charge from solar surplus; discharge ONLY during the evening peak."""
    demand, solar = obs[2] * 3.0, obs[3] * 3.6
    surplus = solar - demand
    if surplus >= 0:
        return float(np.clip(surplus / max_power, -1.0, 1.0))
    if _hour_from_obs(obs) in PEAK_HOURS:
        return float(np.clip(surplus / max_power, -1.0, 1.0))
    return 0.0


class RuleResidualWrapper(gym.Wrapper):
    """Final action = base rule action + scale * agent action (agent learns corrections)."""

    def __init__(self, env, base_fn=rule_action, scale=0.5):
        super().__init__(env)
        self.base_fn = base_fn
        self.scale = scale
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs = obs
        return obs, info

    def step(self, action):
        base = self.base_fn(self._obs, self.env.unwrapped.max_power)
        a = np.clip(base + self.scale * float(np.asarray(action).flatten()[0]), -1.0, 1.0)
        obs, r, term, trunc, info = self.env.step(np.array([a], dtype=np.float32))
        self._obs = obs
        return obs, r, term, trunc, info
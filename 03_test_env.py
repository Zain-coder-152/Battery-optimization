import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from env.battery_env import BatteryEnv

df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
env = BatteryEnv(df)

check_env(env, warn=True)
print("check_env passed")

def run_episode(policy, start):
    obs, _ = env.reset(options={"start": start})
    socs, costs, done = [], [], False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        socs.append(info["soc"])
        costs.append(info["cost"])
        done = term or trunc
    return np.array(socs), np.array(costs)

random_policy = lambda obs: env.action_space.sample()
idle_policy = lambda obs: np.array([0.0], dtype=np.float32)

# 1) battery limits must never be broken
socs, costs = run_episode(random_policy, start=4000)
assert socs.min() >= env.min_soc - 1e-6 and socs.max() <= env.cap + 1e-6
print(f"Battery stayed within [{socs.min():.2f}, {socs.max():.2f}] kWh -> OK")

# 2) compare random vs idle (no-battery-use) over several weeks
starts = [500, 2000, 4000, 5500, 7500]
rand_cost = np.mean([run_episode(random_policy, s)[1].sum() for s in starts])
idle_cost = np.mean([run_episode(idle_policy, s)[1].sum() for s in starts])
print(f"Avg weekly cost  idle: {idle_cost:.2f} EUR | random: {rand_cost:.2f} EUR")

# 3) plot one random week
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(socs); ax[0].set_ylabel("Battery (kWh)")
ax[1].plot(np.cumsum(costs)); ax[1].set_ylabel("Cumulative cost (EUR)")
ax[1].set_xlabel("Hour of episode")
plt.tight_layout()
plt.savefig("data/processed/env_test_week.png")
plt.show()
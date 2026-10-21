import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.battery_env import BatteryEnv

Path("models").mkdir(exist_ok=True)

# ---------- DATA: time-based train/test split ----------
df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
split = int(len(df) * 0.7)
train_df, test_df = df.iloc[:split], df.iloc[split:]
print(f"Train hours: {len(train_df)} | Test hours: {len(test_df)}")

train_env = Monitor(BatteryEnv(train_df))                   # random weeks from the training period
test_env = BatteryEnv(test_df, random_start=False)          # fixed weeks from the test period

# ---------- POLICIES ----------
def idle_policy(obs):
    return np.array([0.0], dtype=np.float32)

def rule_policy(obs):
    # undo the observation scaling used inside BatteryEnv
    demand, solar = obs[2] * 3.0, obs[3] * 3.6
    surplus = solar - demand                                # >0: extra solar, <0: shortfall
    a = np.clip(surplus / test_env.max_power, -1.0, 1.0)    # charge on surplus, discharge on shortfall
    return np.array([a], dtype=np.float32)

# ---------- EVALUATION: same test weeks for every policy ----------
def run_week(policy, start):
    obs, _ = test_env.reset(options={"start": start})
    total, socs, done = 0.0, [], False
    while not done:
        obs, r, term, trunc, info = test_env.step(policy(obs))
        total += info["cost"]
        socs.append(info["soc"])
        done = term or trunc
    return total, socs

def evaluate(policy):
    starts = range(0, len(test_df) - test_env.episode_hours + 1, test_env.episode_hours)
    return np.array([run_week(policy, s)[0] for s in starts])

# ---------- TRAIN PPO ----------
model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./runs", seed=0)
model.learn(total_timesteps=200_000)
model.save("models/ppo_battery_v1")

ppo_policy = lambda obs: model.predict(obs, deterministic=True)[0]

# ---------- COMPARE ----------
results = {
    "No battery": evaluate(idle_policy),
    "Rule-based": evaluate(rule_policy),
    "PPO agent":  evaluate(ppo_policy),
}

base = results["No battery"].mean()
print("\n=== Average weekly cost on UNSEEN test weeks ===")
for name, costs in results.items():
    saving = 100 * (base - costs.mean()) / base
    print(f"{name:12s}: {costs.mean():6.2f} EUR/week | saving vs no battery: {saving:5.1f}%")

# ---------- PLOTS ----------
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for name, costs in results.items():
    ax[0].plot(costs, marker="o", label=name)
ax[0].set_xlabel("Test week"); ax[0].set_ylabel("Weekly cost (EUR)"); ax[0].legend()

_, soc_rule = run_week(rule_policy, 0)
_, soc_ppo = run_week(ppo_policy, 0)
ax[1].plot(soc_rule, label="Rule-based"); ax[1].plot(soc_ppo, label="PPO agent")
ax[1].set_xlabel("Hour of first test week"); ax[1].set_ylabel("Battery (kWh)"); ax[1].legend()
plt.tight_layout()
plt.savefig("data/processed/phase4_comparison.png")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.battery_env import BatteryEnv
from env.wrappers import RuleResidualWrapper, rule_action, peak_aware_action

df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
split = int(len(df) * 0.7)
train_df, test_df = df.iloc[:split], df.iloc[split:]

def make(d, **kw):
    return BatteryEnv(d, tariff="pk_tou", **kw)

train_env = Monitor(RuleResidualWrapper(make(train_df), base_fn=peak_aware_action, scale=0.5))
test_raw = make(test_df, random_start=False)
test_res = RuleResidualWrapper(make(test_df, random_start=False), base_fn=peak_aware_action, scale=0.5)

mp = test_raw.max_power
idle_policy = lambda obs: np.array([0.0], dtype=np.float32)
rule_policy = lambda obs: np.array([rule_action(obs, mp)], dtype=np.float32)
peak_policy = lambda obs: np.array([peak_aware_action(obs, mp)], dtype=np.float32)

def run_week(env, policy, start):
    obs, _ = env.reset(options={"start": start})
    total, socs, done = 0.0, [], False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += info["cost"]
        socs.append(info["soc"])
        done = term or trunc
    return total, socs

def evaluate(env, policy):
    n = test_raw.episode_hours
    return np.array([run_week(env, policy, s)[0] for s in range(0, len(test_df) - n + 1, n)])

# ---------- TRAIN ----------
model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./runs", seed=0,
            policy_kwargs=dict(log_std_init=-1.0))
model.learn(total_timesteps=300_000)
model.save("models/ppo_battery_tou_v3")
ppo_policy = lambda obs: model.predict(obs, deterministic=True)[0]

# ---------- COMPARE (costs in PKR) ----------
results = {
    "No battery":       evaluate(test_raw, idle_policy),
    "Self-use rule":    evaluate(test_raw, rule_policy),
    "Peak-aware rule":  evaluate(test_raw, peak_policy),
    "PPO (on peak rule)": evaluate(test_res, ppo_policy),
}
base = results["No battery"].mean()
print("\n=== Average weekly cost on UNSEEN test weeks (PKR) ===")
for name, c in results.items():
    print(f"{name:19s}: {c.mean():8.0f} PKR/week | saving vs no battery: {100*(base-c.mean())/base:5.1f}%")

best_rule = results["Peak-aware rule"]
ppo_c = results["PPO (on peak rule)"]
print(f"PPO vs peak-aware rule: {100*(best_rule.mean()-ppo_c.mean())/best_rule.mean():+.1f}% (positive = PPO cheaper)")
print("PPO beat the peak-aware rule in", int((ppo_c < best_rule).sum()), "of", len(ppo_c), "test weeks")

# ---------- PLOTS ----------
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for name, c in results.items():
    ax[0].plot(c, marker="o", label=name)
ax[0].set_xlabel("Test week"); ax[0].set_ylabel("Weekly cost (PKR)"); ax[0].legend()

_, s1 = run_week(test_raw, peak_policy, 0)
_, s2 = run_week(test_res, ppo_policy, 0)
ax[1].plot(s1, label="Peak-aware rule"); ax[1].plot(s2, label="PPO (on peak rule)")
ax[1].set_xlabel("Hour of first test week"); ax[1].set_ylabel("Battery (kWh)"); ax[1].legend()
plt.tight_layout()
plt.savefig("data/processed/phase5_tou.png")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.battery_env import BatteryEnv
from env.wrappers import RuleResidualWrapper, rule_action

# ---------- DATA (same split as before) ----------
df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
split = int(len(df) * 0.7)
train_df, test_df = df.iloc[:split], df.iloc[split:]

train_env = Monitor(RuleResidualWrapper(BatteryEnv(train_df), scale=0.5))
test_raw = BatteryEnv(test_df, random_start=False)
test_res = RuleResidualWrapper(BatteryEnv(test_df, random_start=False), scale=0.5)

# ---------- POLICIES ----------
idle_policy = lambda obs: np.array([0.0], dtype=np.float32)
rule_policy = lambda obs: np.array([rule_action(obs, test_raw.max_power)], dtype=np.float32)

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
    starts = range(0, len(test_df) - n + 1, n)
    return np.array([run_week(env, policy, s)[0] for s in starts])

# ---------- TRAIN ----------
model = PPO(
    "MlpPolicy", train_env, verbose=1, tensorboard_log="./runs", seed=0,
    policy_kwargs=dict(log_std_init=-1.0),     # less random exploration
)
model.learn(total_timesteps=300_000)
model.save("models/ppo_battery_residual_v2")

ppo_policy = lambda obs: model.predict(obs, deterministic=True)[0]

# ---------- COMPARE ----------
results = {
    "No battery":    evaluate(test_raw, idle_policy),
    "Rule-based":    evaluate(test_raw, rule_policy),
    "PPO (residual)": evaluate(test_res, ppo_policy),
}

base = results["No battery"].mean()
rule = results["Rule-based"].mean()
print("\n=== Average weekly cost on UNSEEN test weeks ===")
for name, c in results.items():
    print(f"{name:15s}: {c.mean():6.2f} EUR/week | vs no battery: {100*(base-c.mean())/base:5.1f}%")
ppo_mean = results["PPO (residual)"].mean()
print(f"PPO vs rule-based: {100*(rule-ppo_mean)/rule:+.1f}%  (positive = PPO is cheaper)")
print("PPO beat rule in", int((results['PPO (residual)'] < results['Rule-based']).sum()),
      "of", len(results['Rule-based']), "test weeks")

# ---------- PLOTS ----------
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for name, c in results.items():
    ax[0].plot(c, marker="o", label=name)
ax[0].set_xlabel("Test week"); ax[0].set_ylabel("Weekly cost (EUR)"); ax[0].legend()

_, soc_rule = run_week(test_raw, rule_policy, 0)
_, soc_ppo = run_week(test_res, ppo_policy, 0)
ax[1].plot(soc_rule, label="Rule-based"); ax[1].plot(soc_ppo, label="PPO (residual)")
ax[1].set_xlabel("Hour of first test week"); ax[1].set_ylabel("Battery (kWh)"); ax[1].legend()
plt.tight_layout()
plt.savefig("data/processed/phase4_residual.png")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from env.battery_env import BatteryEnv
from env.wrappers import RuleResidualWrapper, rule_action, peak_aware_action

df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
n, W = len(df), 168

# ---------- same week-interleaved split as before ----------
week_starts = np.arange(0, n - W + 1, W)
test_weeks = week_starts[3::4]
is_test = np.zeros(n, dtype=int)
for s in test_weeks:
    is_test[s:s + W] = 1
cum = np.concatenate([[0], np.cumsum(is_test)])
all_starts = np.arange(0, n - W + 1)
train_starts = all_starts[(cum[all_starts + W] - cum[all_starts]) == 0]

def make(forecast, train):
    return BatteryEnv(df, tariff="pk_tou", use_forecast=forecast,
                      allowed_starts=train_starts if train else None, random_start=train)

def wrapped(train):
    return RuleResidualWrapper(make(True, train), base_fn=rule_action, scale=0.5)

# ---------- POLICIES ----------
test_raw = make(False, train=False)
mp = test_raw.max_power
idle_policy = lambda obs: np.array([0.0], dtype=np.float32)
rule_policy = lambda obs: np.array([rule_action(obs, mp)], dtype=np.float32)
peak_policy = lambda obs: np.array([peak_aware_action(obs, mp)], dtype=np.float32)

def run_week(env, policy, start):
    obs, _ = env.reset(options={"start": int(start)})
    total, socs, done = 0.0, [], False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += info["cost"]
        socs.append(info["soc"])
        done = term or trunc
    return total, socs

def evaluate(env, policy):
    return np.array([run_week(env, policy, s)[0] for s in test_weeks])

# ---------- TRAIN ----------
ppo = PPO("MlpPolicy", Monitor(wrapped(True)), verbose=1, seed=0,
          tensorboard_log="./runs", policy_kwargs=dict(log_std_init=-1.0))
ppo.learn(total_timesteps=300_000)
ppo.save("models/ppo_selfuse_fc_v5")

sac = SAC("MlpPolicy", Monitor(wrapped(True)), verbose=1, seed=0,
          tensorboard_log="./runs", buffer_size=200_000, learning_starts=2_000)
sac.learn(total_timesteps=100_000)
sac.save("models/sac_selfuse_fc_v5")

test_ppo, test_sac = wrapped(False), wrapped(False)
ppo_policy = lambda obs: ppo.predict(obs, deterministic=True)[0]
sac_policy = lambda obs: sac.predict(obs, deterministic=True)[0]

# ---------- COMPARE ----------
results = {
    "No battery":      evaluate(test_raw, idle_policy),
    "Self-use rule":   evaluate(test_raw, rule_policy),
    "Peak-aware rule": evaluate(test_raw, peak_policy),
    "PPO (+forecast)": evaluate(test_ppo, ppo_policy),
    "SAC (+forecast)": evaluate(test_sac, sac_policy),
}
base = results["No battery"].mean()
selfuse = results["Self-use rule"]

print("\n=== Average weekly cost on HELD-OUT weeks (PKR) ===")
for name, c in results.items():
    print(f"{name:16s}: {c.mean():8.0f} PKR/week | saving vs no battery: {100*(base-c.mean())/base:5.1f}%")
for name in ["PPO (+forecast)", "SAC (+forecast)"]:
    c = results[name]
    print(f"{name}: vs self-use rule {100*(selfuse.mean()-c.mean())/selfuse.mean():+.1f}% "
          f"| cheaper than self-use in {int((c < selfuse).sum())}/{len(c)} weeks")

# ---------- PLOTS (same colours in both panels) ----------
colors = {"No battery": "tab:blue", "Self-use rule": "tab:orange", "Peak-aware rule": "tab:green",
          "PPO (+forecast)": "tab:red", "SAC (+forecast)": "tab:purple"}
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for name, c in results.items():
    ax[0].plot(c, marker="o", label=name, color=colors[name])
ax[0].set_xlabel("Held-out week"); ax[0].set_ylabel("Weekly cost (PKR)"); ax[0].legend(fontsize=8)

s0 = test_weeks[0]
ax[1].plot(run_week(test_raw, rule_policy, s0)[1], label="Self-use rule", color=colors["Self-use rule"])
ax[1].plot(run_week(test_ppo, ppo_policy, s0)[1], label="PPO (+forecast)", color=colors["PPO (+forecast)"])
ax[1].plot(run_week(test_sac, sac_policy, s0)[1], label="SAC (+forecast)", color=colors["SAC (+forecast)"])
ax[1].set_xlabel("Hour of first held-out week"); ax[1].set_ylabel("Battery (kWh)"); ax[1].legend()
plt.tight_layout()
plt.savefig("data/processed/phase5_sac_vs_ppo.png")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.battery_env import BatteryEnv
from env.wrappers import RuleResidualWrapper, rule_action, peak_aware_action

df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
n, W = len(df), 168

# ---------- WEEK-INTERLEAVED SPLIT: hold out every 4th week of the whole year ----------
week_starts = np.arange(0, n - W + 1, W)                 # 52 whole weeks
test_weeks = week_starts[3::4]                           # 13 held-out weeks, spread over all seasons
is_test = np.zeros(n, dtype=int)
for s in test_weeks:
    is_test[s:s + W] = 1
cum = np.concatenate([[0], np.cumsum(is_test)])
all_starts = np.arange(0, n - W + 1)
train_starts = all_starts[(cum[all_starts + W] - cum[all_starts]) == 0]   # windows that touch no test hour
print(f"Test weeks: {len(test_weeks)} | possible training start hours: {len(train_starts)}")

def make(forecast, train):
    return BatteryEnv(df, tariff="pk_tou", use_forecast=forecast,
                      allowed_starts=train_starts if train else None,
                      random_start=train)

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

# ---------- TRAIN TWO AGENTS ----------
def train(forecast, name):
    env = Monitor(RuleResidualWrapper(make(forecast, train=True), base_fn=peak_aware_action, scale=0.5))
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./runs", seed=0,
                policy_kwargs=dict(log_std_init=-1.0))
    model.learn(total_timesteps=300_000)
    model.save(f"models/{name}")
    return model

model_a = train(False, "ppo_tou_nofc_v4")
model_b = train(True, "ppo_tou_forecast_v4")

res_a = RuleResidualWrapper(make(False, train=False), base_fn=peak_aware_action, scale=0.5)
res_b = RuleResidualWrapper(make(True, train=False), base_fn=peak_aware_action, scale=0.5)
ppo_a = lambda obs: model_a.predict(obs, deterministic=True)[0]
ppo_b = lambda obs: model_b.predict(obs, deterministic=True)[0]

# ---------- COMPARE (same held-out weeks for every policy, PKR) ----------
results = {
    "No battery":        evaluate(test_raw, idle_policy),
    "Self-use rule":     evaluate(test_raw, rule_policy),
    "Peak-aware rule":   evaluate(test_raw, peak_policy),
    "PPO (no forecast)": evaluate(res_a, ppo_a),
    "PPO (+forecast)":   evaluate(res_b, ppo_b),
}
base = results["No battery"].mean()
best_rule = np.minimum(results["Self-use rule"], results["Peak-aware rule"])   # best rule PER WEEK

print("\n=== Average weekly cost on HELD-OUT weeks (PKR) ===")
for name, c in results.items():
    print(f"{name:18s}: {c.mean():8.0f} PKR/week | saving vs no battery: {100*(base-c.mean())/base:5.1f}%")
print(f"{'Best rule per week':18s}: {best_rule.mean():8.0f} PKR/week  (hindsight benchmark)")

for name in ["PPO (no forecast)", "PPO (+forecast)"]:
    c = results[name]
    print(f"{name}: vs peak-aware {100*(results['Peak-aware rule'].mean()-c.mean())/results['Peak-aware rule'].mean():+.1f}% | "
          f"vs self-use {100*(results['Self-use rule'].mean()-c.mean())/results['Self-use rule'].mean():+.1f}% | "
          f"beat both rules in {int(((c < results['Peak-aware rule']) & (c < results['Self-use rule'])).sum())}/{len(c)} weeks")

# ---------- PLOTS ----------
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for name, c in results.items():
    ax[0].plot(c, marker="o", label=name)
ax[0].set_xlabel("Held-out week"); ax[0].set_ylabel("Weekly cost (PKR)"); ax[0].legend(fontsize=8)

s0 = test_weeks[0]
ax[1].plot(run_week(test_raw, peak_policy, s0)[1], label="Peak-aware rule")
ax[1].plot(run_week(res_a, ppo_a, s0)[1], label="PPO (no forecast)")
ax[1].plot(run_week(res_b, ppo_b, s0)[1], label="PPO (+forecast)")
ax[1].set_xlabel("Hour of first held-out week"); ax[1].set_ylabel("Battery (kWh)"); ax[1].legend()
plt.tight_layout()
plt.savefig("data/processed/phase5_forecast.png")
plt.show()
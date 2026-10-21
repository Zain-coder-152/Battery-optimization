import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from env.battery_env import BatteryEnv
from env.wrappers import rule_action, peak_aware_action

torch.manual_seed(0)
df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
n, W = len(df), 168
solar = df["solar_kwh"].to_numpy(float)
demand = df["demand_kwh"].to_numpy(float)
hours = df.index.hour.to_numpy()
PEAK = (18, 19, 20, 21)

# ---------- same week-interleaved split as before ----------
week_starts = np.arange(0, n - W + 1, W)
test_weeks = week_starts[3::4]
train_weeks = [int(s) for s in week_starts if s not in set(test_weeks.tolist())]

env = BatteryEnv(df, tariff="pk_tou", random_start=False)
CAP, PMAX, EFF, MIN_SOC = env.cap, env.max_power, env.eff, env.min_soc
BUY, SELL = env.buy.astype(float), env.sell.astype(float)
TERM_VALUE = EFF * BUY.min()      # rough value of 1 kWh left in the battery at the horizon end


# ---------- LINEAR PROGRAM: best battery schedule with a perfect forecast ----------
def plan(t0, soc0, H, terminal=True):
    """Variables per hour: c (charge from grid side), d (discharge), i (import), e (export), s (soc after hour).
    Returns (net charge in first hour, objective value)."""
    H = min(H, n - t0)
    cost = np.concatenate([np.zeros(2 * H), BUY[t0:t0 + H], -SELL[t0:t0 + H], np.zeros(H)])
    if terminal:
        cost[5 * H - 1] -= TERM_VALUE
    A = np.zeros((2 * H, 5 * H))
    b = np.zeros(2 * H)
    for t in range(H):
        # energy balance: c - d - i + e = solar - demand
        A[t, t], A[t, H + t], A[t, 2 * H + t], A[t, 3 * H + t] = 1, -1, -1, 1
        b[t] = solar[t0 + t] - demand[t0 + t]
        # battery: s_t - s_(t-1) - eff*c + d/eff = 0
        A[H + t, 4 * H + t] = 1
        if t > 0:
            A[H + t, 4 * H + t - 1] = -1
        else:
            b[H + t] = soc0
        A[H + t, t] = -EFF
        A[H + t, H + t] = 1 / EFF
    bounds = [(0, PMAX)] * (2 * H) + [(0, None)] * (2 * H) + [(MIN_SOC, CAP)] * H
    res = linprog(cost, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if res.status != 0:
        return 0.0, np.nan
    return float(res.x[0] - res.x[H]), float(res.fun)


# ---------- generic runner: always scored by the same environment ----------
def run(act, start):
    obs, _ = env.reset(options={"start": int(start)})
    total, done = 0.0, False
    while not done:
        a = float(np.clip(act(obs, env), -1.0, 1.0))
        obs, r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        total += info["cost"]
        done = term or trunc
    return total


idle = lambda o, e: 0.0
selfuse = lambda o, e: rule_action(o, PMAX)
peakrule = lambda o, e: peak_aware_action(o, PMAX)
mpc = lambda o, e: plan(e.t, e.soc, 48)[0] / PMAX


# ---------- IMITATION LEARNING (residual + DAgger) ----------
# The network only learns a CORRECTION on top of the self-use rule, so an output of 0 is exactly
# self-use (the summer optimum). DAgger then labels the states the learned policy itself visits,
# so its small mistakes don't push it into situations it never saw during training.
def rule_at(t):
    return float(np.clip((solar[t] - demand[t]) / PMAX, -1.0, 1.0))

def feat(t, soc):
    h = hours[t]
    idx = np.minimum(np.arange(t, t + 48), n - 1)          # same 48h window MPC sees
    return np.concatenate([
        [soc / CAP, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), float(h in PEAK),
         ((PEAK[0] - h) % 24) / 24, rule_at(t)],
        (solar[idx] - demand[idx]) / 3.0, solar[idx[:24]] / 3.6,
    ]).astype(np.float32)

model = nn.Sequential(nn.Linear(len(feat(0, CAP / 2)), 256), nn.ReLU(),
                      nn.Linear(256, 256), nn.ReLU(),
                      nn.Linear(256, 1), nn.Tanh())
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def residual(t, soc):
    with torch.no_grad():
        return 2.0 * float(model(torch.tensor(feat(t, soc)).unsqueeze(0))[0, 0])

def bc(o, e):
    return rule_at(e.t) + residual(e.t, e.soc)

def collect(policy, soc_init):
    """Roll out `policy` on every training week; label each visited state with the MPC action."""
    for s in train_weeks:
        env.reset(options={"start": s})
        if soc_init is not None:
            env.soc = float(soc_init())
        done = False
        while not done:
            a_mpc = float(np.clip(plan(env.t, env.soc, 48)[0] / PMAX, -1, 1))
            X.append(feat(env.t, env.soc)); Y.append(a_mpc - rule_at(env.t))
            a = a_mpc if policy is None else float(np.clip(policy(None, env), -1, 1))
            _, _, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
            done = term or trunc

def train(epochs):
    X_t = torch.tensor(np.array(X))
    Y_t = torch.tensor(np.array(Y, dtype=np.float32)).unsqueeze(1) / 2.0
    for epoch in range(epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = nn.functional.mse_loss(model(X_t[b]), Y_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

rng = np.random.default_rng(0)
rand_soc = lambda: rng.uniform(MIN_SOC, CAP)
X, Y = [], []
print("DAgger round 0: MPC demonstrations on training weeks...")
collect(None, None); collect(None, rand_soc)
print(f"  loss {train(60):.4f} | samples {len(X)}")
for rnd in range(1, 4):
    print(f"DAgger round {rnd}: learned policy drives, MPC labels...")
    collect(bc, rand_soc if rnd % 2 == 0 else None)
    print(f"  loss {train(40):.4f} | samples {len(X)}")
torch.save(model.state_dict(), "models/imitation_mpc_v7.pt")


# ---------- EVALUATE on HELD-OUT weeks ----------
print("Evaluating on held-out weeks...")
results = {
    "No battery":      np.array([run(idle, s) for s in test_weeks]),
    "Self-use rule":   np.array([run(selfuse, s) for s in test_weeks]),
    "Peak-aware rule": np.array([run(peakrule, s) for s in test_weeks]),
    "MPC (48h)":       np.array([run(mpc, s) for s in test_weeks]),
    "Learned (copies MPC)": np.array([run(bc, s) for s in test_weeks]),
}
oracle = np.array([plan(int(s), 0.5 * CAP, W, terminal=False)[1] for s in test_weeks])

base, su = results["No battery"].mean(), results["Self-use rule"].mean()
print("\n=== Average weekly cost on HELD-OUT weeks (PKR) ===")
for name, c in results.items():
    print(f"{name:21s}: {c.mean():7.0f} | saving vs no battery: {100 * (base - c.mean()) / base:5.1f}% "
          f"| vs self-use: {100 * (su - c.mean()) / su:+5.1f}% | cheaper than self-use in "
          f"{int((c < results['Self-use rule']).sum())}/{len(c)} weeks")
print(f"{'Oracle (ceiling)':21s}: {oracle.mean():7.0f} | saving vs no battery: {100 * (base - oracle.mean()) / base:5.1f}% "
      f"| vs self-use: {100 * (su - oracle.mean()) / su:+5.1f}%")

plt.figure(figsize=(10, 5))
for name, c in results.items():
    plt.plot(c, marker="o", label=name)
plt.plot(oracle, "k--", label="Oracle (ceiling)")
plt.xlabel("Held-out week"); plt.ylabel("Weekly cost (PKR)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("data/processed/phase5_mpc_imitation.png"); plt.show()
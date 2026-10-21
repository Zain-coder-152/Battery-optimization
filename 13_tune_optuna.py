"""Phase 5: hyperparameter tuning with Optuna, tracked in MLflow.
Tunes the optimizer's forecast horizon and end-of-window battery value on TRAINING weeks only
(a sample of 10 of them, to keep each trial fast), then scores the best setting once on the held-out weeks."""
import json

import mlflow
import numpy as np
import optuna
import pandas as pd

from core import simulate

df = pd.read_csv("data/processed/hourly_dataset.csv", index_col="timestamp", parse_dates=True)
solar = df["solar_kwh"].to_numpy(float)
demand = df["demand_kwh"].to_numpy(float)
hours = df.index.hour.to_numpy()
n, W = len(df), 168

week_starts = np.arange(0, n - W + 1, W)
test_weeks = week_starts[3::4]
train_weeks = np.array([s for s in week_starts if s not in set(test_weeks.tolist())])
tune_weeks = train_weeks[::4][:10]                     # 10 training weeks spread over the year
train_mask = np.ones(n, dtype=bool)
for s in test_weeks:
    train_mask[s:s + W] = False
typical = pd.Series(demand[train_mask]).groupby(hours[train_mask]).mean().to_numpy()[hours]


def mean_cost(weeks, horizon, terminal_factor):
    return float(np.mean([simulate(solar, demand, hours, "mpc", int(s), W, horizon=horizon,
                                   terminal_factor=terminal_factor, demand_forecast=typical).cost.sum()
                          for s in weeks]))


def objective(trial):
    horizon = trial.suggest_int("horizon", 12, 72, step=6)
    terminal_factor = trial.suggest_float("terminal_factor", 0.0, 1.2)
    cost = mean_cost(tune_weeks, horizon, terminal_factor)
    with mlflow.start_run(run_name=f"trial {trial.number}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("train_mean_weekly_cost_pkr", cost)
    return cost


mlflow.set_experiment("mpc-tuning")
with mlflow.start_run(run_name="optuna study"):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial({"horizon": 48, "terminal_factor": 1.0})      # the hand-picked default
    study.optimize(objective, n_trials=25, show_progress_bar=False)

    default_train = study.trials[0].value
    best = study.best_params
    print(f"\nDefault (48 h, 1.0): {default_train:.0f} PKR/week on tuning weeks")
    print(f"Best {best}: {study.best_value:.0f} PKR/week on tuning weeks")

    test_default = mean_cost(test_weeks, 48, 1.0)
    test_best = mean_cost(test_weeks, best["horizon"], best["terminal_factor"])
    print(f"Held-out weeks: default {test_default:.0f} | tuned {test_best:.0f} PKR/week")
    mlflow.log_params({f"best_{k}": v for k, v in best.items()})
    mlflow.log_metrics({"best_train_cost": study.best_value, "default_train_cost": default_train,
                        "test_cost_default": test_default, "test_cost_tuned": test_best})

with open("data/processed/mpc_tuning.json", "w") as f:
    json.dump({"best_params": best, "train_cost_best": study.best_value, "train_cost_default": default_train,
               "test_cost_default": test_default, "test_cost_tuned": test_best}, f, indent=1)
print("Saved data/processed/mpc_tuning.json")

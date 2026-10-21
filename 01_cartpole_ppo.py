import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./runs")

before, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Before training: {before:.1f}")

model.learn(total_timesteps=30_000)

after, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"After training:  {after:.1f}")   # should approach 500 (the max)
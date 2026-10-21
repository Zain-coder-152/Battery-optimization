import gymnasium as gym

env = gym.make("CartPole-v1")
state, info = env.reset(seed=42)
print("State (what the agent sees):", state)
print("Action space:", env.action_space)

total_reward = 0
done = False
while not done:
    action = env.action_space.sample()          # random action
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print("Random agent total reward:", total_reward)  # usually ~10-40

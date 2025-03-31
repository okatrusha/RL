
# train_dqn.py
from load_data import load_snp_data
from env import TradingEnv
from agent_dqn import DQNAgent
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load and split data
df = load_snp_data("snp500.csv")
train_df = df[(df["Date"] >= "1990-01-01") & (df["Date"] < "2023-01-01")].reset_index(drop=True)
test_df = df[df["Date"] >= "2023-01-01"].reset_index(drop=True)

# Initialize environment and agent
train_env = TradingEnv(train_df, window_size=30)
test_env = TradingEnv(test_df, window_size=30)

state_dim = len(train_env.reset())
action_dim = 3  # Hold, Buy, Sell

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = DQNAgent(state_dim, action_dim, device=device)

print(device)
# Training loop
num_episodes = 500
train_rewards = []

target_update_freq = 10

for episode in range(num_episodes):
    state = train_env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = train_env.step(action)
        if next_state is None:
            next_state = np.zeros_like(state)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    if episode % target_update_freq == 0:
        agent.update_target_model()

    train_rewards.append(total_reward)
    print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

# Evaluate on test set
state = test_env.reset()
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, done = test_env.step(action)
    state = next_state
    total_reward += reward

print("\n✅ TEST ON 2023:")
print(f"Total reward: {total_reward:.2f}")
print(f"Final balance: {test_env.balance:.2f}")

# Plot reward curve
plt.plot(train_rewards)
plt.title("DQN Training Reward Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.show()

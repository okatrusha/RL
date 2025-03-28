from load_data import load_snp_data
from env import TradingEnv
from agent_qlearn import QLearningAgent
import matplotlib.pyplot as plt
import pickle

episodes = 1000
# 1. Завантажуємо і ділимо дані
df = load_snp_data("snp500.csv")
train_df = df[(df["Date"] >= "1990-01-01") & (df["Date"] < "2023-01-01")].reset_index(drop=True)
test_df  = df[df["Date"] >= "2023-01-01"].reset_index(drop=True)

# 2. Навчаємо на train_df
train_env = TradingEnv(train_df)
agent = QLearningAgent()

balances = []
rewards = []

for episode in range(episodes):
    state = train_env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = train_env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    balances.append(train_env.balance)
    rewards.append(total_reward)

    print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, Final balance: {train_env.balance:.2f}")

# 3. Тестуємо на test_df (2023)
test_env = TradingEnv(test_df, window_size=10)
state = test_env.reset()
done = False
test_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, done = test_env.step(action)
    state = next_state
    test_reward += reward

print("\n✅ TEST ON 2023:")
print(f"Total reward: {test_reward:.2f}")
print(f"Final balance: {test_env.balance:.2f}")

# 4. Побудова графіка
plt.figure(figsize=(10, 5))
plt.plot(balances, label="Train Balance")
plt.plot(rewards, label="Train Reward", linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("Q-Learning Training Performance (pre-2023)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

with open("q_table.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

from load_data import load_snp_data
from env import TradingEnv
from agent_qlearn import QLearningAgent

import matplotlib.pyplot as plt

balances = []
rewards = []
agent = QLearningAgent()
env = TradingEnv(load_snp_data("snp500.csv"))
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    final_balance = env.balance
    balances.append(final_balance)
    rewards.append(total_reward)

    print(f"Episode {episode + 1}, Total reward: {total_reward:.2f}, Final balance: {final_balance:.2f}")



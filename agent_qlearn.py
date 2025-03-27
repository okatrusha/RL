import random
import numpy as np
from collections import defaultdict

def zero_q_values():
    return np.zeros(3)  # or len(actions) if dynamic

class QLearningAgent:
    def __init__(self, actions=[0, 1, 2], alpha=0.1, gamma=0.8, epsilon=0.15):
        self.q_table = defaultdict(zero_q_values)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next = 0 if next_state is None else np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * best_next - self.q_table[state][action])

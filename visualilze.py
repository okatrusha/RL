import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# === CONFIG ===
QTABLE_FILE = "q_table.pkl"  # Path to your saved Q-table
NUM_TOP = 10                 # How many top states to show
NUM_ACTIONS = 3              # Hold, Buy, Sell

# === Load Q-table ===
with open(QTABLE_FILE, "rb") as f:
    raw_q_table = pickle.load(f)

# Re-wrap as defaultdict (if needed)
q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS), raw_q_table)

# === Print top states for each action ===
print(f"\n🧠 Top {NUM_TOP} states where agent prefers BUY:")
top_buy = sorted(q_table.items(), key=lambda x: x[1][1], reverse=True)[:NUM_TOP]
for state, q in top_buy:
    print(f"State: {state} → Q(Hold): {q[0]:.2f}, Q(Buy): {q[1]:.2f}, Q(Sell): {q[2]:.2f}")

print(f"\n🧠 Top {NUM_TOP} states where agent prefers SELL:")
top_sell = sorted(q_table.items(), key=lambda x: x[1][2], reverse=True)[:NUM_TOP]
for state, q in top_sell:
    print(f"State: {state} → Q(Hold): {q[0]:.2f}, Q(Buy): {q[1]:.2f}, Q(Sell): {q[2]:.2f}")

# === Count preferred actions across all states ===
action_counts = {0: 0, 1: 0, 2: 0}
for q in q_table.values():
    best_action = int(np.argmax(q))
    action_counts[best_action] += 1

# === Bar chart of preferred actions ===
plt.figure(figsize=(6, 4))
plt.bar(['Hold', 'Buy', 'Sell'], [action_counts[0], action_counts[1], action_counts[2]])
plt.title("Preferred Action Distribution in Learned Q-table")
plt.ylabel("Number of States")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot distribution of Q-values for each action ===
for i, label in enumerate(["Hold", "Buy", "Sell"]):
    q_vals = [q[i] for q in q_table.values()]
    q_vals.sort()
    plt.plot(q_vals, label=label)

plt.title("Sorted Q-values by Action")
plt.xlabel("State Index (sorted)")
plt.ylabel("Q-value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

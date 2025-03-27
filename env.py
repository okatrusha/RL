# ============================================
# 🎮 Action Legend
# 0: Hold  – Do nothing
# 1: Buy   – Open a long position (if flat)
# 2: Sell  – Close a long position (if holding)

# 📦 Position Legend
# 0: Flat  – No open position
# 1: Long  – Holding a long position

# ✅ Valid transitions:
# Position = 0 → Action = 1 (Buy)   → Position becomes 1
# Position = 1 → Action = 2 (Sell)  → Position becomes 0
# Action = 0 (Hold) → no change

# 🚫 Invalid transitions (ignored in logic):
# Position = 0 → Action = 2 (Sell)
# Position = 1 → Action = 1 (Buy)
# ============================================


class TradingEnv:
    def __init__(self, data, window_size=5, initial_balance=1000):
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = self.window_size
        return self._get_state()

    def _get_state(self):
        r = self.data["Returns"].iloc[self.current_step - self.window_size:self.current_step].values
        a = self.data["AbsChange"].iloc[self.current_step - self.window_size:self.current_step].values
        weekday = self.data["Weekday"].iloc[self.current_step]
        return tuple(r.round(2)) + tuple(a.round(1)) + (self.position, weekday)

    def step(self, action):
        price = self.data["Price"].iloc[self.current_step]
        reward = 0

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:
            reward = price - self.entry_price
            self.balance += reward
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.data)
        next_state = self._get_state() if not done else None
        return next_state, reward, done

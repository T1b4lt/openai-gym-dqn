import numpy as np


class QAgent:
    def __init__(self, n_states, actions, qtable=None, exploration_ratio=0.1, learning_rate=0.2, discount_factor=0.9, e_decay_limit=0.05, e_decay_rate=0.01):
        # Set parameters of the agent
        self.exploration_ratio = exploration_ratio
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e_decay_limit = e_decay_limit
        self.e_decay_rate = e_decay_rate
        self.actions = actions

        # Initialize Q-table
        if qtable is not None:
            self.qtable = qtable
        else:
            self.qtable = np.zeros((n_states, actions.n))

    def get_next_step(self, state):
        # Exploration
        if np.random.rand() < self.exploration_ratio:
            action = self.actions.sample()
        # Exploitation
        else:
            random_values = np.random.uniform(
                low=0, high=1, size=(1, self.actions.n))/1000
            action = np.argmax(self.qtable[state]+random_values)
        return action

    def greedy_decay(self):
        # e-greedy decay
        if self.exploration_ratio > self.e_decay_limit:
            self.exploration_ratio -= self.e_decay_rate

    def update_qtable(self, state, action, reward, next_state, done):
        # Update Q-table if not final state
        if not done:
            self.qtable[state, action] = self.qtable[state, action] + self.learning_rate * (
                reward + self.discount_factor * np.max(self.qtable[next_state]) - self.qtable[state, action])
        # Update Q-table if final state
        else:
            self.qtable[state, action] = self.qtable[state, action] + \
                self.learning_rate * (reward - self.qtable[state, action])

    def get_qtable(self):
        return self.qtable

    def print_qtable(self):
        print(self.qtable)

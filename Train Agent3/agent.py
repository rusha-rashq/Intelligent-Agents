from collections import defaultdict

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.1,
    ):
        """
        Initialize a Q-learning agent.

        Parameters:
            actions (List[int]): Possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
            epsilon_decay (float): Rate at which epsilon decays.
            min_epsilon (float): Minimum epsilon value.
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(42)
        self.Q = defaultdict(lambda: np.ones(len(self.actions)))

    def act(self, state, deterministic=False):
        """
        Choose an action using epsilon-greedy policy.

        With probability epsilon, choose a random action.
        Otherwise, choose the best action according to Q-values.

        Args:
            state: The current state.
            deterministic (bool): If True, always choose the best action.

        Returns:
            int: The chosen action.
        """
        if not deterministic and self.rng.random() < self.epsilon:
            action_idx = self.rng.integers(0, len(self.actions))
        else:
            action_idx = int(np.argmax(self.Q[state]))
        return self.actions[action_idx]

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-values using the Q-learning update rule.

        Args:
            state: Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state: State transitioned to.
            done (bool): Whether the episode is done.
        """
        if done:
            best_next_q = 0.0
        else:
            best_next_q = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """
        Implements epsilon decay.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

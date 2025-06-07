from collections import defaultdict

import numpy as np


def reset(size=5):
    """Reset the line-world environment."""
    state = np.random.choice([x for x in range(size) if x != (size // 2)])
    return state


def step(state, action, size=5, goal=None):
    """
    Take action in the line-world environment.
    Actions: -1=left, +1=right
    """
    if goal is None:
        goal = size // 2

    next_state = state + action
    next_state = max(0, min(size - 1, next_state))

    reward = 1.0 if next_state == goal else 0.0
    done = next_state == goal

    return next_state, reward, done


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(len(actions)))

    def act(self, state):
        """Choose the best action based on the Q-values for the given state."""
        action_idx = np.argmax(self.Q[state])
        return self.actions[action_idx], action_idx

    def learn(self, state, action_idx, reward, next_state, done):
        """Update the Q-value for the given state-action pair."""
        if done:
            best_next_q = 0.0
        else:
            best_next_q = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[state][action_idx]
        self.Q[state][action_idx] += self.alpha * td_error


def print_policy(agent, size=7):
    """Print the policy extracted from the Q-table."""
    goal = size // 2

    print("Line-world policy:")
    for position in range(size):
        if position == goal:
            print(f"{'G':^3}", end="")
        else:
            best_action, _ = agent.act(position)
            direction = "←" if best_action == -1 else "→"
            print(f"{direction:^3}", end="")
    print()

    for position in range(size):
        print(f"{position:^3}", end="")
    print("\n")

    print("Q-values:")
    for position in range(size):
        left_q = agent.Q[position][0]
        right_q = agent.Q[position][1]
        print(f"Position {position:2}: [left: {left_q:.2f}, right: {right_q:.2f}]")


def main():
    # Define environment parameters
    size = 7
    goal = size // 2  # Goal is the center position
    actions = [-1, 1]  # Move left or right

    # Create Q-learning agent
    agent = QLearningAgent(actions, alpha=0.1, gamma=0.99)

    # Set training parameters
    num_episodes = 500

    for episode in range(num_episodes):
        state = reset(size=size)
        done = False

        while not done:
            # Choose an action randomly (exploration)
            if np.random.rand() < 0.1:  # 10% chance to explore
                action_idx = np.random.choice(len(actions))
                action = actions[action_idx]
            else:
                action, action_idx = agent.act(state)

            # Take action in the environment
            next_state, reward, done = step(state, action, size=size, goal=goal)

            # Learn from the transition
            agent.learn(state, action_idx, reward, next_state, done)

            # Move to the next state
            state = next_state

    # Display the learned policy
    print_policy(agent, size)


if __name__ == "__main__":
    main()

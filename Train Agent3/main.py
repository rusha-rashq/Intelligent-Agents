import numpy as np
from agent import QLearningAgent
from environment import GridWorldEnv


def run_episode(env, agent, max_steps=100, render=False, deterministic=False):
    """
    Execute a single episode with the given agent and environment.

    Args:
        env (GridWorldEnv): The environment.
        agent (QLearningAgent): The Q-learning agent.
        max_steps (int): Maximum steps (unused if the environment has its own limit).
        render (bool): If True, will render the environment on each step.
        deterministic (bool): If True, the agent acts deterministically.

    Returns:
        tuple: (total_reward, success)
          total_reward (float): Cumulative reward in the episode.
          success (bool): True if the goal was reached, False otherwise.
    """
    state = env.reset()
    total_reward = 0.0
    done = False

    if render:
        print("Initial State:")
        print(f"Agent: {state[0]}, Goal: {state[1]}")
        env.render()

    steps = 0
    while not done and steps < max_steps:
        action = agent.act(state, deterministic=deterministic)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

        if render:
            env.render()

    success = (not info.get("timeout", False)) and (state[0] == state[1])
    return total_reward, success


def train_agent(env, agent, n_episodes=1000):
    """
    Train the agent for a specified number of episodes.

    Args:
        env (GridWorldEnv): The environment.
        agent (QLearningAgent): The Q-learning agent.
        n_episodes (int): Number of episodes to train for.

    Returns:
        tuple: (rewards, successes)
          rewards (list): Rewards for each episode.
          successes (list): Success status for each episode.
    """
    rewards = []
    successes = []
    print(f"Training for {n_episodes} episodes...")
    for ep in range(n_episodes):
        rwd, succ = run_episode(env, agent)
        rewards.append(rwd)
        successes.append(int(succ))
        agent.decay_epsilon()

    print("Training complete.")
    print(f"Average Reward over last 10 episodes: {np.mean(rewards[-10:]):.2f}")
    print(f"Success Rate over last 10 episodes: {np.mean(successes[-10:]):.2f}")

    return rewards, successes


def test_agent(env, agent, n_episodes=10, render=False):
    """
    Test the agent for a specified number of episodes.

    Args:
        env (GridWorldEnv): The environment.
        agent (QLearningAgent): The Q-learning agent to test.
        n_episodes (int): Number of episodes to test for.
        render (bool): Whether to render the environment.

    Returns:
        tuple: (rewards, successes)
          rewards (list): Rewards for each episode.
          successes (list): Success status for each episode.
    """
    rewards = []
    successes = []
    print(f"Testing for {n_episodes} episodes...")
    for ep in range(n_episodes):
        rwd, succ = run_episode(env, agent, render=render, deterministic=True)
        rewards.append(rwd)
        successes.append(int(succ))

    print("Testing complete.")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Success Rate: {np.mean(successes):.2f}")

    return rewards, successes


def main():
    """
    Demonstrate the environment with random goals using a simple Q-learning agent.
    """
    actions = [0, 1, 2, 3]
    agent = QLearningAgent(actions=actions)

    train_env = GridWorldEnv(size=5, min_goal_distance=3)
    train_agent(train_env, agent, n_episodes=1000)

    test_env = GridWorldEnv(size=5, min_goal_distance=3)
    test_agent(test_env, agent, n_episodes=10, render=True)


if __name__ == "__main__":
    main()

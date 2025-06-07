import matplotlib.pyplot as plt
import numpy as np
from agent import QLearningAgent
from environment import GridWorldEnv
from visualization import visualize_policy, visualize_value_function


def train_agent(env, agent, num_episodes=200, eval_interval=20):
    """
    Train the agent for a specified number of episodes.

    Args:
        env: The environment.
        agent: The agent to train.
        num_episodes (int): Number of training episodes.
        eval_interval (int): How often to display progress.

    Returns:
        dict: Training statistics.
    """
    rewards_per_episode = []
    steps_per_episode = []
    success_count = 0

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.act(state)  # Use agent.act instead of choose_action
            next_state, reward, done, info = env.step(action)  # Expect 4 values

            agent.learn(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        if reward > 0:  # Assuming success is indicated by a positive reward
            success_count += 1

        if episode % eval_interval == 0:
            avg_reward = np.mean(rewards_per_episode[-eval_interval:])
            avg_steps = np.mean(steps_per_episode[-eval_interval:])
            success_rate = (success_count / eval_interval) * 100
            print(
                f"[Episode {episode}] Avg Reward: {avg_reward:.2f}, "
                f"Avg Steps: {avg_steps:.2f}, Success Rate: {success_rate:.1f}%"
            )
            success_count = 0

    return {"rewards": rewards_per_episode, "steps": steps_per_episode}


def main():
    """
    Main function to set up the environment and agent, train the agent, and visualize results.
    """
    env_size = 5
    env = GridWorldEnv(size=env_size)

    actions = [0, 1, 2, 3]  # Up, Down, Left, Right
    agent = QLearningAgent(actions=actions, alpha=0.1, gamma=0.99, epsilon=0.1)

    stats = train_agent(env, agent, num_episodes=200, eval_interval=20)

    # Plot rewards
    plt.plot(stats["rewards"])
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

    # Visualize the learned policy and value function
    visualize_policy(agent, env_size)
    visualize_value_function(agent, env_size)


if __name__ == "__main__":
    main()

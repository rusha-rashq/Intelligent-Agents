import numpy as np
from agent import QLearningAgent
from environment import GridWorldEnv
from visualization import plot_training_stats


def train_agent(env, agent, num_episodes=200, eval_interval=20):
    """
    Train the agent for a specified number of episodes.

    Args:
        env (GridWorldEnv): The environment.
        agent (QLearningAgent): The agent to train.
        num_episodes (int): Number of training episodes.
        eval_interval (int): How often to display progress.

    Returns:
        dict: Training statistics.
    """
    rewards_per_episode = []
    steps_per_episode = []
    success_rates = []
    window_size = 10  # For calculating moving average

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Check if episode was successful (not timed out)
        success = not info.get("timeout", False)
        success_rates.append(float(success))
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(env.steps_count)

        # Periodically print progress using a moving average
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(rewards_per_episode[-window_size:])
            avg_steps = np.mean(steps_per_episode[-window_size:])
            avg_success = np.mean(success_rates[-window_size:])
            print(
                f"Episode {episode+1}/{num_episodes} - "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Avg Steps: {avg_steps:.2f}, "
                f"Success Rate: {avg_success:.2f}"
            )

    # Return collected statistics for visualization
    stats = {
        "rewards": rewards_per_episode,
        "steps": steps_per_episode,
        "success_rates": success_rates,
    }
    return stats


def main():
    """Train and evaluate a Q-learning agent."""
    env = GridWorldEnv(size=5)
    actions = [0, 1, 2, 3]
    agent = QLearningAgent(actions)

    print("Training agent...")
    train_stats = train_agent(env, agent, num_episodes=200)

    print("\nTraining complete!")
    print("Final performance (last 10 episodes):")
    print(f"Average Reward: {np.mean(train_stats['rewards'][-10:]):.2f}")
    print(f"Average Steps: {np.mean(train_stats['steps'][-10:]):.2f}")
    print(f"Success Rate: {np.mean(train_stats['success_rates'][-10:]):.2f}")

    # Generate and save training plots
    plot_training_stats(**train_stats)


if __name__ == "__main__":
    main()

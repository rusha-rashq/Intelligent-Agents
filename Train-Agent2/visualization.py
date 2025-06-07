import matplotlib.pyplot as plt
import numpy as np


def visualize_policy(agent, env_size):
    """
    Display the learned policy as arrows in a grid.

    Args:
        agent (QLearningAgent): The agent with a learned policy.
        env_size (int): Size of the environment grid.
    """
    action_symbols = ["↑", "↓", "←", "→"]

    print("Learned Policy:")
    for row in range(env_size):
        line = []
        for col in range(env_size):
            state = (row, col)
            if state == (env_size - 1, env_size - 1):
                line.append("G")  # Goal
            else:
                best_action = agent.get_policy(state)
                line.append(action_symbols[best_action])
        print(" ".join(line))
    print()


def visualize_value_function(agent, env_size):
    """
    Display the state values as a heatmap.

    Args:
        agent (QLearningAgent): The agent with learned Q-values.
        env_size (int): Size of the environment grid.
    """
    plt.figure(figsize=(8, 6))
    value_grid = np.zeros((env_size, env_size))

    for row in range(env_size):
        for col in range(env_size):
            state = (row, col)
            value_grid[row, col] = agent.get_state_value(state)

    plt.imshow(value_grid, cmap="viridis")
    plt.colorbar(label="State Value")
    plt.title("Learned State Values")

    # Add value text in each cell
    for row in range(env_size):
        for col in range(env_size):
            value = value_grid[row, col]
            text_color = "white" if value < 0.5 * np.max(value_grid) else "black"
            plt.text(
                col, row, f"{value:.2f}", ha="center", va="center", color=text_color
            )

    plt.show()


def visualize_policy_and_value(agent, env_size):
    """
    Display the learned policy and state values in a single figure with two subplots.

    One subplot shows the policy as directional arrows and the other shows the value
    function as a heatmap.

    Args:
        agent (QLearningAgent): The agent with learned Q-values.
        env_size (int): Size of the environment grid.
    """
    action_symbols = ["↑", "↓", "←", "→"]

    fig, ax = plt.subplots(figsize=(8, 6))
    value_grid = np.zeros((env_size, env_size))

    for row in range(env_size):
        for col in range(env_size):
            state = (row, col)
            value_grid[row, col] = agent.get_state_value(state)

    im = ax.imshow(value_grid, cmap="viridis")
    plt.colorbar(im, ax=ax, label="State Value")

    for row in range(env_size):
        for col in range(env_size):
            state = (row, col)
            if state == (env_size - 1, env_size - 1):
                symbol = "G"  # Goal state
            else:
                best_action = agent.get_policy(state)
                symbol = action_symbols[best_action]
            text_color = (
                "white" if value_grid[row, col] < 0.5 * np.max(value_grid) else "black"
            )
            ax.text(
                col,
                row,
                symbol,
                ha="center",
                va="center",
                color=text_color,
                fontsize=16,
            )

    ax.set_title("Learned Policy and State Values")
    ax.set_xticks(np.arange(env_size))
    ax.set_yticks(np.arange(env_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    plt.tight_layout()
    plt.show()

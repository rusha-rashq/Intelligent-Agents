import matplotlib.pyplot as plt
import numpy as np


def plot_metric(ax, data, title, xlabel, ylabel, window=10, ylim=None):
    """
    Helper function to plot a metric with raw data and moving average.

    Args:
        ax: Matplotlib Axes object to draw on.
        data (list): Data points to plot.
        title (str): Title of the subplot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        window (int): Window size for moving average.
        ylim (tuple, optional): Y-axis limits.
    """
    ax.plot(data, alpha=0.5, label="Raw")

    if len(data) >= window:
        moving_avg = np.convolve(data, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, window - 1 + len(moving_avg)),
            moving_avg,
            label="Moving Avg",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend()


def plot_training_stats(rewards, steps, success_rates, window=10):
    """
    Plot training progress over episodes using helper function.

    Args:
        rewards (list): Rewards per episode.
        steps (list): Steps per episode.
        success_rates (list): Success indicators per episode.
        window (int): Moving average window size.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_metric(
        ax=axes[0],
        data=rewards,
        title="Rewards per Episode",
        xlabel="Episode",
        ylabel="Total Reward",
        window=window,
    )

    plot_metric(
        ax=axes[1],
        data=steps,
        title="Steps per Episode",
        xlabel="Episode",
        ylabel="Steps",
        window=window,
    )

    plot_metric(
        ax=axes[2],
        data=success_rates,
        title="Success Rate",
        xlabel="Episode",
        ylabel="Success Rate",
        window=window,
        ylim=(-0.05, 1.05),
    )

    fig.tight_layout()
    plt.show()

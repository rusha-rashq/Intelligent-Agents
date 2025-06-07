from pprint import pprint


# Determines the next state given the current state and action.
def step(state, action, grid_size=4):
    row, col = state

    # Calculate the next position based on the action
    if action == 0:  # up
        next_row = max(0, row - 1)
        next_col = col
    elif action == 1:  # down
        next_row = min(grid_size - 1, row + 1)
        next_col = col
    elif action == 2:  # left
        next_row = row
        next_col = max(0, col - 1)
    elif action == 3:  # right
        next_row = row
        next_col = min(grid_size - 1, col + 1)
    else:
        raise ValueError("Invalid action")

    return (next_row, next_col)


# Define a simple reward function for our grid world.
def reward_fn(state, goal_state=(3, 3)):
    return 1.0 if state == goal_state else 0.0


def main():
    # Define all possible states in a 4x4 grid using a list comprehension.
    # Each state is a tuple (row, column) representing a position in the grid.
    states = [(r, c) for r in range(4) for c in range(4)]
    actions = [0, 1, 2, 3]  # 0=up, 1=down, 2=left, 3=right
    action_meanings = {0: "up", 1: "down", 2: "left", 3: "right"}

    print("States:")
    pprint(states)

    print("\nActions:")
    pprint(actions)
    print("\nAction meanings:")
    pprint(action_meanings)

    # Example usage in a simple environment step
    current_state = states[1]  # Agent is somewhere in the grid
    next_state = (2, 2)  # Agent moves down (or in some action sequence)
    reward = reward_fn(next_state)
    print(f"Moving from {current_state} to {next_state}, Reward: {reward}")

    # When the agent reaches the goal
    next_state = (3, 3)  # Goal state
    reward = reward_fn(next_state)
    print(f"Moving to goal {next_state}, Reward: {reward}")


if __name__ == "__main__":
    main()

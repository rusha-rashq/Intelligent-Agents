class GridWorldEnv:
    def __init__(self, size=5):
        self.size = size
        self.state = None
        self.goal_state = (size - 1, size - 1)
        self.action_space = 4
        self.max_steps = size * 3
        self.steps_count = 0

    def reset(self):
        self.state = (0, 0)
        self.steps_count = 0
        return self.state

    def step(self, action):
        self.steps_count += 1
        r, c = self.state
        if action == 0:  # up
            r = max(r - 1, 0)
        elif action == 1:  # down
            r = min(r + 1, self.size - 1)
        elif action == 2:  # left
            c = max(c - 1, 0)
        elif action == 3:  # right
            c = min(c + 1, self.size - 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.state = (r, c)
        reward = 1.0 if self.state == self.goal_state else 0.0
        done = (self.state == self.goal_state) or (self.steps_count >= self.max_steps)
        info = {"timeout": self.steps_count >= self.max_steps}
        return self.state, reward, done, info

    def render(self):
        for row in range(self.size):
            row_elems = []
            for col in range(self.size):
                if (row, col) == self.state:
                    row_elems.append("A")
                elif (row, col) == self.goal_state:
                    row_elems.append("G")
                else:
                    row_elems.append(".")
            print(" ".join(row_elems))
        print()


def run_episode(env, policy):
    """
    Runs a single episode in the Grid World environment by following a given policy.

    Parameters:
        env (GridWorldEnv): The environment instance.
        policy (list): List of actions to follow.

    Returns:
        tuple: (total_reward, steps_taken)
    """
    state = env.reset()
    total_reward = 0.0
    done = False

    print("Initial State:", state)
    env.render()

    for action in policy:
        if done:
            break
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(
            f"Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}"
        )
        env.render()
        state = next_state

    return total_reward, env.steps_count


def main():
    env = GridWorldEnv(size=5)

    # Policy: move down 4 times, then right 4 times to reach (4, 4)
    policy = [1, 1, 1, 1, 3, 3, 3, 3]

    total_reward, steps = run_episode(env, policy)
    print(f"Episode finished in {steps} steps with total reward {total_reward}.")


if __name__ == "__main__":
    main()

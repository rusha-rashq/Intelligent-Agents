class GridWorldEnv:
    def __init__(self, size=5):
        """
        Initialize a simple grid world environment.

        Args:
            size (int): Size of the grid (size x size)
        """
        self.size = size
        self.state = None
        self.goal_state = (size - 1, size - 1)
        self.max_steps = size * 5
        self.steps_count = 0

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            tuple: The starting state.
        """
        self.state = (0, 0)
        self.steps_count = 0
        return self.state

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action (int): The action to take (0=up, 1=down, 2=left, 3=right)

        Returns:
            tuple: (next_state, reward, done, info)
        """
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

        self.state = (r, c)
        reward = 1.0 if self.state == self.goal_state else 0.0
        done = (self.state == self.goal_state) or (self.steps_count >= self.max_steps)
        info = {"timeout": self.steps_count >= self.max_steps}
        return self.state, reward, done, info

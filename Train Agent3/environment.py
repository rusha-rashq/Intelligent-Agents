import numpy as np


class GridWorldEnv:
    def __init__(self, size=5, min_goal_distance=2):
        """
        Environment with a randomly selected goal each episode.
        By default, the agent starts at (0,0).

        Parameters:
            size (int): Size of the grid.
        """
        self.size = size
        self.state = None
        self.goal_state = None
        self.max_steps = size * 3  # Prevent endless episodes
        self.steps_count = 0
        self.rng = np.random.default_rng(42)  # For reproducibility
        self.min_goal_distance = min_goal_distance

    def reset(self):
        """
        Reset the environment to an initial agent state (0,0)
        and randomly sample a new goal position.

        Returns:
            tuple: ((row, col), (goal_row, goal_col))
        """
        self.state = (0, 0)
        self.goal_state = self._sample_random_goal()
        self.steps_count = 0
        # Return both the agent's position and the goal position.
        return (self.state, self.goal_state)

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action (int): 0=up, 1=down, 2=left, 3=right

        Returns:
            tuple: (next_state, reward, done, info)
              next_state = ((row, col), (goal_row, goal_col))
              reward = 1.0 if goal reached, otherwise 0.0
              done = True if goal reached or max steps exceeded
              info = {"timeout": True if max steps exceeded}
        """
        self.steps_count += 1
        r, c = self.state

        # Update position based on action
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

        return (self.state, self.goal_state), reward, done, info

    def render(self):
        """
        Render the environment with the agent (A) and the random goal (G).
        """
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

    def _sample_random_goal(self):
        """
        Randomly sample a goal cell that's not (0, 0) and has a Manhattan distance from (0, 0)
        greater than or equal to min_goal_distance.

        Returns:
            tuple: (goal_row, goal_col)
        """

        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        start = (0, 0)
        possible_positions = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if manhattan_distance(start, (r, c)) >= self.min_goal_distance
        ]
        return possible_positions[self.rng.integers(len(possible_positions))]

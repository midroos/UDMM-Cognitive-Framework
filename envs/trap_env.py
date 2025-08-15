import random

class TrapEnvironment:
    """
    A placeholder for the TrapEnvironment.
    This is a minimal implementation to satisfy the interface required by run.py.
    """
    def __init__(self, size=8):
        self.size = size
        self.agent_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.actions = ["up", "down", "left", "right"]

    def reset(self):
        """Resets the environment to a starting state."""
        self.agent_pos = (0, 0)
        # Returns the state as ((agent_x, agent_y), (goal_x, goal_y))
        return (self.agent_pos, self.goal_pos)

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and done flag.
        This is a dummy implementation.
        """
        # A real implementation would move the agent based on the action.
        # For now, let's assume the agent moves randomly for demonstration.
        x, y = self.agent_pos
        if action == "up":
            y = min(self.size - 1, y + 1)
        elif action == "down":
            y = max(0, y - 1)
        elif action == "right":
            x = min(self.size - 1, x + 1)
        elif action == "left":
            x = max(0, x - 1)

        self.agent_pos = (x, y)

        # Dummy reward and done signal
        reward = -0.1  # Small penalty for each step
        done = (self.agent_pos == self.goal_pos)
        if done:
            reward = 10.0

        next_state = (self.agent_pos, self.goal_pos)

        return next_state, reward, done

    def render(self):
        """Optional: for visualizing the environment."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        gx, gy = self.goal_pos
        ax, ay = self.agent_pos
        grid[gy][gx] = 'G'
        grid[ay][ax] = 'A'
        for row in reversed(grid):
            print(' '.join(row))
        print("-" * (self.size * 2))

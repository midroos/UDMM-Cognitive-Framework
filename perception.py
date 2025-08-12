class Perception:
    def perceive(self, agent_pos, goal_pos):
        # represent state as tuple coords; for Q-table key use flattened index or string
        return agent_pos, goal_pos

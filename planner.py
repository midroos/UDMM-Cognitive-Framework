import random

class Planner:
    def __init__(self, predictive_model, decision_making, actions, planning_depth=4, num_simulations=10):
        """
        Initializes the Planner.
        :param predictive_model: The world model the planner will use for simulation.
        :param decision_making: The decision module to access the Q-table for smarter simulation.
        :param actions: A list of possible actions the agent can take.
        :param planning_depth: How many steps into the future to simulate.
        :param num_simulations: How many random rollouts to perform for each initial action.
        """
        self.predictive_model = predictive_model
        self.decision_making = decision_making
        self.actions = actions
        self.depth = planning_depth
        self.sims = num_simulations

    def plan(self, start_state):
        """
        Performs planning and returns the best action found.
        It evaluates each possible initial action by simulating future outcomes.
        """
        action_values = {}

        for action in self.actions:
            total_sim_reward = 0.0
            # Run multiple simulations for each initial action to get a stable value
            for _ in range(self.sims):
                # Simulate one sequence (rollout)
                sim_reward = self._simulate_sequence(start_state, action)
                total_sim_reward += sim_reward

            action_values[action] = total_sim_reward / self.sims

        # Return the action with the highest average simulated reward
        if not action_values:
            return random.choice(self.actions) # Fallback if no actions evaluated

        best_action = max(action_values, key=action_values.get)
        return best_action

    def _simulate_sequence(self, current_state, initial_action):
        """
        Simulates a single sequence of actions starting with initial_action.
        For steps after the first, it uses the Q-table to choose the best action.
        """
        total_reward = 0.0

        # First step is taken with the initial_action
        next_state, reward = self.predictive_model.sample(current_state, initial_action)
        if next_state is None:
            return 0.0

        total_reward += reward
        current_state = next_state

        # Subsequent steps are guided by the Q-table
        for _ in range(self.depth - 1):
            # Choose the best action based on current Q-values
            state_key = self.decision_making._state_key(current_state)
            q_values = {a: self.decision_making.get_q(state_key, a) for a in self.actions}
            best_action = max(q_values, key=q_values.get)

            next_state, reward = self.predictive_model.sample(current_state, best_action)

            if next_state is None:
                break

            total_reward += reward
            current_state = next_state

        return total_reward

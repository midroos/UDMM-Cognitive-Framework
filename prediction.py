class Prediction:
    """
    A simple predictive model that learns to anticipate the outcomes of actions.
    The model is a table mapping state-action pairs to (predicted_reward, predicted_next_state).
    It uses a string-based key for the state to be consistent with the DecisionMaking module.
    """
    def __init__(self):
        self.model = {}

    def _state_key(self, state_tuple):
        """Creates a consistent, hashable key from a state tuple."""
        (ax, ay), (gx, gy) = state_tuple
        return f"{ax},{ay}|{gx},{gy}"

    def predict(self, state_tuple, action):
        """
        Predicts the reward and next state for a given state-action pair.
        If the pair has not been seen before, it returns a neutral default prediction.
        """
        state_key = self._state_key(state_tuple)
        # Default prediction: 0 reward and no change in state.
        default_prediction = (0.0, state_tuple)
        return self.model.get((state_key, action), default_prediction)

    def update(self, state_tuple, action, reward, next_state_tuple):
        """
        Updates the model with the actual outcome of an action.
        """
        state_key = self._state_key(state_tuple)
        self.model[(state_key, action)] = (reward, next_state_tuple)

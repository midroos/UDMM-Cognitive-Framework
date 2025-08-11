# generative_model.py
import random

class GenerativeModelHierarchy:
    def __init__(self, high_level_intent):
        self.intent = high_level_intent
        
        self.world_model = {
            "proximity_to_mother_means_safety": 0.9,
            "novel_stimulus_is_curious": 0.7 
        }

        self.self_model = {
            "position": {"x": 0, "y": 0},
            "current_affect": "Calm"
        }

    def generate_predictions(self, recent_memory):
        predictions = {}

        if self.intent == "survival_with_family":
            predictions["safety_level"] = self.world_model["proximity_to_mother_means_safety"] * 10
        
        predictions["distance_to_mother"] = 5.0
        
        predictions["my_position"] = self.self_model["position"]

        return predictions

    def update_with_errors(self, errors):
        for key, error_value in errors.items():
            if error_value > 5.0:
                print(f"!! Critical Error in {key}. Updating worldview.")

    def update_self_representation(self, new_data):
        self.self_model.update(new_data)

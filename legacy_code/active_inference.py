# active_inference.py
import random

class ActiveInferenceEngine:
    def compute_free_energy(self, predictions, perceptions):
        errors = {}
        for key in predictions.keys():
            errors[key] = abs(predictions[key] - perceptions.get(key, random.uniform(0, 20)))
        
        total_error = sum(errors.values())
        return total_error, errors

    def generate_affect(self, errors):
        total_error = sum(errors.values())
        
        if total_error > 20:
            return "Afraid"
        elif errors.get("distance_to_mother", 0) > 8:
            return "Anxious"
        elif errors.get("novelty_level", 0) > 5 and total_error < 15:
            return "Curious"
        
        return "Calm"

    def select_action_to_minimize_surprise(self, predictions, errors):
        current_affect = self.generate_affect(errors)

        if current_affect == "Afraid" or current_affect == "Anxious":
            return "MoveTowardMother"
        elif current_affect == "Curious":
            return "ExploreNovelty"
        
        return "MoveTowardsIntent"

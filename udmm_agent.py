# udmm_agent.py
from generative_model import GenerativeModelHierarchy
from active_inference import ActiveInferenceEngine
from perception_memory import PerceptionLayer, MemoryTrace

class Agent:
    def __init__(self, name, age, high_level_intent):
        self.name = name
        self.age = age
        self.generative_model = GenerativeModelHierarchy(high_level_intent)
        self.active_inference_engine = ActiveInferenceEngine()
        self.perception = PerceptionLayer()
        self.memory = MemoryTrace()
        self.time = 0

    def run_simulation(self, current_env_data):
        self.time += 1
        
        predictions = self.generative_model.generate_predictions(self.memory.get_recent_data())

        self.perception.update_perception(current_env_data)

        free_energy, errors = self.active_inference_engine.compute_free_energy(predictions, self.perception.get_data())
        
        affect = self.active_inference_engine.generate_affect(errors)
        self.generative_model.update_self_representation({"current_affect": affect})

        action = self.active_inference_engine.select_action_to_minimize_surprise(predictions, errors)

        self.generative_model.update_with_errors(errors)

        self.memory.store(self.time, self.perception.get_data(), predictions, errors, action, affect)

        self.perform_action(action)

        print(f"[{self.time}] Action: {action}, Affect: {affect}, Free Energy: {free_energy:.2f}")

    def perform_action(self, action):
        print(f"Performing action: {action}")
        
    def get_self_data(self):
        return {"health": 100, "energy": 80}

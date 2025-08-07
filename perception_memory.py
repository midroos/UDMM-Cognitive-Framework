# perception_memory.py

class PerceptionLayer:
    def __init__(self):
        self.current_data = {}

    def update_perception(self, env_data):
        self.current_data = env_data
    
    def get_data(self):
        return self.current_data

class MemoryTrace:
    def __init__(self):
        self.trace = []

    def store(self, time, perception, predictions, errors, action, affect):
        experience = {
            "time": time,
            "perception": perception,
            "predictions": predictions,
            "errors": errors,
            "action": action,
            "affect": affect
        }
        self.trace.append(experience)

    def get_recent_data(self, n=5):
        return self.trace[-n:]

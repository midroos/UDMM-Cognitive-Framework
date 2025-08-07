# main.py
from udmm_agent import Agent
import random
import time

def simulate_environment():
    """محاكاة بسيطة للبيئة"""
    env_data = {
        "distance_to_mother": random.uniform(0, 15),
        "novelty_level": random.uniform(0, 10)
    }
    return env_data

if __name__ == "__main__":
    child_agent = Agent(name="Child", age=11, high_level_intent="survival_with_family")
    print("Starting UDMM agent simulation...")
    
    for i in range(10):
        env_state = simulate_environment()
        child_agent.run_simulation(env_state)
        time.sleep(1)
    
    print("\nSimulation complete.")

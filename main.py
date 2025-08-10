from udmm_agent import UDMM_Agent, Environment
import random

if __name__ == "__main__":
    env = Environment()
    
    print("Welcome to the UDMM Agent V3 simulation! Now with Epsilon-Greedy strategy.")
    
    # We will use a simple epsilon value for now
    epsilon_value = 0.1
    
    for episode in range(10):
        print(f"\n--- Starting Episode {episode + 1} ---")
        
        # Create a new agent for each episode with the epsilon value
        agent = UDMM_Agent(epsilon=epsilon_value)
        env.reset()
        
        goal_pos = env.goal_pos
        goal_state = agent.perception.perceive(goal_pos)
        agent.intention.set_goal(goal_state)
        
        for step in range(30):
            reward, emotion_state, current_pos = agent.step(env)
            
            env.render()
            
            print(f"Step: {step+1} | Position: {current_pos} | Emotion: {emotion_state} | Reward: {reward}")
            
            if reward == 1:
                print("Goal reached! Episode finished.")
                break

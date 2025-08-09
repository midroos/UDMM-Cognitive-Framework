from udmm_agent import UDMM_Agent, Environment
import random

if __name__ == "__main__":
    env = Environment()
    agent = UDMM_Agent()
    
    print("Welcome to the UDMM Agent V2 simulation!")
    
    # Run the simulation for a number of episodes
    for episode in range(10): # We will run for more episodes to see the agent learn
        print(f"\n--- Starting Episode {episode + 1} ---")
        
        env.reset()
        
        # Give the agent a goal
        goal_pos = env.goal_pos
        goal_state = agent.perception.perceive(goal_pos)
        agent.intention.set_goal(goal_state)
        
        # Run for a fixed number of steps within the episode
        for step in range(30): # We will run for more steps to give the agent time to learn
            reward, emotion_state, current_pos = agent.step(env)
            
            env.render()
            
            print(f"Step: {step+1} | Position: {current_pos} | Emotion: {emotion_state} | Reward: {reward}")
            
            if reward == 1:
                print("Goal reached! Episode finished.")
                break

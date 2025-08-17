import os
import json
from trap_env import TrapEnv
from agent.udmm_agent import UDMMAgent
from identity.self_identity_system import WorldManager, SymbolicSelf

def run_experiment(agent, env, num_episodes, run_name):
    print(f"Starting experiment: {run_name} for {num_episodes} episodes.")

    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = f"{log_dir}/progress.jsonl"

    with open(log_file_path, "w") as f:
        f.write("")

    # Initialize symbolic systems
    symbolic_self = SymbolicSelf()
    world_manager = WorldManager()

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        steps = 0
        total_reward = 0

        # Get symbolic info for narrative generation
        best_world, world_conf = world_manager.get_best_for(symbolic_self)

        while not done and steps < 500:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if steps % 50 == 0:
                current_intent = agent.identity.choose_intent({
                    "map_gap": agent.memory.update_map_and_get_gap(state) if agent.memory else 0,
                    "novelty": agent.memory.estimate_novelty(state) if agent.memory else 0,
                    "success_rate": agent._episode_success,
                    "recent_risk": agent._episode_traps,
                })
                narrative_text = agent.narrative_engine.generate(
                    intent=current_intent,
                    identity=agent.identity,
                    confidence=world_conf
                )
                print(f"[{episode+1}] Step {steps}: {narrative_text}")

        agent.end_episode()

        # Update symbolic self with final traits from the episode
        final_traits = agent.identity.self_gap()
        for trait, delta in final_traits.items():
            symbolic_self.update_trait(trait, -delta) # Invert gap to update towards ideal

        log_entry = {
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "success": "Goal Reached" if done else "Max Steps",
            "identity_traits": agent.identity.describe_self(),
            "symbolic_self_nodes": symbolic_self.describe_traits(),
            "narrative": agent.narrative_engine.generate(
                intent=agent.identity.choose_intent({}),
                identity=agent.identity,
                confidence=world_conf
            ),
            "gap_to_ideal": agent.identity.self_gap(),
            "world_fit": {"world": best_world, "confidence": round(world_conf, 3)}
        }

        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Episode {episode+1}/{num_episodes} finished. Total Reward: {total_reward:.2f}, Steps: {steps}")
        print(f"Final Identity at end of episode: {agent.identity.describe_self()}")
        print("-" * 50)

    print(f"Experiment {run_name} completed.")

if __name__ == "__main__":
    actions = ["up", "down", "left", "right"]
    env = TrapEnv(size=10, num_traps=5)
    agent = UDMMAgent(actions=actions)
    run_experiment(agent, env, num_episodes=500, run_name="jules_udmm_self_aware_final")

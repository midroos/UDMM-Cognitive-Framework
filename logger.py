import json
import os

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.step_log_file = open(os.path.join(log_dir, 'steps.jsonl'), 'w')
        self.episode_log_file = open(os.path.join(log_dir, 'episodes.jsonl'), 'w')

    def log_step(self, data):
        """
        Log data for a single step.
        Expected data keys: ep, step, pe, reward, dist_to_goal, emotion,
                             intention, chose_from, schema_id, conf, epsilon
        """
        self.step_log_file.write(json.dumps(data) + '\n')

    def log_episode(self, data):
        """
        Log data for a full episode.
        Expected data keys: success, steps, sum_reward, avg_pe, pe_std,
                             n_consolidated, n_replay_updates
        """
        self.episode_log_file.write(json.dumps(data) + '\n')

    def close(self):
        self.step_log_file.close()
        self.episode_log_file.close()

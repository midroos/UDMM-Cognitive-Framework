import pandas as pd
import matplotlib.pyplot as plt
import json

# Load the data from the JSONL file
log_file_path = "runs/jules_udmm_self_aware_final/progress.jsonl"
with open(log_file_path, 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Function to safely extract values
def safe_get(data, keys):
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return None
    return data

# Extract nested data safely
df['curiosity'] = df['symbolic_self_nodes'].apply(lambda x: safe_get(x, ['curiosity', 'value']))
df['risk_tolerance'] = df['symbolic_self_nodes'].apply(lambda x: safe_get(x, ['risk_tolerance', 'value']))
df['consistency'] = df['symbolic_self_nodes'].apply(lambda x: safe_get(x, ['consistency', 'value']))
df['goal_alignment'] = df['symbolic_self_nodes'].apply(lambda x: safe_get(x, ['goal_alignment', 'value']))

df['ideal_curiosity'] = df['ideal_self'].apply(lambda x: safe_get(x, ['curiosity']))
df['ideal_risk_tolerance'] = df['ideal_self'].apply(lambda x: safe_get(x, ['risk_tolerance']))
df['ideal_consistency'] = df['ideal_self'].apply(lambda x: safe_get(x, ['consistency']))
df['ideal_goal_alignment'] = df['ideal_self'].apply(lambda x: safe_get(x, ['goal_alignment']))

df['gap_curiosity'] = df['gap_to_ideal'].apply(lambda x: safe_get(x, ['curiosity']))
df['gap_risk_tolerance'] = df['gap_to_ideal'].apply(lambda x: safe_get(x, ['risk_tolerance']))
df['gap_consistency'] = df['gap_to_ideal'].apply(lambda x: safe_get(x, ['consistency']))
df['gap_goal_alignment'] = df['gap_to_ideal'].apply(lambda x: safe_get(x, ['goal_alignment']))

df['frustration'] = df['frustration']
df['ambition_patience'] = df['ambition_patience']

# Fill NaN values that may result from safe_get
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)


df['total_gap'] = df[['gap_curiosity', 'gap_risk_tolerance', 'gap_consistency', 'gap_goal_alignment']].sum(axis=1)

# Plotting
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(5, 1, figsize=(15, 24), sharex=True)

# 1. Personality Trait and Ideal Self Evolution
axs[0].set_title('Personality Trait and Ideal Self Evolution', fontsize=16)
axs[0].plot(df['episode'], df['curiosity'], label='Actual Curiosity', color='blue')
axs[0].plot(df['episode'], df['ideal_curiosity'], label='Ideal Curiosity', linestyle='--', color='lightblue')
axs[0].plot(df['episode'], df['risk_tolerance'], label='Actual Risk Tolerance', color='red')
axs[0].plot(df['episode'], df['ideal_risk_tolerance'], label='Ideal Risk Tolerance', linestyle='--', color='lightcoral')
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs[0].set_ylabel('Trait Value')

# 2. Self-Gap Evolution
axs[1].set_title('Self-Gap Evolution', fontsize=16)
axs[1].plot(df['episode'], df['total_gap'], label='Total Self-Gap', color='purple')
axs[1].plot(df['episode'], df['gap_curiosity'], label='Curiosity Gap', linestyle=':')
axs[1].plot(df['episode'], df['gap_risk_tolerance'], label='Risk Tolerance Gap', linestyle=':')
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs[1].set_ylabel('Gap Value')

# 3. Episode Rewards
axs[2].set_title('Episode Rewards', fontsize=16)
axs[2].plot(df['episode'], df['reward'], label='Reward per Episode', color='green')
axs[2].plot(df['episode'], df['reward'].rolling(window=50).mean(), label='50-Episode Moving Average', color='darkgreen')
axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs[2].set_ylabel('Reward')

# 4. Steps per Episode
axs[3].set_title('Steps per Episode', fontsize=16)
axs[3].plot(df['episode'], df['steps'], label='Steps per Episode', color='orange')
axs[3].plot(df['episode'], df['steps'].rolling(window=50).mean(), label='50-Episode Moving Average', color='darkorange')
axs[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs[3].set_ylabel('Steps')

# 5. Frustration and Patience
axs[4].set_title('Frustration and Patience Dynamics', fontsize=16)
axs[4].plot(df['episode'], df['frustration'], label='Frustration', color='purple')
axs[4].plot(df['episode'], df['ambition_patience'] / df['ambition_patience'].max(), label='Patience (Normalized)', linestyle='--', color='gray')
axs[4].axhline(y=0.35, color='r', linestyle=':', label='Frustration Threshold (0.35)')
axs[4].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs[4].set_ylabel('Value')
axs[4].set_xlabel('Episode')


plt.tight_layout()
plt.savefig('final_analysis.png')

print("Analysis complete. Plot saved to final_analysis.png")

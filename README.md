# 🧠 UDMM Cognitive Framework - Agent with Long-Term Memory
*An implementation of a learning agent based on the UDMM concept, now featuring a Long-Term Memory (LTM) system.*

[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--1948--402X-green)](https://orcid.org/0009-0005-1948-402X)
[![GitHub Repo](https://img.shields.io/badge/GitHub-UDMM--Cognitive--Framework-blue?logo=github)](https://github.com/midroos/UDMM-Cognitive-Framework)

---

## 📖 Overview
This project implements a cognitive agent that learns to navigate a grid-world environment. The agent is inspired by the **Unified Dynamic Model of Mind (UDMM)** and uses **Q-learning** alongside a sophisticated **Long-Term Memory (LTM)** system to learn from its experiences, generalize patterns, and make more intelligent decisions.

---

## 🧩 Core Architecture
The agent's architecture integrates standard reinforcement learning with a cognitive memory system.

| Module            | Function |
|-------------------|----------|
| **Environment**   | A simple grid world where the agent operates. |
| **Perception**    | Perceives the agent's current position (state). |
| **Decision Making** | Uses a Q-table, biased by memory, to select actions. |
| **Emotion**       | Simulates emotional states (Joy, Anxiety, Boredom) that influence behavior. |
| **Intention**     | Sets a high-level goal (e.g., Explore, Exploit). |
| **LTM System**    | A new cognitive core composed of Episodic and Semantic memory. |

---

## 🧠 How Memory Works
The LTM system enables the agent to learn from past experiences, both by reinforcing specific memories and by generalizing patterns into abstract rules (schemas).

**The Core Loop:**
1.  **Predict & Act**: The agent makes a prediction about the outcome and chooses an action. The decision can be biased by a relevant `schema` from Semantic Memory.
2.  **Measure PE**: The agent observes the actual outcome and calculates the **Prediction Error (PE)**. This PE is a crucial signal.
3.  **Emotion & Intention**: The PE and reward update the agent's emotional state (e.g., high error causes `Anxiety`, low reward causes `Boredom`). This emotional state, in turn, influences future intentions and behaviors (like the exploration rate).
4.  **Store**: The entire experience `(s, a, r, s', pe)` is stored in `EpisodicMemory`, with the PE determining its priority. High-PE memories are "surprising" and important to learn from.
5.  **Learn (Replay)**: At the end of an episode, the agent performs offline learning by replaying batches of high-priority memories from its episodic buffer.
6.  **Consolidate**: Periodically, the agent consolidates high-error memories into new, generalized `schemas` in `SemanticMemory`. For example, it might learn that "in the top-left corner, moving right is generally a good strategy."

**Schema Example:**
A schema in `runs/<run_name>/memory/mem_ep_X_semantic.json` might look like this:
```json
"schema_18446744073709551615": {
  "precondition": [ 3.5, 0.5, 7.0, 7.0 ],
  "action_model": { "down": 0.8, "right": 0.2 },
  "expected_reward": 5.0,
  "confidence": 0.85,
  "use_count": 0
}
```

---

## ▶️ Reproducing Results
You can run the simulation in a local environment or on a platform like Google Colab.

**Prerequisites:**  
- Python 3
- `numpy`

**Running the Simulation:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/midroos/UDMM-Cognitive-Framework.git
   cd UDMM-Cognitive-Framework
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy
   ```

3. **Run a simulation:**
   ```bash
   # Run with the full LTM system for 200 episodes, with a specific seed
   python run.py --episodes 200 --config full_ltm --seed 42 --run_name "MyFirstRun"
   ```
   You can change the `--config` to `episodic_only` or `no_ltm` to compare performance.

---

## 📂 Artifacts
All outputs from a run are saved to a unique directory to ensure reproducibility.
- **Run Directory**: `runs/<run_name>/`
- **Logs**: Detailed step and episode logs are saved in `runs/<run_name>/logs/`. These are `jsonl` files that can be easily parsed for analysis and plotting.
- **Memory**: If you use the `--save_memory` flag, the agent's memory state is saved periodically to `runs/<run_name>/memory/`.

# 🧠 UDMM Cognitive Framework - Q-Learning Agent
*An implementation of a learning agent based on the UDMM concept.*

[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--1948--402X-green)](https://orcid.org/0009-0005-1948-402X)
[![GitHub Repo](https://img.shields.io/badge/GitHub-UDMM--Cognitive--Framework-blue?logo=github)](https://github.com/midroos/UDMM-Cognitive-Framework)

---

## 📖 Overview
This project implements a cognitive agent that learns to navigate a grid-world environment to reach a goal. The agent is inspired by the **Unified Dynamic Model of Mind (UDMM)** and uses **Q-learning**, a standard reinforcement learning algorithm, to adapt its behavior based on experience.

This version has been refactored to provide a clear, functional, and learning-based implementation.

---

## 🧩 Core Architecture
The agent's architecture is composed of several key modules that work together:

| Module            | Function |
|-------------------|----------|
| **Environment**   | A simple grid world where the agent operates. |
| **Perception**    | Perceives the agent's current position (state). |
| **Memory**        | Stores a history of the agent's experiences (state, action, reward). |
| **Decision Making** | Uses a **Q-table** to select the best action for a given state, balancing exploration and exploitation (epsilon-greedy). |
| **Emotion**       | A simple module that simulates an emotional state based on rewards. |
| **Intention**     | Sets a high-level goal for the agent. |

The agent learns by updating its Q-table after each action, gradually improving its ability to find the most efficient path to the goal.

---

## 💻 Code Structure
The project is organized into two main files:

- **`main.py`**: The entry point for the simulation. It sets up the environment and the agent, runs the training loop, and reports on the agent's performance.
- **`udmm_agent.py`**: Contains the full implementation of the `UDMM_Agent` and its components, including the `Environment`, `DecisionMaking` (with Q-learning), `Memory`, and other modules.
- **`legacy_code/`**: This directory contains unused files from a previous, more complex "Active Inference" design. They are preserved for reference but are not part of the current implementation.

---

## ▶️ Getting Started
You can run the simulation in a local environment or on a platform like Google Colab.

**Prerequisites:**  
- Python 3
- `numpy`

**Running Locally or in a Notebook:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/midroos/UDMM-Cognitive-Framework.git
   cd UDMM-Cognitive-Framework
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy
   ```

3. **Run the simulation:**
   ```bash
   python main.py
   ```

You will see output showing the agent's learning progress over 100 episodes, with the average reward and steps per episode printed every 10 episodes.

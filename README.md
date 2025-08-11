# 🧠 UDMM Cognitive Framework - Version 2
*A Unified Dynamic Model of Mind for Artificial Agents*

[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--1948--402X-green)](https://orcid.org/0009-0005-1948-402X)
[![GitHub Repo](https://img.shields.io/badge/GitHub-UDMM--Cognitive--Framework-blue?logo=github)](https://github.com/midroos/UDMM-Cognitive-Framework)

---

## 📖 Overview
The **UDMM Cognitive Framework** is a novel approach to building artificial agents inspired by the **Unified Dynamic Model of Mind (UDMM)**.
It frames cognition as a dynamic, predictive, and interactive process between an agent's internal generative models and the external environment.

**What's New in Version 2 (V2)?**
In this version, we have transformed the agent from a purely reactive entity into a **learning and adaptive system**. Key updates include:

-   **Upgraded Memory:** The agent now stores complete experiences (state, action, reward).
-   **Functional Learning System:** The agent uses its memory to discover which actions lead to rewards.
-   **Intelligent Decision-Making:** Decisions are no longer random but are guided by the agent's learned knowledge.
-   **More Accurate Predictions:** The prediction module now uses learned information to make more informed predictions about the future.

---

## 🧩 Architecture
The UDMM architecture is divided into **functional modules**, each representing a core cognitive capability:

| Module | Function |
|--------|----------|
| **Perception** | Gathers and interprets sensory inputs |
| **Prediction** | Generates future states and evaluates possible worlds based on learning |
| **Memory** | Stores complete past experiences (state, action, reward) |
| **Learning** | Analyzes experiences to update the agent's internal model |
| **Emotion** | Simulates affective states based on prediction errors |
| **Intention** | Sets and updates goals |
| **Decision Making** | Selects the optimal action based on learning and prediction |
| **Action** | Executes actions in the environment |
| **World Simulator** | Generates and tests possible worlds based on agent and environment dynamics |
| **Time Representation** | Synchronizes the model with reality, allows for recall and forecasting |

---

## 📜 Core Principles
1.  **Predictive Processing** – The agent constantly compares its predicted and actual sensory states.
2.  **Possible Worlds** – The agent simulates multiple potential futures before acting.
3.  **Error Minimization** – Decisions aim to reduce uncertainty and surprise.
4.  **Continuous Synchronization** – The agent keeps its internal model aligned with the real environment.

---

## 💻 Integrated Code
This project provides a simple, self-contained implementation of the UDMM agent. The code is structured into modular classes, allowing you to easily understand and expand on the core concepts.

**File: `udmm_agent.py`**
*(The full code for the agent and its modules is here)*

**File: `main.py`**
*(This file runs the simulation and orchestrates the agent's interaction with the environment)*

---

## ▶️ Getting Started
This guide will walk you through how to run the UDMM agent simulation on Google Colab.

**Prerequisites**
You will need to have **Python 3** and **numpy** installed.

1.  **Open Google Colab**
    Navigate to: [https://colab.research.google.com/](https://colab.research.google.com/)
2.  **Create a new notebook**
    Click `File` then `New notebook`.
3.  **Execute the following commands in a single cell:**
    ```python
    # 1. تثبيت المكتبات المطلوبة
!pip install numpy

# 2. استنساخ المشروع من GitHub
!git clone https://github.com/midroos/UDMM-Cognitive-Framework.git

# 3. الدخول إلى مجلد المشروع
%cd UDMM-Cognitive-Framework

# 4. (اختياري) التحويل إلى فرع التطوير إذا كان موجود
!git checkout v2-development || echo "فرع v2-development غير موجود، سيتم استخدام الفرع الرئيسي"

# 5. تشغيل الكود الأساسي
!python main.py
    ```

**What to expect?**
The simulation will begin and output a visual grid representing the agent's world. You will see the agent move, its emotional state and reward updated at each step. You will notice how the agent's behavior changes over time to become more efficient at reaching its goal, demonstrating how the UDMM architecture drives behavior and adaptation.

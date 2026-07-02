# 🏆 REDROB AI: Intelligent Candidate Ranker

This repository contains the final submission for the **Intelligent Candidate Discovery & Ranking Challenge**.

Our solution abandons brittle Applicant Tracking System (ATS) keyword matching in favor of a **Hybrid Semantic & Deterministic Ranking Pipeline**. It uses AI to understand true capability, while employing a rigorous Python rules engine to enforce business logic, penalize bad behavior, and sidestep explicit hackathon traps.

## 🧠 Architecture Overview

Our final ranking algorithm (`rank_candidates.py`) evaluates 100,000 candidates through a strict multi-stage pipeline:

### Stage 1: Dense Semantic Search (Capability Matching)
We use `all-MiniLM-L6-v2` to map candidates to a 384-dimensional semantic space.
* **Query Optimization:** We engineered a dense vector query focusing strictly on core ML capabilities (RAG, vector retrieval, fine-tuning) to bypass token limits and maximize the signal-to-noise ratio.
* **Candidate Flattening:** We parse skills, experience, and professional summaries to create a high-context string for the embedding model.

### Stage 2: The Deterministic Rules Engine (Heuristics)
We apply hardcoded logic to adjust the baseline semantic score:
* **The "Hidden Gem" Override:** If the AI determines a candidate is highly capable (Semantic Score >= 0.55), they receive a bonus and are *exempt* from strict keyword penalties, capturing high-potential prodigies.
* **The Title Trap:** Candidates with non-technical titles (Marketing, Sales, HR, Recruiter) receive a massive `-0.60` penalty.
* **The Job Hopper Trap:** Candidates with >2 jobs and an average tenure under 18 months are penalized (`-0.30`).
* **The Closed-Source Veteran Trap:** Senior engineers (5+ years) with a missing/zero GitHub activity score are penalized (`-0.25`).
* **Stackable Bonuses:** Candidates possessing explicit "nice-to-have" skills (LoRA, PEFT, XGBoost, etc.) receive cumulative `+0.03` boosts.

### Stage 3: Behavioral Multipliers
A perfect resume is useless if the candidate is inactive. We parse the `redrob_signals` object to multiply the final score based on real-world behavior:
* **Ghosting:** Recruiter response rates < 20% reduce the total score by 90% (`x0.1`).
* **Flaking:** Interview completion rates < 50% reduce the total score by 80% (`x0.2`).
* **Active Seekers:** The `open_to_work` flag applies a 20% boost.

*Note: Final ties are broken by sorting `candidate_id` alphabetically to ensure 100% deterministic reproducibility.*

## 🚀 How to Run the Code

### Prerequisites
1. Ensure you have Python 3.8+ installed.
2. Place your `candidates.jsonl` file in the root directory.

### Installation
Install the required machine learning libraries:
```bash
pip install -r requirements.txt

### 🧪 Live Sandbox / Demo

Want to see the ranking engine in action without installing anything locally? 
You can run our exact pipeline on a sample dataset using our Google Colab environment:

👉 **[Run Technyx Ranker in Google Colab](https://colab.research.google.com/drive/1bbacnUOKvEezPGhCVlUtaRm8_b_am8BJ?usp=sharing)**

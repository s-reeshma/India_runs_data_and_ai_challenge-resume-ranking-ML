# 🏆 Redrob Hackathon: Intelligent Candidate Ranker

This repository contains our submission for the **Intelligent Candidate Discovery & Ranking Challenge**. 

Our solution moves beyond brittle keyword matching by implementing a **Two-Stage Retrieval & Behavioral Ranking Pipeline**. It evaluates candidates based on semantic capability fit and filters out "on-paper perfect" candidates who exhibit poor behavioral signals (keyword stuffers, unengaged users, ghosters).

## 🧠 Methodology: The Two-Stage Ranker

1. **Stage 1: Semantic Filtering (The "Fast Filter")**
   We utilize dense vector embeddings (`all-MiniLM-L6-v2` via `sentence-transformers`) to calculate the cosine similarity between the Job Description and the candidates' flattened career/skills profiles. This catches candidates who have the right experience even if they don't use exact keyword matches.

2. **Stage 2: Behavioral Multipliers (The "Trap Detector")**
   Semantic fit is irrelevant if a candidate is unresponsive. We parse the `redrob_signals` object to apply mathematical penalties for red flags, such as:
   * Low recruiter response rates (<20% heavily penalized)
   * Poor interview completion histories
   * Disengagement (not logging into the platform)
   
   Candidates actively looking for work (`open_to_work_flag`) receive a slight score boost.

## 📂 Repository Structure

* `cand_clas.py` - The core ranking engine (Embeddings + Behavioral logic + CSV export).
* `requirements.txt` - Python dependencies needed to run the script.
* `submission_metadata.yaml` - Challenge submission metadata and declarations.
* `team_name.csv` - The final output of our top 100 ranked candidates.
* `presentation.pdf` - Our 4-slide pitch deck explaining the architecture and reasoning.

## 🚀 How to Run the Code

### Prerequisites
1. Ensure you have Python 3.8+ installed.
2. Download the `candidates.jsonl` file from the Hackathon dataset and place it in the root directory of this project. *(Note: This file is ignored in git due to size).*

### Installation
Install the required machine learning libraries:
```bash
pip install -r requirements.txt

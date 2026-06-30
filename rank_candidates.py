import json
from sentence_transformers import SentenceTransformer,util 
import torch
file_path = r"C:\Users\csekh\OneDrive\Desktop\[PUB] India_runs_data_and_ai_challenge (1)\[PUB] India_runs_data_and_ai_challenge\data\candidates.jsonl"

candidates = []
def flatten_candidate(candidate):
    skills_list = candidate.get('skills', [])
    skill_strings = []
    for skill in skills_list:
        name = skill.get('name', '')
        prof = skill.get('proficiency', '')
        months = skill.get('duration_months', 0)
    skill_strings.append(f"{name} ({prof}, {months} months)")
    skills_text = ", ".join(skill_strings)
    profile = candidate.get('profile', {})
    title = profile.get('current_title', 'Unknown Title')
    exp = profile.get('years_of_experience', 0)
    return f"Title: {title}. Experience: {exp} years. Skills: {skills_text}"
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        candidates.append(json.loads(line))

print(f"Successfully loaded {len(candidates)} candidates!")
print("Flattening data...")
test_candidates = candidates[:1000]
texts_to_embed = [flatten_candidate(c) for c in test_candidates]
print(f"First candidate looks like this to the AI:\n{texts_to_embed[0]}\n")
print("Downloading/Loading Embedding Model (this takes a moment the first time)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating Vector Embeddings (turning text into numbers)...")
candidate_embeddings = model.encode(texts_to_embed)
print(f"Success! Generated {len(candidate_embeddings)} embeddings.")
jd_text = """
Senior AI Engineer. Deep technical depth in modern ML systems — embeddings, 
retrieval, ranking, LLMs, fine-tuning. Scrappy product-engineering attitude. 
Willing to ship a working ranker in a week. Located in or willing to relocate 
to Noida or Pune. Needs real engineering experience, not just AI keywords. 
Must not be a purely non-technical role like Marketing Manager.
"""
print("\nEmbedding Job Description...")
jd_embedding = model.encode(jd_text, convert_to_tensor=True)
candidate_embeddings_tensor = torch.tensor(candidate_embeddings)
print("Calculating Similarity Scores...")
cosine_scores = util.cos_sim(jd_embedding, candidate_embeddings_tensor)[0]
scored_candidates = []
for i, candidate in enumerate(test_candidates): 
    semantic_score = cosine_scores[i].item()
    scored_candidates.append({
        "candidate": candidate,
        "semantic_score": semantic_score
    })
scored_candidates = sorted(scored_candidates, key=lambda x: x['semantic_score'], reverse=True)
top_match = scored_candidates[0]
top_candidate_id = top_match['candidate']['candidate_id']
top_title = top_match['candidate']['profile']['current_title']
top_score = top_match['semantic_score']
print(f"\n--- SEMANTIC SEARCH COMPLETE ---")
print(f"Top match ID: {top_candidate_id}")
print(f"Top match Title: {top_title}")
print(f"Top match Score: {top_score:.4f}")
print("\nApplying Behavioral Traps & Multipliers...")
def calculate_behavioral_multiplier(signals):
    multiplier = 1.0
    response_rate = signals.get('recruiter_response_rate', 0.5)
    if response_rate < 0.20:
        multiplier *= 0.1
    elif response_rate < 0.50:
        multiplier *= 0.6
    interview_completion = signals.get('interview_completion_rate', 1.0)
    if interview_completion < 0.50:
        multiplier *= 0.2
    if signals.get('open_to_work_flag', False):
        multiplier *= 1.2
    recent_searches = signals.get('search_appearance_30d', 1)
    if recent_searches == 0:
        multiplier *= 0.8
    return multiplier
for item in scored_candidates:
    signals = item['candidate'].get('redrob_signals', {})
    behavior_multiplier = calculate_behavioral_multiplier(signals)
    item['final_score'] = item['semantic_score'] * behavior_multiplier
scored_candidates = sorted(scored_candidates, key=lambda x: x['final_score'], reverse=True)
new_top = scored_candidates[0]
print(f"--- BEHAVIORAL FILTER COMPLETE ---")
print(f"New Top match ID: {new_top['candidate']['candidate_id']}")
print(f"New Top Title: {new_top['candidate']['profile']['current_title']}")
print(f"New Top Score: {new_top['final_score']:.4f}")
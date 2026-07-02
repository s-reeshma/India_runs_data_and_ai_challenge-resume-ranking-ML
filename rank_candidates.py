import json
from sentence_transformers import SentenceTransformer, util 
import torch
import csv
import multiprocessing
import pandas as pd

file_path = r"C:\Users\csekh\OneDrive\Desktop\[PUB] India_runs_data_and_ai_challenge (1)\[PUB] India_runs_data_and_ai_challenge\data\candidates.jsonl"

def calculate_explicit_modifiers(candidate, semantic_score):
    modifier = 0.0
    reasons = []
    
    profile = candidate.get('profile', {})
    title = profile.get('current_title', '').lower()
    total_exp = profile.get('years_of_experience', 0)
    
    skills = [s.get('name', '').lower() for s in candidate.get('skills', [])]
    career_history = candidate.get('career_history', [])
    signals = candidate.get('redrob_signals', {})
    
    trap_titles = ['marketing', 'sales', 'hr', 'human resources', 'recruiter', 'accountant', 'content', 'civil', 'mechanical']
    if any(trap in title for trap in trap_titles):
        modifier -= 0.60
        reasons.append("Title Trap Penalty")
        
    is_proven_builder = semantic_score >= 0.55 
    
    vector_dbs = ['pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'faiss']
    embeddings = ['sentence-transformers', 'openai', 'bge', 'e5', 'embeddings', 'retrieval', 'rag']
    
    has_vector_db = any(any(v in s for v in vector_dbs) for s in skills)
    has_embedding = any(any(e in s for e in embeddings) for s in skills)
    
    if is_proven_builder:
        modifier += 0.05
        reasons.append("Semantic Capability Bonus")
    else:
        if not has_vector_db:
            modifier -= 0.15
            reasons.append("Missing Vector DB")
        if not has_embedding:
            modifier -= 0.15
            reasons.append("Missing Embeddings")
            
    bonus_keywords = ['lora', 'qlora', 'peft', 'xgboost', 'learning-to-rank', 'distributed systems', 'open-source']
    for skill in skills:
        if any(bonus in skill for bonus in bonus_keywords):
            modifier += 0.03
            reasons.append(f"Bonus: {skill}")
            
    if len(career_history) > 2:
        total_months = sum(job.get('duration_months', 0) for job in career_history)
        avg_tenure = total_months / len(career_history)
        if avg_tenure < 18:
            modifier -= 0.30
            reasons.append("Job Hopper Penalty")
            
    github_score = signals.get('github_activity_score', -1)
    if total_exp >= 5.0 and github_score == -1:
        modifier -= 0.25
        reasons.append("Closed-Source Veteran Penalty")
        
    return modifier, reasons

def flatten_candidate(candidate):
    skills_list = candidate.get('skills', [])
    skill_strings = []
    name, prof, months = "", "", 0
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

def calculate_behavioral_multiplier(signals):
    multiplier = 1.0
    reasons = []
    
    response_rate = signals.get('recruiter_response_rate', 0.5)
    if response_rate < 0.20:
        multiplier *= 0.1
        reasons.append("Severe Ghosting Penalty")
    elif response_rate < 0.50:
        multiplier *= 0.6
        reasons.append("Ghosting Penalty")
        
    interview_completion = signals.get('interview_completion_rate', 1.0)
    if interview_completion < 0.50:
        multiplier *= 0.2
        reasons.append("Interview Flake Penalty")
        
    if signals.get('open_to_work_flag', False):
        multiplier *= 1.2
        reasons.append("Active Seeker Boost")
        
    recent_searches = signals.get('search_appearance_30d', 1)
    if recent_searches == 0:
        multiplier *= 0.8
        reasons.append("Inactivity Penalty")
        
    return multiplier, reasons

def main():
    candidates = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            candidates.append(json.loads(line))

    print(f"Successfully loaded {len(candidates)} candidates!")
    print("Flattening data...")
    test_candidates = candidates
    texts_to_embed = [flatten_candidate(c) for c in test_candidates]
    print(f"First candidate looks like this to the AI:\n{texts_to_embed[0]}\n")
    
    print("Downloading/Loading Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Configuring multi-processing pool across available CPU cores...")
    num_cores = multiprocessing.cpu_count()
    pool = model.start_multi_process_pool(target_devices=['cpu'] * num_cores)
    
    print(f"Generating Vector Embeddings using {num_cores} cores in parallel (this will be significantly faster)...")
    candidate_embeddings = model.encode_multi_process(texts_to_embed, pool, batch_size=256)
    model.stop_multi_process_pool(pool)
    print(f"Success! Generated {len(candidate_embeddings)} embeddings.")
    
    jd_text = """
    Senior AI Engineer. Deep technical depth in modern ML systems embeddings, 
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
    for item in scored_candidates:
        candidate = item['candidate']
        signals = candidate.get('redrob_signals', {})
        base_score = item['semantic_score']
        
        explicit_modifier, explicit_reasons = calculate_explicit_modifiers(candidate, base_score)
        behavior_multiplier, behavior_reasons = calculate_behavioral_multiplier(signals)
        
        item['final_score'] = (base_score + explicit_modifier) * behavior_multiplier
        
        all_reasons = explicit_reasons + behavior_reasons
        if not all_reasons:
            all_reasons = ["Strong baseline match"]
            
        item['reasoning'] = ", ".join(all_reasons)

    scored_candidates.sort(key=lambda x: (-round(x['final_score'], 4), x['candidate']['candidate_id']))
    new_top = scored_candidates[0]
    print(f"--- BEHAVIORAL FILTER COMPLETE AND TRAPS IDENTIFIED ---")
    print(f"New Top match ID: {new_top['candidate']['candidate_id']}")
    print(f"New Top Title: {new_top['candidate']['profile']['current_title']}")
    print(f"New Top Score: {new_top['final_score']:.4f}")
    
    print("\nGenerating technyx.xlsx...")
    output_filename = 'technyx.xlsx'
    output_rows = []
    for rank_index, item in enumerate(scored_candidates[:100]):
        candidate_id = item['candidate']['candidate_id']
        rank = rank_index + 1
        score = round(item['final_score'], 4)
        score = min(score, 0.9999) 
        reasoning = item.get('reasoning', "Strong baseline match")
        output_rows.append([candidate_id, rank, score, reasoning])

    df = pd.DataFrame(output_rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df.to_excel(output_filename, index=False)

    print(f"\nDone! Top 100 candidates saved to {output_filename}. Tie-breaker applied.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
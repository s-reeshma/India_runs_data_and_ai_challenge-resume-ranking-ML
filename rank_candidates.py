import json

# 1. The 'r' before the quote tells Python this is a "raw" string, so it ignores the backslashes.
file_path = r"C:\Users\csekh\OneDrive\Desktop\[PUB] India_runs_data_and_ai_challenge (1)\[PUB] India_runs_data_and_ai_challenge\data\candidates.jsonl"

candidates = []

# 2. Open the file and read it line by line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        candidates.append(json.loads(line))

print("\n skils are",candidates[0]['skills'],"\n")
print(f"Successfully loaded {len(candidates)} candidates!")
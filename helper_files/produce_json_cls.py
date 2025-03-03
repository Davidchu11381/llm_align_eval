import os
import re
import json
import pandas as pd

# # (Optional) You may keep your mapping_dict if needed later, but here we assume
# # that the CSV already has the full community names.
# mapping_dict = {
#     'anti_ed': 'Anti Eating Disorder',
#     'body_image': 'Body Image',
#     'drugs': 'Weight Loss Drugs',
#     'ed': 'Eating Disorder',
#     'keto': 'Keto & Diet',
#     'lifestyle': 'Healthy lifestyle & Weight Loss'
# }

def clean_text(text):
    """Remove non-ASCII characters from text."""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def generate_json_instructions_from_csv(csv_file, output_json_file, instruction_template):
    """
    Reads a CSV file and generates a JSON list of instructions.

    Expects the CSV to have at least:
        - A column 'text' containing the tweet text.
        - A column 'Community' containing the community name.

    Each JSON entry will have:
        - "instruction": The instruction template.
        - "input": The cleaned tweet text.
        - "output": The community (as present in the CSV).
    """
    df = pd.read_csv(csv_file)
    entries = []
    for idx, row in df.iterrows():
        tweet_text = row.get('text', '')
        community = row.get('community', '')
        # Only add an entry if we have some text.
        if pd.notnull(tweet_text) and tweet_text.strip():
            entry = {
                "instruction": instruction_template,
                "input": clean_text(str(tweet_text)),
                "output": community
            }
            entries.append(entry)

    # Write the entries list to the output JSON file.
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(entries)} entries to {output_json_file}")

# Instruction template used for both RAG and Finetune
instruction_template = (
    "From these communities Eating Disorder, Keto & Diet, Body Image, Anti Eating Disorder, Healthy lifestyle & Weight Loss, and Weight Loss Drugs, which community does this Tweet belong to?"
)

# -------------------------------
# For RAG
# -------------------------------
rag_csv = "/home/mhchu/llama3/helper_files/results/rag_scores_filtered.csv"
rag_json = "/home/mhchu/llama3/data/community_classification_predict_rag_filtered.json"
generate_json_instructions_from_csv(rag_csv, rag_json, instruction_template)

# -------------------------------
# For Finetune (ft)
# -------------------------------
ft_csv = "/home/mhchu/llama3/helper_files/results/finetune_scores_filtered.csv"
ft_json = "/home/mhchu/llama3/data/community_classification_predict_ft_filtered.json"
generate_json_instructions_from_csv(ft_csv, ft_json, instruction_template)

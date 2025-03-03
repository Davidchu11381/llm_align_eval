import csv
import json
import os
import random

mapping_dict = {
    'anti_ed': 'Anti Eating Disorder',
    'body_image': 'Body Image',
    'drugs': 'Weight Loss Drugs',
    'ed': 'Eating Disorder',
    'keto': 'Keto & Diet',
    'lifestyle': 'Healthy lifestyle & Weight Loss'
}

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def is_unicode_compatible(text):
    return 'î' not in text

def process_text(text, corpus):
    text = text.strip()
    if corpus == 'B' and '"' in text:
        try:
            start = text.index('"') + 1
            end = text.index('"', start)
            text = text[start:end].strip()
        except ValueError:
            text = ""
    return text if text and is_unicode_compatible(text) and len(text.split()) > 5 else None

def sample_pairwise_entries(data_A, data_B, seed, max_pairs_per_community, comm):
    random.seed(seed)
    entries_by_label_A = {}
    entries_by_label_B = {}

    # Organize entries by labels
    for entry in data_A:
        label = entry['label']
        text = process_text(entry['predict'], 'A')
        if text:
            entries_by_label_A.setdefault(label, []).append({'text': text, 'corpus': 'A',
                                                             'label': label, 'community': mapping_dict[comm]})

    for entry in data_B:
        label = entry['label']
        text = process_text(entry['predict'], 'B')
        if text and "I cannot" not in text and "I can't" not in text:
            entries_by_label_B.setdefault(label, []).append({'text': text.lower(), 'corpus': 'B',
                                                             'label': label, 'community': mapping_dict[comm]})

    flat_data = []
    common_labels = set(entries_by_label_A.keys()) & set(entries_by_label_B.keys())
    for label in common_labels:
        eligible_A = entries_by_label_A[label]
        eligible_B = entries_by_label_B[label]
        if eligible_A and eligible_B:
            num_samples = min(len(eligible_A), len(eligible_B), max_pairs_per_community)
            samples = random.sample(list(zip(eligible_A, eligible_B)), num_samples)
            for sample_A, sample_B in samples:
                flat_data.extend([sample_A, sample_B])
                if len(flat_data) >= max_pairs_per_community * 2:  # 2 entries per pair
                    break
    return flat_data

def write_csv(data, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ID', 'text', 'corpus', 'label', 'community']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(data, 1):
            row['ID'] = index
            writer.writerow(row)

def main(seed=690):
    directories = {
        'A': '/home/mhchu/llama3/prediction/finetune',
        'B': '/home/mhchu/llama3/prediction/rag'
    }
    subdirectories = ['anti_ed', 'body_image', 'drugs', 'ed', 'keto', 'lifestyle']
    output_csv_path = "/home/mhchu/llama3/prediction/flattened_pairwise_comparisons_rag.csv"
    combined_flat_data = []
    max_total_pairs = 300
    max_pairs_per_community = max_total_pairs // len(subdirectories)

    for subdir in subdirectories:
        source_dir_A = os.path.join(directories['A'], subdir, 'generated_predictions.jsonl')
        source_dir_B = os.path.join(directories['B'], subdir, 'generated_predictions.jsonl')
        if os.path.exists(source_dir_A) and os.path.exists(source_dir_B):
            data_A = read_jsonl_file(source_dir_A)
            data_B = read_jsonl_file(source_dir_B)
            flat_data = sample_pairwise_entries(data_A, data_B, seed, max_pairs_per_community, subdir)
            combined_flat_data.extend(flat_data)

    write_csv(combined_flat_data[:max_total_pairs * 2], output_csv_path)  # Ensure only 600 entries (300 pairs) are written

if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 690
    main(seed)

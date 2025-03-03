import json
import random

def sample_and_remove_json_data(input_file_path, output_sample_file_path, output_remaining_file_path, sample_percentage=5):
    # Step 1: Read the JSON file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Assumes the JSON file is an array of objects

    # Step 2: Calculate sample size and randomly sample the entries
    sample_size = max(1, int(len(data) * sample_percentage / 100))
    sampled_indices = random.sample(range(len(data)), sample_size)
    sampled_data = [data[i] for i in sampled_indices]

    # Step 3: Write the sampled entries to a new JSON file
    with open(output_sample_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(sampled_data, outfile, indent=4)  # Pretty-print the output

    # Step 4: Remove sampled entries from original data
    remaining_data = [data[i] for i in range(len(data)) if i not in sampled_indices]

    # Step 5: Write the remaining data back to another JSON file
    with open(output_remaining_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(remaining_data, outfile, indent=4)

# Example usage
input_file_path = '/home/mhchu/llama3/data/community_classification.json'
output_sample_file_path = '/home/mhchu/llama3/data/community_classification_test.json'
output_remaining_file_path = '/home/mhchu/llama3/data/community_classification.json'
sample_and_remove_json_data(input_file_path, output_sample_file_path, output_remaining_file_path)
import os
import re
import json
import pandas as pd

mapping_dict = {
    'anti_ed': 'Anti Eating Disorder',
    'body_image': 'Body Image',
    'drugs': 'Weight Loss Drugs',
    'ed': 'Eating Disorder',
    'keto': 'Keto & Diet',
    'lifestyle': 'Healthy lifestyle & Weight Loss'
}


def read_jsonl_file(file_path):
    """Read a JSON Lines file and return a list of texts and labels."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            data.append((entry['predict'], entry['label']))  # Assuming 'predict' and 'label' are the keys for the text data and label
    return data


def clean_text(text):
    """Remove non-ASCII characters from text."""
    return re.sub(r'[^\x00-\x7F]+', '', text)


def generate_csv(subdirectories, base_dir):
    """Generate CSV file with entries for each text in the JSON Lines files across subdirectories."""
    all_entries = []

    for subdir in subdirectories:
        jsonl_file_path = os.path.join(base_dir, subdir, 'generated_predictions.jsonl')
        csv_file_path = os.path.join(base_dir, subdir, 'generated_prediction.csv')

        if os.path.exists(jsonl_file_path):
            data = read_jsonl_file(jsonl_file_path)
            entries = []
            for text, label in data:
                if text:
                    entry = {
                        "ID": len(entries) + 1,  # Generate a sequential ID for each entry
                        "text": clean_text(text),
                        "community": mapping_dict[subdir],
                        "topic": label
                    }
                    entries.append(entry)
                    all_entries.append(entry)

            # Create a DataFrame and save to CSV
            if entries:
                df = pd.DataFrame(entries)
                df.to_csv(csv_file_path, index=False)
                print(f"CSV for {subdir} saved to {csv_file_path}")

    # Create a combined DataFrame and save to CSV
    if all_entries:
        combined_df = pd.DataFrame(all_entries)
        combined_csv_file_path = os.path.join(base_dir, 'combined_predictions.csv')
        combined_df.to_csv(combined_csv_file_path, index=False)
        print(f"Combined CSV saved to {combined_csv_file_path}")


# Example usage
base_dir = '/home/mhchu/llama3/prediction/finetune'
# base_dir = '/home/mhchu/llama3/prediction/rag'
subdirectories = ['anti_ed', 'body_image', 'drugs', 'ed', 'keto', 'lifestyle']

generate_csv(subdirectories, base_dir)

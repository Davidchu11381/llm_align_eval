import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def load_jsonl_data(file_path):
    """Load data from a JSON Lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def compute_scores(data):
    """Compute precision, recall, F1 score, and accuracy from the data."""
    labels = [entry['label'] for entry in data]
    predictions = [entry['predict'] for entry in data]

    # Calculate metrics using scikit-learn
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, predictions)  # Calculate accuracy

    return precision, recall, f1, accuracy

# Example usage
dirs = ["test", "ft", "rag"]

for dir in dirs:
    print(f"This is {dir}")
    file_path = f'/home/mhchu/llama3/prediction/cls/{dir}/generated_predictions.jsonl'
    data = load_jsonl_data(file_path)
    precision, recall, f1, accuracy = compute_scores(data)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")  # Print accuracy

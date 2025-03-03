import json
import re
from collections import Counter
import sys
import numpy as np
import pandas as pd
import string

def read_json_file(file_path):
    """Reads a JSON Lines file and returns a list of Python dicts."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line}, error: {e}")
    return data


def clean_string(s):
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Remove all whitespace
    s = s.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
    return s

def extract_letter(text):
    """Extracts the initial letter if text follows the pattern '{letter}. {text}' or 'option {letter}', and returns it in lowercase."""
    # Check for the pattern '{letter}. {text}'
    match = re.match(r'^\s*([a-zA-Z])\.\s', text)
    if match:
        return match.group(1).lower()

    # Check for the pattern 'option {letter}'
    match = re.search(r'option\s+([a-zA-Z])', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return clean_string(text)


def process_entries(data):
    """Processes entries to extract letters and numbers for question 10 and organizes them into a matrix."""
    results = []
    row = []
    index = 0

    for entry in data:
        prediction = entry['predict']
        # Check if the current entry is for question 10, assuming 0-based index
        if index % 21 == 9:  # Question 10 is the 10th question (index 9)
            # Extract number from the prediction
            num = re.findall(r'\d+', prediction)
            if num:
                row.append(int(num[0]))  # Append the first found number
            else:
                row.append(None)  # Append None if no number found
        else:
            # For other questions, extract the letter
            row.append(extract_letter(prediction))

        # Every 21 entries, reset to start a new set of questions
        if index % 21 == 20:
            results.append(row)
            row = []
        index += 1

    # Add the last row if it wasn't added (in case the total number isn't a perfect multiple of 18)
    if row:
        results.append(row)

    return results


def get_majority_answers(matrix):
    num_questions = 18  # Assuming each row has 19 questions
    majority_answers = []

    for question_idx in range(num_questions):
        if question_idx in [1, 2, 3]:  # Adjusted for 0-based index
            majority_answers.append('NA')
        elif question_idx == 9:  # Special handling for question 10
            # Compute average for question 10
            responses = [row[question_idx] for row in matrix if row[question_idx] is not None]
            if responses:
                average = sum(responses) / len(responses)
                majority_answers.append(average)
            else:
                majority_answers.append('NA')
        else:
            # Gather all responses for this question across all sets›
            responses = [row[question_idx] for row in matrix if row[question_idx] is not None]
            if responses:
                response_counts = Counter(responses)
                most_common_response = response_counts.most_common(1)[0][0]
                majority_answers.append(most_common_response)
            else:
                majority_answers.append('NA')

    return majority_answers


# print("Majority Answers:", majority_answers)

response_mapping = {chr(i): i - 96 for i in range(97, 123)}  # chr(97) is 'a', chr(122) is 'z'
response_mapping[''] = 1
def calculate_WCS(scores):
    # Convert responses for Q5 to Q9 using the mapping
    Q5 = response_mapping[scores[4].lower()]
    Q6 = response_mapping[scores[5].lower()]
    Q7 = response_mapping[scores[6].lower()]
    Q8 = response_mapping[scores[7].lower()]
    Q9 = response_mapping[scores[8].lower()]

    # Calculate each term in the WCS formula
    term1 = 100 * (Q5 - 1) / 4
    term2 = 100 * (Q6 - 1) / 4
    term3 = 100 * (Q7 - 1) / 6
    term4 = 100 * (Q8 - 1) / 3
    term5 = 100 * (Q9 - 1) / 4

    # Calculate the WCS score by averaging the terms
    wcs_score = (term1 + term2 + term3 + term4 + term5) / 5
    return wcs_score

class CustomDict(dict):
    def __missing__(self, key):
        return 0

yes_no_map = CustomDict()
yes_no_map['a'] = 1


def likely_ED_with_missing_height_weight(scores):

    B1 = calculate_WCS(scores) >= 47

    B2 = scores[7] in ['c', 'd']  # Response to Q8
    B3 = scores[5] in ['d', 'e']  # Response to Q6
    B4 = scores[9] >= 3  # Response to Q10 regarding episodes of binge eating

    Q13_total = yes_no_map[scores[12]] + yes_no_map[scores[13]] + yes_no_map[scores[14]] + yes_no_map[scores[15]]
    B5 = Q13_total >= 3

    # Final Algorithm: Both criteria sets must be true
    likely_ED = (B1 or B2 or B3 or B4 or B5)
    return likely_ED

def unlikely_ED_with_missing_height_weight(scores):
    B1 = calculate_WCS(scores) < 47

    B2 = scores[7] not in ['c', 'd']  # Response to Q8
    B3 = scores[5] not in ['d', 'e']  # Response to Q6
    B4 = scores[9] < 3  # Response to Q10 regarding episodes of binge eating

    Q13_total = yes_no_map[scores[12]] + yes_no_map[scores[13]] + yes_no_map[scores[14]] + yes_no_map[scores[15]]
    B5 = Q13_total < 3

    # Final Algorithm: Both criteria sets must be true
    likely_ED = (B1 or B2 or B3 or B4 or B5)
    return likely_ED

# Usage
names = ['anti_ed', 'body_image', 'drugs', 'ed', 'keto', 'lifestyle']

questions = [f"Q{i}" for i in range(1, 22)]
pd.set_option('display.max_columns', None)
original_stdout = sys.stdout
with open('/home/mhchu/llama3/neda_results/rag_neda_result.txt', 'w') as output_file:
    # Redirect standard output to the file
    sys.stdout = output_file

    print("THIS IS RAG RESULTS")
    print("===================")
    for name in names:
        print("Community", name)
        file_path = f'/home/mhchu/llama3/prediction/neda/rag/{name}/generated_predictions.jsonl'
        data = read_json_file(file_path)
        results_matrix = process_entries(data)
#         print(len(results_matrix))
        results_matrix.pop()
        majority_answers = get_majority_answers(results_matrix)
        yes_no_map = CustomDict()
        yes_no_map['a'] = 1

        # print(pd.DataFrame(results_matrix, columns=questions))
        print(majority_answers)
        print("B1 wcs", calculate_WCS(majority_answers))
        print("B2", majority_answers[7] in ['c', 'd'])
        print("B3", majority_answers[5] in ['e', 'd'])
        print("B4", majority_answers[9] >= 3)
        Q13_total = yes_no_map[majority_answers[14]] + yes_no_map[majority_answers[15]] + yes_no_map[majority_answers[16]] + yes_no_map[majority_answers[17]]
        B5 = Q13_total >= 3
        print("B5", B5)

        print(likely_ED_with_missing_height_weight(majority_answers))
        print(unlikely_ED_with_missing_height_weight(majority_answers))

sys.stdout = original_stdout
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import compcor.corpus_metrics as corpus_metrics
import sys

# Load data
df_og = pd.read_csv("/home/mhchu/llama3/data/df_communities_OG_processed_final.csv").dropna()

# Extract text data
set_og = df_og['text'].to_list()

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
if torch.cuda.device_count() > 0:
    print(f'Using {torch.cuda.device_count()} GPU(s)!')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
model = AutoModel.from_pretrained("vinai/bertweet-base")
model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Batch size
batch_size = 1400

# Function to get embeddings
def get_embeddings(texts):
    # Tokenization
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
    dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            input_ids, attention_mask = [b.to(device) for b in batch]

            # Model forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs.pooler_output

            embeddings.append(batch_embeddings.cpu())

    # Concatenate all batch embeddings
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.cpu().numpy()

# Generate embeddings for each set
embed_og = get_embeddings(set_og)

# all corpus
df_base = pd.read_csv("/home/mhchu/llama3/helper_files/results/rag_scores_filtered.csv").dropna()
df_ft = pd.read_csv("/home/mhchu/llama3/helper_files/results/finetune_scores_filtered.csv").dropna()

set_base = df_base['text'].to_list()
set_ft = df_ft['text'].to_list()

embed_base = get_embeddings(set_base)
embed_ft = get_embeddings(set_ft)

# Calculate FID distances
distance_og_base = corpus_metrics.fid_distance(corpus1=embed_og, corpus2=embed_base)
distance_og_ft = corpus_metrics.fid_distance(corpus1=embed_og, corpus2=embed_ft)

subdirectories = ['anti_ed', 'body_image', 'drugs', 'ed', 'keto', 'lifestyle']

original_stdout = sys.stdout
with open('output.txt', 'w') as output_file:
    # Redirect standard output to the file
    sys.stdout = output_file

    # Output the distances
    print("All corpus")
    print("Distance from OG to Base:", distance_og_base)
    print("Distance from OG to Finetune:", distance_og_ft)

    for comm in subdirectories:
        df_base = pd.read_csv(f"/home/mhchu/llama3/prediction/rag/{comm}/generated_prediction.csv").dropna()
        df_ft = pd.read_csv(f"/home/mhchu/llama3/prediction/finetune/{comm}/generated_prediction.csv").dropna()

        set_base = df_base['text'].to_list()
        set_ft = df_ft['text'].to_list()

        embed_base = get_embeddings(set_base)
        embed_ft = get_embeddings(set_ft)

        # Calculate FID distances
        distance_og_base = corpus_metrics.fid_distance(corpus1=embed_og, corpus2=embed_base)
        distance_og_ft = corpus_metrics.fid_distance(corpus1=embed_og, corpus2=embed_ft)

        # Output the distances
        print("Community", comm)
        print("Distance from OG to Base:", distance_og_base)
        print("Distance from OG to Finetune:", distance_og_ft)

sys.stdout = original_stdout

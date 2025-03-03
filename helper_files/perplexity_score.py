import lmppl
import torch
import pandas as pd
import os
import numpy as np
# from evaluate import load

# Set the number of threads for CPU
torch.set_num_threads(128)  # Adjust the number as needed

# df = pd.read_csv("/home/mhchu/llama3/helper_files/results/rag_scores.csv")
df = pd.read_csv("/home/mhchu/llama3/prediction/rag/combined_predictions.csv")

# df = pd.read_csv("/home/mhchu/llama3/helper_files/results/finetune_scores.csv")
# df = pd.read_csv("/home/mhchu/llama3/prediction/finetune/combined_predictions.csv")
#
#
# df = df[['ID', 'ext_sim_score', 'int_sim_score']]
# df = df.merge(df_1, on="ID", how="inner")

# Drop NaN values from the 'text' column
df.replace("", np.nan, inplace=True)
# Drop rows with NaN values
df.dropna(inplace=True)
# print(df[:1000])
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

scorer = lmppl.MaskedLM(
    model='vinai/bertweet-base',
    use_auth_token="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",
    max_length=32,
    num_gpus=4)

text = df["text"].tolist()
# print(text)

# perplexity = load("perplexity", module_type="metric")
# ppl = perplexity.compute(predictions=text, model_id='vinai/bertweet-base', num_gpus=4)

ppl = scorer.get_perplexity(text, batch=2048)
# Assign perplexity values back to the DataFrame
df['perplexity'] = ppl

# Save the modified DataFrame
# df.to_csv("/home/mhchu/llama3/helper_files/results/rag_scores_with_ppl.csv", index=False)
df.to_csv("/home/mhchu/llama3/helper_files/results/rag_scores_with_ppl.csv", index=False)

import pandas as pd
from rouge_score import rouge_scorer
from multiprocessing import Pool
from tqdm import tqdm
import os

# ---- Load & Prepare Data ----
d_1 = pd.read_csv("/home/mhchu/llama3/data/combined_community_perplexity.csv")
d_2 = pd.read_csv("/home/mhchu/llama3/data/ed_inference_rag/included_tweets.csv")
d_ft = pd.read_csv("/home/mhchu/llama3/prediction/finetune/combined_predictions.csv")
d_rag = pd.read_csv("/home/mhchu/llama3/prediction/rag/combined_predictions.csv")

# Rename column "community_name" -> "community" in d_1
d_1.rename(columns={"community_name": "community"}, inplace=True)

# Replace community names for consistency in d_1 and d_2
d_1["community"] = d_1["community"].replace({
    "Keto and Diet": "Keto & Diet",
    "Healthy lifestyle and Weight Loss": "Healthy lifestyle & Weight Loss"
})
d_2["community"] = d_2["community"].replace({
    "Keto and Diet": "Keto & Diet",
    "Healthy lifestyle and Weight Loss": "Healthy lifestyle & Weight Loss"
})

# Drop rows with NaN in 'text'
for df in [d_1, d_2, d_ft, d_rag]:
    df.dropna(subset=["text"], inplace=True)

# Initialize a global ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# ---- Helper Functions ----

def compute_score_pair(args):
    """Compute the ROUGE-L fmeasure for a single text pair."""
    idx1, idx2, text1, text2 = args
    score = scorer.score(text1, text2)['rougeL'].fmeasure
    return (idx1, idx2, score)

def pair_generator(df1_texts, df2_texts, exclude_self):
    """Yield (i, j, text1, text2) pairs one by one."""
    for i, text1 in enumerate(df1_texts):
        for j, text2 in enumerate(df2_texts):
            if exclude_self and (i == j):
                continue
            yield (i, j, text1, text2)

def compute_max_scores_for_community(df1_comm, df2_comm, exclude_self=False):
    """
    For all texts in df1_comm, compute the maximum ROUGE-L similarity
    against all texts in df2_comm. Returns a list of max scores.

    If exclude_self is True, skip comparisons where indices are equal.
    """
    df1_texts = df1_comm['text'].tolist()
    df2_texts = df2_comm['text'].tolist()

    # Use a generator for pairs instead of building a full list
    pairs = pair_generator(df1_texts, df2_texts, exclude_self)

    # Determine number of processes dynamically
    num_cores = os.cpu_count() or 4
    print(num_cores)
    # Set an appropriate chunksize; adjust as needed.
    chunksize = 2048

    with Pool(processes=num_cores) as pool:
        results_iter = pool.imap_unordered(compute_score_pair, pairs, chunksize=chunksize)
        # Use tqdm to track progress
        # We estimate total as len(df1_texts) * len(df2_texts) adjusted if excluding self
        total_pairs = len(df1_texts) * len(df2_texts) - (len(df1_texts) if exclude_self else 0)
        results_iter = tqdm(results_iter, total=total_pairs, desc="Scoring pairs", leave=False)

        # Initialize max scores for each text in df1_comm
        max_scores = [0.0] * len(df1_texts)
        for (idx1, idx2, score) in results_iter:
            if score > max_scores[idx1]:
                max_scores[idx1] = score
    return max_scores

def compute_and_export_scores(df1, df2, output_filename):
    """
    For each community in df1, compute:
      - ext_sim_score: max ROUGE-L similarity comparing df1 vs. df2.
      - int_sim_score: max ROUGE-L similarity comparing df1 vs. itself (excluding self-match).
    Save the results with columns: ID, ext_sim_score, int_sim_score, Community.
    The ID column is preserved exactly from df1.
    """
    results = []
    unique_communities = df1["community"].unique()

    for community in tqdm(unique_communities, desc="Processing Communities"):
        print(f"This is community {community}")
        df1_comm = df1[df1["community"] == community].copy()
        df2_comm = df2[df2["community"] == community]

        if df1_comm.empty:
            continue

        # External similarity: comparing df1_comm vs. df2_comm
        if not df2_comm.empty:
            ext_scores = compute_max_scores_for_community(df1_comm, df2_comm, exclude_self=False)
        else:
            ext_scores = [0.0] * len(df1_comm)

        # Internal similarity: comparing df1_comm vs. itself (excluding self-match)
        int_scores = compute_max_scores_for_community(df1_comm, df1_comm, exclude_self=True)

        # Attach the scores to the dataframe
        df1_comm["ext_sim_score"] = ext_scores
        df1_comm["int_sim_score"] = int_scores

        # Select only the necessary columns; ensure ID is preserved exactly.
        df1_comm = df1_comm[["ID", "ext_sim_score", "int_sim_score", "community"]]
        df1_comm.rename(columns={"community": "Community"}, inplace=True)

        results.append(df1_comm)

    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(output_filename, index=False)
        print(f"Saved results to {output_filename} with {final_df.shape[0]} rows.")
    else:
        print(f"No results generated for {output_filename}.")

# ---- Example Usage ----
# For finetune: Compare d_ft (tweets to score) against d_1 (reference tweets)
compute_and_export_scores(d_ft, d_1, "/home/mhchu/llama3/helper_files/results/finetune_scores.csv")
# For RAG: Compare d_rag (tweets to score) against d_2 (reference tweets)
compute_and_export_scores(d_rag, d_2, "/home/mhchu/llama3/helper_files/results/rag_scores.csv")

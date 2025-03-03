import pandas as pd
import numpy as np

def merge_scores_and_predictions(scores_csv, preds_csv, output_csv):
    """
    Reads scores and predictions CSV files, resets the ID columns,
    merges them using an inner join, and saves the merged DataFrame.
    Also asserts that both input CSVs have the same number of rows.
    """
    # Read the scores and predictions CSVs
    df_scores = pd.read_csv(scores_csv)
    df_preds = pd.read_csv(preds_csv)

    # Assert that both CSVs have the same number of rows
    assert len(df_scores) == len(df_preds), (
        f"Row count mismatch: {scores_csv} has {len(df_scores)} rows, "
        f"but {preds_csv} has {len(df_preds)} rows."
    )

    # Select the necessary columns from the scores file
    df_scores = df_scores[['ID', 'ext_sim_score', 'int_sim_score']]

    # Reset the ID columns to be a vector from 1 to len(df)
    df_scores["ID"] = np.arange(1, len(df_scores) + 1)
    df_preds["ID"] = np.arange(1, len(df_preds) + 1)

    # Merge the two DataFrames on ID using an inner join
    merged_df = df_scores.merge(df_preds, on="ID", how="inner")

    # Save the merged DataFrame to CSV
    merged_df.to_csv(output_csv, index=False)

    # Print the line count (number of rows) of the final CSV
    final_row_count = merged_df.shape[0]
    print(f"Merged CSV saved to: {output_csv}")
    print(f"Final CSV row count: {final_row_count}")

def filter_merged_data(input_csv, output_csv, sim_threshold=0.7, ppl_threshold=400, sample_size=None):
    """
    Reads the merged CSV and keeps only rows (tweets) that satisfy:
      - ext_sim_score < sim_threshold,
      - perplexity < ppl_threshold, and
      - int_sim_score < 1.0.

    Then, for each (community, topic) combination, it randomly samples
    up to `sample_size` rows. The result is saved to output_csv.

    For example, for RAG sample_size=2800, for Finetune sample_size=4000.
    """
    df = pd.read_csv(input_csv)
    initial_count = df.shape[0]

    # Keep only rows that satisfy the filtering conditions
    df_filtered = df[
        (df['ext_sim_score'] < sim_threshold) &
        (df['perplexity'] < ppl_threshold) &
        (df['int_sim_score'] < 1.0)
    ]
    final_count = df_filtered.shape[0]
    print(f"Rows before filtering: {initial_count}, Rows after filtering: {final_count}")

    # If sample_size is provided, sample that many rows per (community, topic) group
    if sample_size is not None:
        # Group by both community and topic (adjust column names if needed)
        df_sampled = df_filtered.groupby(["community"], group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), sample_size), random_state=42)
        )
        sampled_count = df_sampled.shape[0]
        print(f"Rows after sampling: {sampled_count}")
    else:
        df_sampled = df_filtered

    df_sampled.to_csv(output_csv, index=False)
    print(f"Filtered and sampled CSV saved to: {output_csv}")

if __name__ == '__main__':
    # Finetune: Merge and then filter & sample 4000 rows per (community, topic)
    finetune_merged = "/home/mhchu/llama3/helper_files/results/finetune_scores_all.csv"
    merge_scores_and_predictions(
        scores_csv="/home/mhchu/llama3/helper_files/results/finetune_scores.csv",
        preds_csv="/home/mhchu/llama3/helper_files/results/finetune_scores_with_ppl.csv",
        output_csv=finetune_merged
    )
    filter_merged_data(
        input_csv=finetune_merged,
        output_csv="/home/mhchu/llama3/helper_files/results/finetune_scores_filtered.csv",
        sim_threshold=0.7,
        ppl_threshold=400,
        sample_size=6000
    )

    # RAG: Merge and then filter & sample 2800 rows per (community, topic)
    rag_merged = "/home/mhchu/llama3/helper_files/results/rag_scores_all.csv"
    merge_scores_and_predictions(
        scores_csv="/home/mhchu/llama3/helper_files/results/rag_scores.csv",
        preds_csv="/home/mhchu/llama3/helper_files/results/rag_scores_with_ppl.csv",
        output_csv=rag_merged
    )
    filter_merged_data(
        input_csv=rag_merged,
        output_csv="/home/mhchu/llama3/helper_files/results/rag_scores_filtered.csv",
        sim_threshold=0.7,
        ppl_threshold=400,
        sample_size=6000
    )

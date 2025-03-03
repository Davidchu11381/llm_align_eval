import pandas as pd
import matplotlib.pyplot as plt

# Load the processed data from CSV files
ft_scores_df = pd.read_csv("/home/mhchu/llama3/helper_files/results_similarity_scores/finetune_scores.csv")
rag_scores_df = pd.read_csv("/home/mhchu/llama3/helper_files/results_similarity_scores/rag_scores.csv")

def plot_histogram_overlay(ft_scores_df, rag_scores_df):
    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, community in enumerate(ft_scores_df.columns):
        # Plot histograms for fine-tuned model
        ax.hist(ft_scores_df[community].dropna(), bins=30, edgecolor='black', linewidth=1.2,
                color='skyblue', alpha=0.5, label=f'{community} FT' if idx == 0 else "",
                histtype='stepfilled', linestyle='-')

        # Plot histograms for RAG model
        ax.hist(rag_scores_df[community].dropna(), bins=30, edgecolor='black', linewidth=1.2,
                color='orange', alpha=0.5, label=f'{community} RAG' if idx == 0 else "",
                histtype='stepfilled', linestyle='--')

    ax.set_xlabel('Similarity Score', fontsize=20)
    ax.set_ylabel('# of Entries', fontsize=20)

    # Customize the grid
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add padding between the axes and the plot
    ax.tick_params(axis='both', which='major', labelsize=16, pad=8)

    # Add legend
    ax.legend(loc='upper right', fontsize=14, ncol=2)

    # Adjust the layout
    fig.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(f'histogram_similarity_all_communities.pdf', format='pdf', dpi=300, bbox_inches='tight')

    # Show the plot (optional)
    plt.show()

# Plot the overlaid histograms
plot_histogram_overlay(ft_scores_df, rag_scores_df)

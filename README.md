## Framework to align LLMs to online communities

- This repo is built off https://github.com/hiyouga/LLaMA-Factory 

- Go to bashscripts/full_multi_gpu and run single_node.sh and single_node_rag.sh

All sampled data are provided in data.

## Framework to evaluate LLM Alignment to online communities
- helper_files/similarity_score.py to remove tweets that are too similar to each other.

- Run bashscripts/full_multi_gpu/classification.sh to train the models to classify tweets on their originality and use bashscripts/full_multi_gpu/classification_predict.sh to test.

- helper_files/ibm_corpora_comparison.py is used to compare the semantic similarity between the text corpus.

- helper_files/toxicity.py to generate toxicity scores

- Refer to https://github.com/gchochla/Demux-MEmo to compute emotion scores.

Contact: davidchu11381@gmail.com 

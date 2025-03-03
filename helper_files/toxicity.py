from detoxify import Detoxify
import pandas as pd
import torch
import argparse
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


class Detoxify2(Detoxify):
    def predict2(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = (
                scores[0][i] if isinstance(input_ids[0], int) else [scores[ex_i][i].tolist() for ex_i in
                                                                    range(len(scores))]
            )
        return results


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        item = {
            'text': self.texts[index],
        }
        return item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--file_path', type=str, default='/home/mhchu/llama3/helper_files/results/rag_scores_filtered.csv')
    parser.add_argument('--field', type=str, default='text')
    parser.add_argument('--max_len', type=int, default=64)

    # inference
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.cuda.empty_cache()

    custom_data_types = {'text': 'str', 'label': 'str',
                         'community': 'str'}
    df = pd.read_csv(args.file_path, lineterminator='\n',
                     index_col=None, dtype=custom_data_types).dropna()
    df.replace("", np.nan, inplace=True)
    # Drop rows with NaN values
    df.dropna(inplace=True)
    # df = df.sample(frac=0.1)
    # df = df.reset_index(drop=True)
    texts = df[args.field].tolist()
    n = len(texts)
    print(f'{n} texts!')
    print("type", type(texts))

    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    if torch.cuda.device_count() > 0:
        print(f'Using {torch.cuda.device_count()} GPU(s)!')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Detoxify2('original', device=device)
    text_dataset = TextDataset(texts)
    loader = DataLoader(text_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False)

    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    results_all = {}
    # interval = len(loader) // 20
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            batch_texts = batch['text']
            encodings = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding='max_length',
                                  max_length=args.max_len)
            batch_input_ids = encodings['input_ids'].to(device)
            batch_attention_mask = encodings['attention_mask'].to(device)
            results = model.predict2(batch_input_ids, batch_attention_mask)

            # if i % interval == 0 or i == len(texts) - 1:
            #     print(f'{i}/{len(loader)}')

            for key in results:
                results_all.setdefault(key, [])
                results_all[key].extend(results[key])

        for key in results_all:
            df[key] = results_all[key]

    basename = os.path.basename(args.file_path)
    os.makedirs('results', exist_ok=True)
    df.to_csv('/home/mhchu/llama3/toxicity_scores/rag_toxicity.csv', index=False)
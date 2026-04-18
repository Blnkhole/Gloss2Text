from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fasttext
import pandas as pd
import torch


class GlossData(Dataset):
    def __init__(self, path: str, subset, tokenizer, type="enc_dec", back_trans=False):

        df = pd.read_csv(path + f"/{subset}.tsv", sep="\t", header=None)
        df = df.dropna().drop_duplicates()

        self.gloss = df[0].astype(str).str.lower().tolist()
        self.translation = df[1].astype(str).tolist()

        self.tokenizer = tokenizer
        self.subset = subset

        self.X = self.gloss
        self.Y = self.translation

        print(type)

        used_tokens = set()
        for text in self.Y:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            used_tokens.update(ids)

        vocab_size = len(self.tokenizer)
        used_tokens = [t for t in used_tokens if t < vocab_size]

        self.used_tokens = sorted(list(used_tokens))

        print("Used vocab size:", len(self.used_tokens))

        vocab = self.tokenizer.get_vocab()
        id2token = {v: k for k, v in vocab.items()}

        tokens = [id2token[i] for i in self.used_tokens]

        #IMPORTANT
        #decode tokens first
    
        self.list_of_texts = self.tokenizer.batch_decode(
            np.array(self.used_tokens).reshape(-1, 1)
        )

        #normalize token artifacts
        self.list_of_texts = [
            t.replace("▁", " ").strip() if isinstance(t, str) else "<unk>"
            for t in self.list_of_texts
        ]

        model_ft = fasttext.load_model('cc.vi.300.bin')

        embeddings = [
            model_ft.get_sentence_vector(text)
            for text in self.list_of_texts
        ]

        embeddings = np.nan_to_num(np.array(embeddings))

        # cosine similarity
        sim = cosine_similarity(embeddings)
        sim = np.nan_to_num(sim)

        self.similarity_matrix = torch.tensor(sim, dtype=torch.float32)

        # mapping
        self.token_id_map = {
            tok_id: idx for idx, tok_id in enumerate(self.used_tokens)
        }

        print("Similarity matrix shape:", self.similarity_matrix.shape)

    def __len__(self):
        return len(self.X)

    def return_sim(self):
        return self.similarity_matrix, self.token_id_map

    def __getitem__(self, idx):
        x = self.tokenizer(
            self.X[idx],
            truncation=True,
            padding="max_length",
            max_length=128
        )

        y = self.tokenizer(
            self.Y[idx],
            truncation=True,
            padding="max_length",
            max_length=128
        )

        labels = torch.tensor(y["input_ids"])
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": torch.tensor(x["input_ids"]),
            "attention_mask": torch.tensor(x["attention_mask"]),
            "labels": labels
        }
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
        
        self.X = []
        self.Y = []

        print(type)

        for idx, i in enumerate(self.gloss):
            self.X.append(i)                      # gloss input
            self.Y.append(self.translation[idx])  # text target

        used_tokens = set()

        for text in self.Y:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            used_tokens.update(ids)

        self.used_tokens = sorted(list(used_tokens))

        vocab_size = len(self.tokenizer)

        self.used_tokens = [t for t in self.used_tokens if t < vocab_size]

        print("Used vocab size:", len(self.used_tokens))

        print("DATA SIZE:", len(self.gloss))
        print("SAMPLE:", self.gloss[0], "->", self.translation[0])

        print("Building similarity matrix for SALS...")

        vocab = self.tokenizer.get_vocab()
        self.id2token = {v: k for k, v in vocab.items()}

        tokens = [self.id2token[i] for i in self.used_tokens]

        model_ft = fasttext.load_model('cc.vi.300.bin')

        embeddings = []
        for tok in tokens:
            tok = tok.replace("▁", " ").strip()  # normalize BPE
            if tok == "":
                tok = "<unk>"

            vec = model_ft.get_word_vector(tok)
            vec = np.nan_to_num(vec)
            embeddings.append(vec)

        embeddings = np.array(embeddings)
        embeddings = np.nan_to_num(embeddings)

        self.embeddings = embeddings

        sim = cosine_similarity(embeddings)
        sim = np.nan_to_num(sim)

        sim = (sim + 1) / 2

        temperature = 0.05
        sim = sim ** (1 / temperature)
        
        sim = sim / (sim.sum(axis=1, keepdims=True) + 1e-8)

        self.similarity_matrix = torch.tensor(sim, dtype=torch.float32)

        """
        temperature = 0.3
        self.similarity_matrix = self.similarity_matrix / temperature
        self.similarity_matrix -= np.max(self.similarity_matrix, axis=1, keepdims=True)

        self.similarity_matrix = np.exp(self.similarity_matrix)

        row_sums = self.similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.similarity_matrix /= row_sums
        """
        self.token_id_map = {tok_id: idx for idx, tok_id in enumerate(self.used_tokens)}
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

        # ignore padding token in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": torch.tensor(x["input_ids"]),
            "attention_mask": torch.tensor(x["attention_mask"]),
            "labels": labels
        }

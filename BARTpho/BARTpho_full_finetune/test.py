import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import Dataset
from metrics import report_all
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate


model_path = "/content/drive/MyDrive/Gloss2Text/best_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only = True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def evaluate_on_test_set(test_dataset, batch_size = 64):
    model.eval()
    predictions = []
    references = []

    print("Testing...")

    for i in tqdm(range(0, len(test_dataset), batch_size)):

        batch = test_dataset[i : (i + batch_size)]

        inputs = tokenizer(batch["gloss"], return_tensors = "pt",  padding = True, truncation = True, max_length = 128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length = 128,
                num_beams = 5,
                early_stopping = True
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens = True)

        predictions.extend([d.strip() for d in decoded])
        references.extend([t.strip() for t in batch["text"]])

    refs = [ref for ref in references]
    preds = [pred.strip() for pred in predictions]

    metrics = report_all(refs, preds)

    return metrics, predictions, references

test_df = pd.read_csv("/content/drive/MyDrive/Gloss2Text/test.tsv", sep = "\t", header = None)
test_df.columns = ["gloss", "text"]

test_dataset = Dataset.from_pandas(test_df, preserve_index = False)

#load cpkt model
test_results, all_preds, all_labels = evaluate_on_test_set(test_dataset)

for k, v in test_results.items():
    print(f"{k}: {v:.4f}")

df_results = pd.DataFrame({
    "Gloss": [test_dataset[i]["gloss"] for i in range(len(test_dataset))],
    "Reference": [all_labels[i] for i in range(len(all_labels))],
    "Prediction": all_preds
})
df_results.to_csv("test_prediction.tsv", sep = "\t", index = False)
print("Complete!")

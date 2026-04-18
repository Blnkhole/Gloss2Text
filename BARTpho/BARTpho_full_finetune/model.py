from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from datasets import Dataset
import pandas as pd
import numpy as np
import evaluate
import torch

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

def preprocess(example):

    model_inputs = tokenizer(
        example["gloss"],
        max_length = 64,
        truncation = True,
        padding = "max_length"
    )

    labels = tokenizer(
        example["text"],
        max_length = 64,
        truncation = True,
        padding = "max_length"
    )

    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#load data

path = "/content/drive/MyDrive/Gloss2Text/"

train_dataset = Dataset.from_pandas(pd.read_csv("/content/drive/MyDrive/Gloss2Text/train.tsv", sep = "\t", names = ["gloss", "text"]))
val_dataset = Dataset.from_pandas(pd.read_csv("/content/drive/MyDrive/Gloss2Text/dev.tsv", sep = "\t", names = ["gloss", "text"]))

train_dataset = train_dataset.map(preprocess, batched = True)
val_dataset = val_dataset.map(preprocess, batched = True)


#load model
model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable")

#data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

#training arguments

training_args = Seq2SeqTrainingArguments(
    output_dir = "/content/drive/MyDrive/Gloss2Text/checkpoints",
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    learning_rate = 1e-5,
    num_train_epochs = 6,
    logging_steps = 100,

    eval_strategy = "epoch",
    save_strategy = "epoch",

    load_best_model_at_end = True,
    metric_for_best_model = "bleu",
    greater_is_better = True,

    predict_with_generate = True,
    generation_max_length = 64,
    generation_num_beams = 5,

    save_total_limit = 2,
    fp16 = True,
)

#load metrics

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
chrf = evaluate.load("chrf")


#compute metrics

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.array(preds).astype(int)
    labels = np.array(labels).astype(int)

    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    labels = np.where(labels >= 0, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens = True)
    decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens = True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    #BLEU + CHRF
    refs_for_bleu = [[l] for l in decoded_labels]

    #training metrics
    bleu_score = bleu.compute(predictions = decoded_preds, references = refs_for_bleu) #bleu 4
    chrf_score = chrf.compute(predictions = decoded_preds, references = refs_for_bleu)
    rouge_score = rouge.compute(predictions = decoded_preds, references = decoded_labels)

    return {
        "bleu": bleu_score["bleu"],
        "chrf": chrf_score["score"],
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
    }

#gen sample in test set

class PrintBatchCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):

        if state.global_step % 1000 == 0 and state.global_step > 0:

            model.eval()

            samples = [val_dataset[i] for i in range(5)]  #5 sents
            glosses = [s["gloss"] for s in samples]
            refs = [s["text"] for s in samples]

            inputs = tokenizer(glosses, return_tensors = "pt", padding = True, truncation = True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length = 64, num_beams = 1)

            preds = tokenizer.batch_decode(outputs, skip_special_tokens = True)

            print("\n===== SAMPLES =====")
            for g, r, p in zip(glosses, refs, preds):
                print(f"G: {g}")
                print(f"R: {r}")
                print(f"P: {p}")
                print("-" * 30)

            model.train()

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model),
    compute_metrics = compute_metrics
)

trainer.add_callback(PrintBatchCallback())
trainer.train()

trainer.save_model("/content/drive/MyDrive/Gloss2Text/best_model")
tokenizer.save_pretrained("/content/drive/MyDrive/Gloss2Text/best_model")

def generate(gloss):
    inputs = tokenizer(gloss, return_tensors = "pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length = 64,
        num_beams = 5,
        early_stopping = True,
        no_repeat_ngram_size = 3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens = True)

print(generate("Tôi con dê ghét nhất"))
print(generate("Tôi thịt heo thích nhất"))
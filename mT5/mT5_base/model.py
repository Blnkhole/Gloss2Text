from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from datasets import Dataset
import pandas as pd
import numpy as np
import evaluate
import torch

tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

print(tokenizer.pad_token, tokenizer.pad_token_id)

def preprocess(example):

    inputs = ["translate gloss to vietnamese: " + g for g in example["gloss"]]

    model_inputs = tokenizer(
        inputs,
        truncation = True,
        #padding = "max_length",
        max_length = 64
    )

    labels = tokenizer(
        text_target = example["text"],
        truncation=True,
        #padding = "max_length",
        max_length = 64
    )

    model_inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels["input_ids"]
    ]

    return model_inputs

#load data

path = "/content/drive/MyDrive/Gloss2Text/"

raw_train_dataset = Dataset.from_pandas(pd.read_csv("/content/drive/MyDrive/Gloss2Text/train.tsv", sep = "\t", names = ["gloss", "text"]))
raw_val_dataset = Dataset.from_pandas(pd.read_csv("/content/drive/MyDrive/Gloss2Text/dev.tsv", sep = "\t", names = ["gloss", "text"]))

train_dataset = raw_train_dataset.map(preprocess, batched = True)
val_dataset = raw_val_dataset.map(preprocess, batched = True)

sample = raw_train_dataset[0]
processed = preprocess({"gloss":[sample["gloss"]], "text":[sample["text"]]})

print(tokenizer.decode([t for t in processed["labels"][0] if t != -100]))


#load model
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
model.gradient_checkpointing_enable()

#data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

#training arguments

training_args = Seq2SeqTrainingArguments(
    output_dir = "/content/drive/MyDrive/Gloss2Text/checkpoints",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 8,
    per_device_eval_batch_size = 2,
    optim = "adafactor",
    learning_rate = 5e-5,
    #warmup_steps = 500,
    gradient_checkpointing = True,


    #learning_rate = 5e-5,
    num_train_epochs = 15,
    logging_steps = 100,

    #label_smoothing_factor = 0.1,

    eval_strategy = "epoch",
    save_strategy = "epoch",

    load_best_model_at_end = True,
    metric_for_best_model = "bleu",
    greater_is_better = True,

    predict_with_generate = True,
    generation_max_length = 64,
    generation_num_beams = 5,

    save_total_limit = 2,
    fp16 = False,
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

    if hasattr(preds, "tolist"): preds = preds.tolist()
    if hasattr(labels, "tolist"): labels = labels.tolist()

    labels = [[(t if t != -100 else tokenizer.pad_token_id) for t in label] for label in labels]

    vocab_size = tokenizer.vocab_size
    preds = [[(t if (0 <= t < vocab_size) else tokenizer.pad_token_id) for t in pred] for pred in preds]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens = True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens = True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    #BLEU + CHRF
    refs_for_bleu = [[l] for l in decoded_labels]

    #training metrics
    bleu_score = bleu.compute(predictions = decoded_preds, references = refs_for_bleu) #bleu 4
    chrf_score = chrf.compute(predictions = decoded_preds, references = refs_for_bleu)
    rouge_score = rouge.compute(predictions = decoded_preds, references = decoded_labels)

    return {
        "bleu": bleu_score["bleu"] * 100,
        "chrf": chrf_score["score"],
        "rouge1": rouge_score["rouge1"] * 100,
        "rouge2": rouge_score["rouge2"] * 100,
        "rougeL": rouge_score["rougeL"] * 100,
    }

#gen sample in test set

class PrintBatchCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):

        if state.global_step % 300 == 0 and state.global_step > 0:

            model.eval()

            samples = [raw_val_dataset[i] for i in range(5)]  #5 sents
            glosses = [s["gloss"] for s in samples]
            refs = [s["text"] for s in samples]

            inputs = tokenizer(
                ["translate gloss to vietnamese: " + g for g in glosses],
                return_tensors = "pt", padding = True, truncation = True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length = 64, num_beams = 5,
                                no_repeat_ngram_size = 3, repetition_penalty = 1.2)

            preds = tokenizer.batch_decode(outputs.detach().cpu().tolist(), skip_special_tokens = True)

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
    model.eval()
    gloss = "translate gloss to vietnamese: " + gloss
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
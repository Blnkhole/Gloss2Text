import torch
import tqdm
import pandas as pd
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler, SchedulerType
from peft import LoraConfig, get_peft_model, TaskType
from GlossData import GlossData
from loss import SALSLoss
from metrics import report_all

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
epochs = 8
annotation_dir = "/content/drive/MyDrive/Gloss2text"
model_name = "vinai/bartpho-syllable"
best_bleu = 0

def save_checkpoint(model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

#Model Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#LoRA Config
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

#Dataset & Loss
train_dataset = GlossData(annotation_dir, "train", tokenizer)
dev_dataset   = GlossData(annotation_dir, "dev", tokenizer)
test_dataset  = GlossData(annotation_dir, "test", tokenizer)

pad_idx = tokenizer.pad_token_id
sim_matrix, token_id_map = train_dataset.return_sim()
sals_loss_fn = SALSLoss(sim_matrix, token_id_map, pad_idx).to(device)

#LoRA
model = get_peft_model(model, config)
model.to(device)

#DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(dev_dataset, batch_size=batch_size * 2)

#Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
num_training_steps = epochs * len(train_loader)
scheduler = get_scheduler(
    name=SchedulerType.COSINE,
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

def evaluate(model, dataloader, num_samples=5):
    model.eval()
    preds, refs, inputs_all = [], [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            #Decode Gloss
            inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            inputs_all.extend(inputs)

            #Generate
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

            pred_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            preds.extend(pred_texts)
            refs.extend(ref_texts)

    #Get BLEU, Rouge,
    metrics = report_all(refs, preds)
    print("\n--- Evaluation Metrics ---")
    print(metrics)

    #Sample
    random.seed(42)
    indices = random.sample(range(len(preds)), min(num_samples, len(preds)))
    for i, idx in enumerate(indices):
        print(f"\nExample {i+1}")
        print(f"GLOSS: {inputs_all[idx]}")
        print(f"REF  : {refs[idx]}")
        print(f"PRED : {preds[idx]}")

    return metrics.get("bleu", 0)

def train(model, train_loader, val_loader, optimizer, scheduler):
    global best_bleu

    steps_per_epoch = len(train_loader)
    print_every = steps_per_epoch // 2 

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

        for step, batch in enumerate(loop):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            logits = outputs.logits

            #SALSLoss
            ce_loss = outputs.loss
            sals_loss = sals_loss_fn(logits, batch["labels"])

            loss = ce_loss + 0.3 * sals_loss

            if torch.isnan(loss) or torch.isinf(loss):
              print("NaN LOSS DETECTED")
              print("logits:", logits.min().item(), logits.max().item())
              return 

            optimizer.zero_grad()
            loss.backward()

            #Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

            #print loss(0.5 epoch)
            if (step + 1) % print_every == 0:
                avg_loss = total_loss / (step + 1)
                loop.set_postfix(avg_loss=f"{avg_loss:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\nAverage Epoch {epoch+1} Loss: {avg_loss:.4f}")

        current_bleu = evaluate(model, val_loader)

        #Save cpkt
        save_checkpoint(model, tokenizer, f"checkpoint_epoch_{epoch+1}")

        #Best model(BLEU)
        if current_bleu > best_bleu:
            best_bleu = current_bleu
            print(f"New best BLEU: {best_bleu:.4f}, saving model...")
            save_checkpoint(model, tokenizer, "best_model")

#Training
if __name__ == "__main__":
    print("Starting training...")
    train(model, train_loader, val_loader, optimizer, scheduler)
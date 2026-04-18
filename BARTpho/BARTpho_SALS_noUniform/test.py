from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from torch.utils.data import DataLoader
from GlossData import GlossData
import torch
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "vinai/bartpho-syllable"
best_model_path = "/content/drive/MyDrive/best_model"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(base_model, best_model_path)
model.to(device)
model.eval()

annotation_dir = "/content/drive/MyDrive/Gloss2text"
test_dataset = GlossData(annotation_dir, "test", tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


glosses = []
refs = []
preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

        pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        labels = batch["labels"].cpu().numpy()
        ref_texts = [
            tokenizer.decode([t for t in seq if t != -100], skip_special_tokens=True)
            for seq in labels
        ]

        input_texts = tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=True)

        glosses.extend(input_texts)
        preds.extend(pred_texts)
        refs.extend(ref_texts)

metrics = report_all(refs, preds)

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

df = pd.DataFrame({
    "gloss": glosses,
    "reference": refs,
    "prediction": preds
})

output_path = "test_predictions.tsv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Complete!")
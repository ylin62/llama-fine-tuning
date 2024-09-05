import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, AdamW, get_scheduler
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

df = pd.read_csv("data/Combined Data.csv", index_col=0).dropna(how="any")

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df["status"])

train_text, test_text, train_labels, test_labels = train_test_split(df["statement"], labels, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(pd.DataFrame({"text": train_text, "label": train_labels}))
test_dataset = Dataset.from_pandas(pd.DataFrame({"text": test_text, "label": test_labels}))

model_path = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path) #, token="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")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
def preprocess_data(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    tokenized["label"] = examples["label"]
    return tokenized

train_dataset = train_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_labels = np.array(train_dataset["label"])
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(label_encoder.classes_), ignore_mismatched_sizes=False) #, token="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")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=1e-4)

num_epochs = 10
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

try:
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        total_train_losses = 0
        train_correct_predictions = 0
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(outputs.logits, batch["label"])
            total_train_losses += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            train_correct_predictions += torch.sum(torch.argmax(outputs.logits, dim=-1) == batch["label"]).item()

            progress_bar.set_postfix({"loss": loss.item()})

        train_loss = total_train_losses / len(train_loader)
        train_accuracy = train_correct_predictions / len(train_dataset)

        model.eval()
        total_eval_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = criterion(outputs.logits, batch["label"])
                total_eval_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                correct_predictions += torch.sum(preds == batch["label"]).item()

        avg_eval_loss = total_eval_loss / len(test_loader)
        accuracy = correct_predictions / len(test_dataset)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Accuracy = {100 * train_accuracy:.2f}%; Eval Loss = {avg_eval_loss:.4f}, Accuracy = {100 * accuracy:.2f}%")

    print("Training complete.")
    
except KeyboardInterrupt as e:
    raise e
finally:
    torch.save(model.state_dict(), "model_state.pth")
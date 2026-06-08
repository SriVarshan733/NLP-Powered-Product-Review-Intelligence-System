import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score

# 1. Prepare Dataset (0 = Negative, 1 = Neutral, 2 = Positive)
def prepare_data():
    # Synthetic dataset for execution demo
    data = {
        "text": ["Great product, loved it!", "Horrible customer support.", "It was okay, nothing special.", "Waste of money.", "Amazing performance"],
        "label": [2, 0, 1, 0, 2]
    }
    return Dataset.from_dict(data)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_score": f1}

def train():
    dataset = prepare_data().train_test_split(test_size=0.2)
    
    # Load DistilBERT tokenizer & model for low-latency/serverless constraints
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_fn, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=3)
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=00.1,
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    print("🔥 Starting BERT Fine-Tuning...")
    trainer.train()
    
    # Save Model Artifacts for production deployment
    model.save_pretrained("./saved_bert_model")
    tokenizer.save_pretrained("./saved_bert_model")
    print("✅ Model saved to './saved_bert_model'")

if __name__ == "__main__":
    train()
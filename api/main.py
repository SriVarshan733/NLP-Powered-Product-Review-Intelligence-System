from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os

app = FastAPI(title="NLP Review Intelligence API", version="1.0")

# Lazy-loading model for Cloud Run cold-start optimization
MODEL_PATH = "./saved_bert_model"
tokenizer = None
model = None

class InferenceRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global tokenizer, model
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        # Fallback if custom model isn't built yet inside container
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict_sentiment(payload: InferenceRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty")
        
    inputs = tokenizer(payload.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        
    classes = ["Negative", "Neutral", "Positive"] if model.config.num_labels == 3 else ["Negative", "Positive"]
    
    return {
        "text": payload.text,
        "sentiment": classes[pred_class],
        "confidence": float(probs[0][pred_class].item())
    }
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os
import re
import torch
import torch.nn.functional as F

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# ===============================
# App Configuration
# ===============================

app = FastAPI(
    title="EmailSort Backend API",
    description="Enterprise Email Classification & Urgency Detection",
    version="1.0"
)


# ===============================
# Model Paths
# ===============================
CATEGORY_MODEL_DIR = "models/category_model"
URGENCY_MODEL_DIR = "models/urgency_level_model"

for path in [CATEGORY_MODEL_DIR, URGENCY_MODEL_DIR]:
    if not os.path.exists(path):
        raise RuntimeError(f"Model folder missing: {path}")

# ===============================
# Labels
# ===============================
CATEGORY_LABELS = ["Complaint", "Feedback", "Spam", "Inquiry"]
URGENCY_LABELS = ["Low", "Medium", "High"]

# ===============================
# Load Tokenizer
# ===============================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# ===============================
# Load Models
# ===============================
category_model = DistilBertForSequenceClassification.from_pretrained(
    CATEGORY_MODEL_DIR)
urgency_model = DistilBertForSequenceClassification.from_pretrained(
    URGENCY_MODEL_DIR)


class EmailRequest(BaseModel):
    subject: str
    body: str

class PredictionResponse(BaseModel):
    category: str
    category_confidence: float
    urgency: str
    urgency_confidence: float
    urgency_source: str
    timestamp: str

# ===============================
# Text Preprocessing
# ===============================
def preprocess_text(subject: str, body: str):
    text = f"{subject} {body}".lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# Rule-Based Urgency
# ===============================
HIGH_URGENCY = [
    "urgent", "asap", "immediately", "system down",
    "not working", "failed", "critical"
]

MEDIUM_URGENCY = [
    "delay", "issue", "problem", "request", "help"
]

def rule_based_urgency(text: str) -> str:
    if any(k in text for k in HIGH_URGENCY):
        return "High"
    if any(k in text for k in MEDIUM_URGENCY):
        return "Medium"
    return "Low"

def hybrid_urgency(ml: str, rule: str) -> str:
    priority = {"Low": 0, "Medium": 1, "High": 2}
    return rule if priority[rule] > priority[ml] else ml

# ==============================
# Prediction Endpoint (Public)
# ===============================
@app.post("/predict", response_model=PredictionResponse)
def predict_email(email: EmailRequest):
    text = preprocess_text(email.subject, email.body)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v for k, v in inputs.items()}

    with torch.no_grad():
        cat_outputs = category_model(**inputs)
        urg_outputs = urgency_model(**inputs)

    cat_probs = F.softmax(cat_outputs.logits, dim=1)
    urg_probs = F.softmax(urg_outputs.logits, dim=1)

    cat_idx = torch.argmax(cat_probs).item()
    urg_idx = torch.argmax(urg_probs).item()

    ml_urgency = URGENCY_LABELS[urg_idx]
    rule_urgency = rule_based_urgency(text)
    final_urgency = hybrid_urgency(ml_urgency, rule_urgency)

    return PredictionResponse(
        category=CATEGORY_LABELS[cat_idx],
        category_confidence=round(cat_probs[0][cat_idx].item(), 4),
        urgency=final_urgency,
        urgency_confidence=round(urg_probs[0][urg_idx].item(), 4),
        urgency_source="hybrid",
        timestamp=datetime.utcnow().isoformat()
    )

# ===============================
# Health Check
# ===============================
@app.get("/")
def health_check():
    return {"status": "EmailSort API running"}

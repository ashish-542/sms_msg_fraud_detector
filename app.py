from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load trained model + vectorizer
model = joblib.load("fraud_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="AI Fraud Call/SMS Detector")

class AnalyzeRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-fraud-detector"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Transform text and predict
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    # Map ML prediction to status
    if prediction == "ham":
        status = "safe"
        risk_score = 10
        reasons = []
        confidence = float(model.predict_proba(X)[0][model.classes_.tolist().index("ham")])
    else:
        status = "fraud_detected"
        risk_score = 90
        reasons = ["ML model flagged as spam/fraud"]
        confidence = float(model.predict_proba(X)[0][model.classes_.tolist().index("spam")])

    return {
        "status": status,
        "risk_score": risk_score,
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "prediction": prediction
    }

import os
import joblib
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, Path
from pydantic import BaseModel
from pymongo import MongoClient
from bson.objectid import ObjectId
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv

# --- LOAD ENV ---
load_dotenv()

# --- CONFIG ---
MONGO_URL = os.getenv("MONGO_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 6000))

# --- INIT ---
app = FastAPI(title="Fraud Detector Backend with Auth")
client = MongoClient(MONGO_URL)
db = client["fraudDB"]
users_collection = db["users"]
messages_collection = db["messages"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- MODELS ---
class RegisterRequest(BaseModel):
    name: str
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class AnalyzeRequest(BaseModel):
    text: str

# --- HELPERS ---
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

# --- ROUTES ---

@app.post("/register")
def register(req: RegisterRequest):
    if users_collection.find_one({"username": req.username}):
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_pw = hash_password(req.password)
    user = {
        "username": req.username,
        "name": req.name,
        "password": hashed_pw,
        "created_at": datetime.utcnow()
    }
    result = users_collection.insert_one(user)
    return {"message": "✅ User registered", "user_id": str(result.inserted_id)}

@app.post("/login")
def login(req: LoginRequest):
    user = users_collection.find_one({"username": req.username})
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": str(user["_id"]), "username": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-fraud-detector"}

# --- ML MODEL ---
model = joblib.load("fraud_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.post("/analyze")
def analyze(req: AnalyzeRequest, user: dict = Depends(get_current_user)):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Transform text and predict
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    if prediction == "ham":
        status = "safe"
        risk_score = 10
        reasons = []
        confidence = float(model.predict_proba(X)[0][model.classes_.tolist().index("ham")])
    else:
        status = "fraud_detected"
        risk_score = 90
        # reasons = ["ML model flagged as spam/fraud"]
        confidence = float(model.predict_proba(X)[0][model.classes_.tolist().index("spam")])
        keywords = extract_keywords(text, vectorizer, model)
        reasons = [f"Suspicious keywords found: {', '.join(keywords)}"]


    # Save message in DB with user_id from auth token
    new_msg_id = save_message_to_db(
        str(user["_id"]), text, prediction, status, risk_score, confidence, reasons
    )

    return {
        "status": status,
        "risk_score": risk_score,
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "prediction": prediction,
        "analyzed_by": user["username"],
        "message_id": new_msg_id,
    }

# --- HELPER FUNCTION ---
def save_message_to_db(user_id: str, text: str, prediction: str, status: str, risk_score: int, confidence: float, reasons: list):
    """Save analyzed message to MongoDB messages collection."""
    message_doc = {
        "user_id": user_id,
        "text": text,
        "prediction": prediction,
        "status": status,
        "risk_score": risk_score,
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "timestamp": datetime.utcnow()
    }
    result = messages_collection.insert_one(message_doc)
    return str(result.inserted_id)

def extract_keywords(text, vectorizer, model, top_n=3):
    """Return top keywords from the message contributing to spam classification."""
    feature_names = vectorizer.get_feature_names_out()
    X = vectorizer.transform([text])

    # Get indices of non-zero features
    nonzero_indices = X.nonzero()[1]

    # For spam class → usually index 1 (if classes_ = ["ham", "spam"])
    spam_idx = list(model.classes_).index("spam")

    # Collect feature weights for present words
    word_scores = {
        feature_names[i]: model.feature_log_prob_[spam_idx, i]
        for i in nonzero_indices
    }

    # Sort by importance
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    return [word for word, score in sorted_words[:top_n]]

@app.post("/messages/{message_id}/not_spam")
def mark_not_spam(
    message_id: str = Path(..., description="Message ID to mark as NOT SPAM"),
    user: dict = Depends(get_current_user)
):
    # Find the message
    message = messages_collection.find_one({"_id": ObjectId(message_id), "user_id": str(user["_id"])})
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # Update message → override prediction & status
    update_data = {
        "prediction": "ham",
        "status": "safe",
        "risk_score": 0,
        "reasons": ["User manually marked as NOT SPAM"],
        "corrected_at": datetime.utcnow(),
        "corrected_by": user["username"]
    }

    messages_collection.update_one(
        {"_id": ObjectId(message_id)},
        {"$set": update_data}
    )

    return {
        "message": "✅ Message marked as NOT SPAM",
        "message_id": message_id,
        "updated_fields": update_data
    }

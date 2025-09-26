import os
import joblib
import bcrypt
from fastapi import FastAPI, HTTPException, Depends
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
        reasons = ["ML model flagged as spam/fraud"]
        confidence = float(model.predict_proba(X)[0][model.classes_.tolist().index("spam")])

    return {
        "status": status,
        "risk_score": risk_score,
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "prediction": prediction,
        "analyzed_by": user["username"]
    }

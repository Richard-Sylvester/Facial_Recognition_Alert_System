from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, etc)
    allow_headers=["*"], # Allow all headers
)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["face_recognition"]
logs_collection =  db["logs"]

# Pydantic model for incoming log data 
class DetectionLog(BaseModel):
    name: str
    confidence: float
    timestamp: str

# Endpoint to store logs
@app.post("/log_detection/")
async def log_detection(log: DetectionLog):
    log_data = log.dict()
    result = logs_collection.insert_one(log_data)
    print(f"Inserted log with ID: {result.inserted_id}")
    return {"message": "Detection logged successfully"}

# Endpoint to get logs
@app.get("/get_logs/")
async def get_logs():
    logs = list(logs_collection.find({}, {"_id":0}))
    return {"logs": logs}
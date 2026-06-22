from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from analyzer import analyze_text

app = FastAPI(title="Emotion Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(payload: TextRequest):
    return analyze_text(payload.text)
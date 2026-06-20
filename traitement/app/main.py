from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()

    ### TODO: appeler le traitement 
    
    results = [
        {"id": 1, "x": 38, "y": 42, "radius": 3.2, "confidence": 94, "severity": "modéré", "type": "Mou", "size": "Grand"},
        {"id": 2, "x": 52, "y": 58, "radius": 2.1, "confidence": 87, "severity": "modéré", "type": "Dur", "size": "Grand"},
    ]

    return {"results": results}

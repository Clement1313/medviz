from fastAPI import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "processing running"}


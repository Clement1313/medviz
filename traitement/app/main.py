from fastAPI import FastAPI

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "processing running"}


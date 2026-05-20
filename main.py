from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InvokeRequest(BaseModel):
    a: float
    b: float

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/invoke")
async def invoke(request: InvokeRequest):
    return {"result": request.a + request.b}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

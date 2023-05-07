# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Value(BaseModel):
    value: int

@app.get("/")
async def greeting():
    return {"msg": "Thank you for access"}

@app.post("/infer/")
async def create_item(value: Value):
    return {"msg": value}

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

from TextPreprocessor import TextAnalyzer

app = FastAPI()

class userRequest(BaseModel):
    searchQuery : str
    paraArray: list

@app.post('/summarizer',response_model= str)
async def getUserRequest(userR : userRequest):
    mlmodel = TextAnalyzer()
    summ = mlmodel.runnerClass(userR.paraArray,userR.searchQuery)
    return summ



from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

from TextPreprocessor import TextAnalyzer

app = FastAPI()

class userRequest(BaseModel):
    searchQuery : str
    paraArray: list


@app.get('/')
async def simpleGet():
    return {"hello" : "world"}

# @app.get('/shal')
# async def simpleGet():
#     return {"name" : "Udara"}

@app.post('/summarizer',response_model= str)
async def getUserRequest(userR : userRequest):
    mlmodel = TextAnalyzer()
    summ = mlmodel.runnerClass(userR.paraArray,userR.searchQuery)
    return summ



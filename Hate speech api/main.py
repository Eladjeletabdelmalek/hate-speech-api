from fastapi import FastAPI,File,UploadFile
from pydantic import BaseModel
import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
import uvicorn
app =FastAPI()
with open("model/Hs_model.pkl", "rb") as file:
    model =pk.load(file)

with open("model/Hs_cv.pkl", "rb") as file:
    cv =pk.load(file)
class textin (BaseModel):
    text:str

@app.get("/")
def Home():
    return model.predict(cv.transform(["You are Awful"]).toarray())
@app.post("/predict")
def predict(data:textin):
    prediction=model.predict(cv.transform([data.text]).toarray())
    return prediction[0]

from fastapi import FastAPI
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(data: list):
    features = np.array(data).reshape(1, -1)
    prediction = model.predict(features)[0]
    classes = ["Setosa", "Versicolor", "Virginica"]
    return {"prediction": classes[prediction]}

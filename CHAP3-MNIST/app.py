from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = joblib.load("mnist.pkl")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("L").resize((28, 28))
    prediction = model.predict(np.array(img).reshape(1, -1))
    return {"prediction": int(prediction[0])}
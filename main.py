from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from inference import EmotionClassifier

app = FastAPI(
    title="Emotion Classifier with Face Detection",
    description="Detects faces and classifies facial emotions using ConvNeXt-Tiny.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup
model = EmotionClassifier(model_path="emotion_model3_clean.onnx")

@app.post("/predict", summary="Predict emotion from image")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = model.predict(image)
        return JSONResponse(content={"prediction": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

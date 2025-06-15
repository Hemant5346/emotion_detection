import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as ort

class CustomEmotionModel(nn.Module):
    """Custom model wrapper that matches the training architecture"""
    def __init__(self, model_name: str = "convnext_tiny", num_classes: int = 4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)  # num_classes=0 removes the head
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class EmotionClassifier:
    def __init__(self, model_path: str, class_names=None):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        self.class_names = class_names or ['angry', 'happy', 'neutral', 'sad']

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def extract_face(self, image: Image.Image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.face_detector.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return None

        h, w, _ = img_cv.shape
        box = results.detections[0].location_data.relative_bounding_box
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = int((box.xmin + box.width) * w)
        y2 = int((box.ymin + box.height) * h)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        face = img_cv[y1:y2, x1:x2]
        if face.size == 0:
            return None

        return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    def predict(self, image: Image.Image):
        face = self.extract_face(image)
        if face is None:
            return {"error": "No face detected."}

        image_tensor = self.transform(face).unsqueeze(0).numpy()

        # Run inference
        outputs = self.session.run(None, {self.input_name: image_tensor.astype(np.float32)})
        probs = outputs[0][0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "label": self.class_names[pred_idx],
            "confidence": round(confidence, 4)
        }
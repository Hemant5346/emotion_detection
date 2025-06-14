import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

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
    def __init__(self, model_path: str, model_name: str = "convnext_tiny", num_classes: int = 4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use the custom model that matches training architecture
        self.model = CustomEmotionModel(model_name, num_classes)
        
        # Load the state dict
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("✅ Model loaded successfully with custom architecture")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # If the above fails, try loading with strict=False
            self.model.load_state_dict(state_dict, strict=False)
            print("⚠️ Model loaded with strict=False")
        
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

        self.class_names = ['angry', 'happy', 'neutral', 'sad']
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
        # Take first detected face
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

        image_tensor = self.transform(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        
        return {
            "label": self.class_names[pred_idx],
            "confidence": round(confidence, 4)
        }
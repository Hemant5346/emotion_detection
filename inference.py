import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import tensorflow as tf
import torchvision.transforms as T

class EmotionClassifier:
    def __init__(self, model_path: str, class_names=None):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.class_names = class_names or ['angry', 'happy', 'neutral', 'sad']

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),
            T.Normalize([0.5] * 3, [0.5] * 3)
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

        face_resized = face.resize((224, 224))
        image_np = np.array(face_resized).astype(np.float32) / 255.0
        image_np = (image_np - 0.5) / 0.5  # Normalize to [-1, 1]
        image_tensor = np.expand_dims(image_np, axis=0).astype(np.float32)  # Shape: [1, 224, 224, 3]

        input_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_index, image_tensor)
        self.interpreter.invoke()

        output_index = self.output_details[0]['index']
        output = self.interpreter.get_tensor(output_index)[0]
        probs = tf.nn.softmax(output).numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "label": self.class_names[pred_idx],
            "confidence": round(confidence, 4)
        }

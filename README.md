# Emotion Classification API 🎭

A FastAPI-based emotion classification service that detects faces in images and classifies facial emotions using a fine-tuned ConvNeXt-Tiny model. The system automatically extracts faces from uploaded images and predicts one of four emotions: angry, happy, neutral, or sad.

## Features ✨

- **Face Detection**: Automatic face detection and extraction using MediaPipe
- **Emotion Classification**: ConvNeXt-Tiny model trained on emotion datasets
- **REST API**: Easy-to-use FastAPI endpoints
- **Docker Support**: Containerized deployment
- **CORS Enabled**: Ready for web applications
- **Real-time Processing**: Fast inference with GPU support

## Supported Emotions 😊😠😐😢

- **Angry** 😠
- **Happy** 😊
- **Neutral** 😐
- **Sad** 😢

## Project Structure 📁

```
emotion-api/
├── main.py                 # FastAPI application
├── inference.py            # Emotion classification model
├── train.py               # Training script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── best_emotion_model.pt  # Trained model weights
└── README.md             # This file
```

## Installation 🚀

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd emotion-api
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs

### Option 2: Local Installation

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have the trained model**
   - Place `best_emotion_model.pt` in the project root

3. **Run the application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## API Usage 📡

### Predict Emotion

**Endpoint**: `POST /predict`

**Description**: Upload an image to detect faces and classify emotions

**Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

**Response**:
```json
{
  "prediction": {
    "label": "happy",
    "confidence": 0.9234
  }
}
```

**Error Response** (No face detected):
```json
{
  "prediction": {
    "error": "No face detected."
  }
}
```

### Interactive Documentation

Visit http://localhost:8000/docs for Swagger UI documentation where you can:
- Test the API directly
- Upload images through the web interface
- View request/response schemas

## Model Architecture 🧠

### Training Configuration
- **Base Model**: ConvNeXt-Tiny (timm)
- **Input Size**: 224x224 RGB images
- **Classes**: 4 emotions (angry, happy, neutral, sad)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine Annealing
- **Augmentation**: MixUp + standard image augmentations
- **Loss**: Cross-Entropy with class weights

### Model Performance
- **Training Features**:
  - Balanced class weights
  - Early stopping (patience=5)
  - Mixed precision training
  - Data augmentation with MixUp

### Face Detection
- **Framework**: MediaPipe Face Detection
- **Model**: BlazeFace (model_selection=0)
- **Confidence Threshold**: 0.6
- **Processing**: Automatic face extraction and cropping

## Training 🏋️‍♂️

To train your own model:

1. **Prepare your dataset**
   ```
   output/merge_data/
   ├── angry/
   ├── happy/
   ├── neutral/
   └── sad/
   ```

2. **Run training**
   ```bash
   python train.py
   ```

3. **Training features**:
   - Automatic train/validation split (80/20)
   - Class weight balancing
   - Progress logging with timestamps
   - Best model checkpoint saving
   - Early stopping to prevent overfitting

## Dependencies 📦

### Core Requirements
```
torch==2.1.2           # PyTorch framework
timm==0.9.12           # Pre-trained models
fastapi==0.110.0       # Web framework
uvicorn==0.29.0        # ASGI server
pillow==10.3.0         # Image processing
opencv-python-headless==4.9.0.80  # Computer vision
mediapipe==0.10.9      # Face detection
numpy<2.0              # Numerical computing
```

### System Requirements
- Python 3.10+
- CUDA (optional, for GPU acceleration)
- 4GB+ RAM
- 2GB+ disk space

## Configuration ⚙️

### Environment Variables
- `MODEL_PATH`: Path to model weights (default: "best_emotion_model.pt")
- `DEVICE`: Computing device ("cuda" or "cpu", auto-detected)
- `PORT`: API port (default: 8000)

### Model Parameters
```python
IMG_SIZE = 224          # Input image size
BATCH_SIZE = 128        # Training batch size
NUM_CLASSES = 4         # Number of emotion classes
MODEL_NAME = "convnext_tiny"  # Base model architecture
```

## Troubleshooting 🔧

### Common Issues

1. **Model Loading Error**
   ```
   RuntimeError: Error(s) in loading state_dict for ConvNeXt
   ```
   **Solution**: The inference code includes automatic handling of model architecture mismatches.

2. **No Face Detected**
   - Ensure images contain clear, front-facing faces
   - Check image quality and lighting
   - Adjust MediaPipe confidence threshold if needed

3. **CUDA Out of Memory**
   - Reduce batch size during training
   - Use CPU inference: set `device="cpu"`

4. **Docker Build Issues**
   - Ensure sufficient disk space
   - Check Docker daemon is running
   - Verify internet connection for dependency downloads

### Performance Optimization

- **GPU Acceleration**: Install CUDA-compatible PyTorch
- **Batch Processing**: Process multiple images in batches
- **Model Quantization**: Consider model quantization for production
- **Caching**: Implement result caching for repeated requests

## Development 👨‍💻

### Adding New Emotions

1. **Update class names**:
   ```python
   self.class_names = ['angry', 'happy', 'neutral', 'sad', 'new_emotion']
   ```

2. **Update NUM_CLASSES**:
   ```python
   NUM_CLASSES = 5  # Updated count
   ```

3. **Retrain the model** with new data

### API Extensions

- Add batch prediction endpoint
- Implement confidence thresholds
- Add emotion intensity scoring
- Include face bounding box coordinates

## Contributing 🤝

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- **timm**: Excellent pre-trained model library
- **MediaPipe**: Robust face detection
- **FastAPI**: Modern, fast web framework
- **ConvNeXt**: State-of-the-art computer vision architecture

## Contact 📧

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.

---

**Happy Emotion Detection! 🎭✨**
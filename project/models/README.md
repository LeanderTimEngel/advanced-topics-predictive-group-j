# Models Directory

This directory is used to store trained models for the sentiment analysis system. The models are organized into subdirectories based on their type:

## Directory Structure

```
models/
├── text/                 # Text sentiment analysis models
│   ├── lstm/             # LSTM-based models
│   └── transformers/     # Transformer-based models
│
└── speech/               # Speech-related models
    ├── whisper/          # Fine-tuned Whisper models
    └── emotion/          # Direct speech emotion models
```

## Model Formats

The project supports several model formats:

- `.h5` and `.keras` - Keras model files
- `.pt` and `.pth` - PyTorch model files
- `.onnx` - ONNX format for cross-framework compatibility
- `.tflite` - TensorFlow Lite format for mobile/edge deployment

## Usage Guidelines

### Saving Models

When saving a model from a notebook or script, use the following pattern:

```python
# Example for saving a Keras model
model.save(f"project/models/text/lstm/sentiment_model_{date.today().strftime('%Y%m%d')}.keras")
```

### Loading Models

When loading a model, use the appropriate framework's loading function:

```python
# Example for loading a Keras model
from tensorflow.keras.models import load_model
model = load_model("project/models/text/lstm/sentiment_model_20240501.keras")
```

## Model Versioning

Include a timestamp or version number in the model filename to track different versions. This makes it easier to compare model performance over time.

## Note

Large model files are not tracked in Git. Make sure to document the training parameters and performance metrics in your notebooks or separate documentation files so models can be recreated if needed. 
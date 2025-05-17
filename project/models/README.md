# Models Directory

This directory stores trained models for the sentiment analysis system. Models are organized by architecture:

## Directory Structure

```
models/
├── transformer_sentiment/   # Transformer-based sentiment models
├── cnn_sentiment/           # 1D-CNN-based sentiment models
├── lstm_sentiment/          # BiLSTM-based sentiment models
└── README.md                # This file
```

Each model subfolder typically contains:
- `model.keras`           : The trained Keras model file
- `config.json`           : Model configuration (e.g., hyperparameters)
- `label_cols.json`       : List of label columns/classes
- `word_index.pkl`        : Tokenizer word index used for text preprocessing
- `opt_thresholds.npy`    : Optimal thresholds for multi-label classification

## Model Formats

- `.keras` - Keras model files
- `.npy`   - Numpy arrays (e.g., thresholds)
- `.json`  - Configuration and label info
- `.pkl`   - Pickled Python objects (e.g., tokenizer)

## Usage Guidelines

### Saving Models

When saving a model from a notebook or script, use the following pattern:

```python
model.save("project/models/transformer_sentiment/model.keras")
```

### Loading Models

```python
from tensorflow.keras.models import load_model
model = load_model("project/models/transformer_sentiment/model.keras")
```

## Model Versioning

Include a timestamp or version number in the model filename if you want to track different versions. This makes it easier to compare model performance over time.

## Note

Large model files are not tracked in Git. Document training parameters and performance metrics in your notebooks or separate documentation files so models can be recreated if needed. 
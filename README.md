# Speech-Based Sentiment Analysis Project

## Project Overview
This project develops a comprehensive sentiment analysis system that processes both text and speech inputs. We compare traditional neural network approaches (BiLSTM, CNN, Transformer) with state-of-the-art large language models (LLMs), providing valuable insights into the evolution and capabilities of different sentiment analysis techniques.

The system consists of three integrated components:
1. **Multi-architecture text sentiment analysis** using BiLSTM, CNN, and Transformer models
2. **Speech-to-text processing** with OpenAI's Whisper model
3. **LLM-based sentiment analysis** using GPT-4o-mini through LangChain integration

## Quick Start

### Setting Up the Environment
```bash
# Clone the repository
git clone https://github.com/your-username/advanced-topics-predictive-group-j.git
cd advanced-topics-predictive-group-j

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

### Running the Notebooks
Navigate to the `project/notebooks` directory and run the Jupyter notebooks in sequence:
1. `01_text_models.ipynb` - Multi-architecture text sentiment analysis
2. `02_speech_pipeline.ipynb` - Speech-to-text integration and evaluation
3. `03_llm_sentiment.ipynb` - LLM-based sentiment analysis and comparison

## Project Structure
```
advanced-topics-predictive-group-j/
│
├── project/                  # Main project directory
│   ├── __init__.py           # Package initialization
│   │
│   ├── data/                 # Data files and utilities
│   │   ├── audio/            # Audio data
│   │   │   ├── custom/       # Custom team recordings (29 audio files)
│   │   │   ├── samples/      # RAVDESS dataset samples
│   │   │   └── download_samples.py  # Script to download audio files
│   │   │
│   │   ├── processed/        # Processed datasets
│   │   ├── raw/              # Raw datasets
│   │   └── goemotions/       # GoEmotions dataset files
│   │
│   ├── models/               # Saved models
│   │   ├── lstm_sentiment/   # BiLSTM model artifacts
│   │   ├── cnn_sentiment/    # CNN model artifacts
│   │   └── transformer_sentiment/  # Transformer model artifacts (best performer)
│   │
│   ├── notebooks/            # Jupyter notebooks for experimentation
│   │   ├── 01_text_models.ipynb     # Multi-architecture text sentiment analysis
│   │   ├── 02_speech_pipeline.ipynb # Speech-to-text pipeline integration
│   │   └── 03_llm_sentiment.ipynb   # LLM sentiment analysis comparison
│   │
│   ├── results/              # Experiment results and metrics
│   │   ├── whisper_transcripts.csv  # Speech transcription results
│   │   └── llm_outputs.json         # Cached LLM responses
│   │
│   └── src/                  # Source code (production-ready)
│
├── requirements.txt          # Project dependencies
├── check_environment.py     # Environment validation script
└── Advanced Predictive Analytics.pdf  # Project assignment specification
```

## Usage Examples

### Text Sentiment Analysis
```python
# Load the best-performing transformer model
import tensorflow as tf
import pickle
import json

# Load saved model and preprocessing components
model = tf.keras.models.load_model("project/models/transformer_sentiment/model.keras")
with open("project/models/transformer_sentiment/word_index.pkl", "rb") as f:
    word_index = pickle.load(f)
with open("project/models/transformer_sentiment/label_cols.json") as f:
    label_cols = json.load(f)

# Predict sentiment
def predict_sentiment(text):
    # Preprocess text (simplified)
    tokens = [word_index.get(word, 1) for word in text.lower().split()]
    padded = (tokens[:32] + [0]*32)[:32]
    prediction = model.predict([padded])
    return dict(zip(label_cols, prediction[0]))
```

### Speech-to-Text + Sentiment Analysis
```python
import whisper
import numpy as np

# Load Whisper model
whisper_model = whisper.load_model("small")

# Transcribe and analyze
def analyze_speech(audio_path):
    # Transcribe audio
    result = whisper_model.transcribe(audio_path)
    text = result["text"]
    
    # Analyze sentiment using transformer model
    sentiment = predict_sentiment(text)
    
    return {
        "transcription": text,
        "sentiment": sentiment
    }
```

### LLM Sentiment Analysis
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import json

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

# Analyze sentiment
def llm_sentiment_analysis(text):
    prompt = """
    Return **only** a valid JSON object with two keys:
      "sentiment": exactly one of [`positive`, `negative`, `neutral`]
      "emotion": exactly one of [`joy`, `sad`, `anger`, `fear`, `surprise`, `disgust`, `neutral`]
    
    Text: {text}
    JSON:
    """
    
    response = llm.predict(prompt.format(text=text))
    return json.loads(response)
```

## License
This project is part of the Advanced Topics in Predictive Analytics course.

## Acknowledgments
- GoEmotions dataset for comprehensive emotion classification
- OpenAI Whisper for state-of-the-art speech-to-text capabilities
- LangChain for seamless LLM integration
- Team members (Jannik, Marc, Paul) for custom audio recordings


# Speech-Based Sentiment Analysis Project

## Project Overview
This project develops a comprehensive sentiment analysis system that processes both text and speech inputs. We compare traditional neural network approaches with state-of-the-art large language models (LLMs), providing valuable insights into the evolution and capabilities of different sentiment analysis techniques.

The system consists of three integrated components:
1. **Text-based sentiment analysis** using LSTM networks
2. **Speech-to-text processing** with OpenAI's Whisper model
3. **LLM-based sentiment analysis** through LangChain integration

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

### Downloading Audio Samples
```bash
# Run the download script to get RAVDESS dataset samples
python project/data/audio/download_samples.py
```

### Running the Notebooks
Navigate to the `project/notebooks` directory and run the Jupyter notebooks in sequence:
1. `01_text_models.ipynb` - Text-based sentiment analysis
2. `02_whisper_integration.ipynb` - Speech-to-text integration
3. `03_llm_comparison.ipynb` - Comparison of LLMs and traditional models

## Project Structure
```
advanced-topics-predictive-group-j/
│
├── project/                  # Main project directory
│   ├── __init__.py           # Package initialization
│   │
│   ├── data/                 # Data files and utilities
│   │   ├── audio/            # Audio data
│   │   │   ├── samples/      # RAVDESS audio samples
│   │   │   └── download_samples.py  # Script to download audio files
│   │   │
│   │   ├── processed/        # Processed datasets
│   │   ├── raw/              # Raw datasets
│   │   └── goemotions/       # GoEmotions dataset files
│   │
│   ├── models/               # Saved models
│   │   ├── text/             # Text analysis models
│   │   └── speech/           # Speech analysis models
│   │
│   ├── notebooks/            # Jupyter notebooks for experimentation
│   │   ├── 01_text_models.ipynb          # Text-based sentiment analysis
│   │   ├── 02_whisper_integration.ipynb  # Speech-to-text integration
│   │   └── 03_llm_comparison.ipynb       # LLM vs traditional models comparison
│   │
│   └── src/                  # Source code (production-ready)
│
├── requirements.txt          # Project dependencies
└── Advanced Predictive Analytics.pdf  # Project assignment specification
```

## Components in Detail

### 1. Text-based Sentiment Analysis
The text analysis component uses Long Short-Term Memory (LSTM) networks to classify text sentiment:

- **Implementation**: `project/src/text_analysis/models.py`
- **Features**:
  - LSTM architecture with embedding layer
  - Text tokenization and padding
  - Binary sentiment classification (positive/negative)
- **Experimentation**: `project/notebooks/01_text_models.ipynb`

### 2. Speech-to-Text Processing
This component uses OpenAI's Whisper model to transcribe speech:

- **Implementation**: `project/src/speech_analysis/whisper_processor.py`
- **Features**:
  - Support for multiple Whisper model sizes (tiny, base, small, medium, large)
  - Processing of audio files and raw audio data
  - Efficient temporary file handling
- **Sample Data**: RAVDESS dataset audio samples (neutral, happy, sad, angry emotions)
- **Experimentation**: `project/notebooks/02_whisper_integration.ipynb`

### 3. LLM-based Sentiment Analysis
This component uses modern LLMs through LangChain for sentiment analysis:

- **Implementation**: `project/src/llm_integration/llm_analyzer.py`
- **Features**:
  - Support for OpenAI (GPT) and Anthropic (Claude) models
  - JSON-structured sentiment analysis responses
  - Confidence scores and explanations
- **Experimentation**: `project/notebooks/03_llm_comparison.ipynb`

### 4. Web Interface
A Flask-based web application for interacting with the sentiment analysis system:

- **Implementation**: `project/src/web_interface/app.py`
- **Features**:
  - Text input for direct sentiment analysis
  - Audio upload for speech-based sentiment analysis
  - JSON response with sentiment results

## Usage Examples

### Text Sentiment Analysis
```python
from project.src.text_analysis.models import LSTMSentimentModel

# Create and train model
model = LSTMSentimentModel()
model.fit(texts, labels, epochs=5)

# Predict sentiment
results = model.predict(["I love this product!", "This is terrible service."])
```

### Speech-to-Text Transcription
```python
from project.src.speech_analysis.whisper_processor import WhisperTranscriber

# Initialize transcriber
transcriber = WhisperTranscriber(model_name="base")

# Transcribe audio file
result = transcriber.transcribe_file("project/data/audio/samples/03-01-03-01-01-01-01.wav")
text = result["text"]
```

### LLM Sentiment Analysis
```python
from project.src.llm_integration.llm_analyzer import LLMSentimentAnalyzer

# Initialize analyzer (requires API key in environment variables)
analyzer = LLMSentimentAnalyzer(provider="openai")

# Analyze sentiment
result = analyzer.analyze("I absolutely love this new camera!")
```

## License
This project is part of the Advanced Topics in Predictive Analytics course.

## Acknowledgments
- RAVDESS dataset: Ryerson Audio-Visual Database of Emotional Speech and Song
- OpenAI Whisper for speech-to-text capabilities
- LangChain for LLM integration


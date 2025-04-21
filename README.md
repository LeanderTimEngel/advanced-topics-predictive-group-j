# advanced-topics-predictive-group-j
Group J - Project in Advanced Topics in Predictive Analytics

## Project Overview
In this project, you will develop a comprehensive sentiment analysis system that processes both
text and speech inputs. You will compare traditional neural network approaches with state-of-the-art large language models, providing valuable insights into the evolution and capabilities of different sentiment analysis techniques.

## Environment Setup

### Virtual Environment
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to latest version
pip install --upgrade pip
```

### Dependencies
The project requires the following libraries:
```bash
# Core ML frameworks (choose one or both)
# PyTorch ecosystem
pip install torch torchvision torchaudio

# TensorFlow/Keras ecosystem
pip install tensorflow keras

# Speech-to-Text
pip install openai-whisper                   # Whisper STT

# LLM integration
pip install langchain openai anthropic       # LLM integration libraries

# Data processing & evaluation
pip install numpy pandas scikit-learn jiwer  # Data & metrics

# Visualization & demo
pip install matplotlib                       # Plotting
pip install flask                            # Web demo

# Or install everything at once
pip install torch torchvision torchaudio transformers openai-whisper langchain openai anthropic numpy pandas scikit-learn jiwer matplotlib flask tensorflow keras
```

## Project Structure
- `data/` - Dataset files
- `project/` - Main project code

## Team Members
- Group J


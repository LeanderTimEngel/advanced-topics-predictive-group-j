import os
import json
import pickle
import numpy as np
from tensorflow import keras
import re

class SentimentInferenceModel:
    """
    Class for loading and using saved sentiment models for inference.
    """
    def __init__(self, model_dir):
        """
        Load a model and its components from a directory.
        
        Args:
            model_dir: Directory containing saved model and components
        """
        # Load model
        self.model = keras.models.load_model(os.path.join(model_dir, "model.h5"))
        
        # Load vocabulary
        with open(os.path.join(model_dir, "word_index.pkl"), "rb") as f:
            self.word_index = pickle.load(f)
        
        # Load label columns
        with open(os.path.join(model_dir, "label_cols.json"), "r") as f:
            self.label_cols = json.load(f)
        
        # Load configuration
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
            self.max_len = config["max_len"]
            self.model_type = config.get("model_type", "unknown")
        
        # Load thresholds if available
        thresh_path = os.path.join(model_dir, "opt_thresholds.npy")
        if os.path.exists(thresh_path):
            self.thresholds = np.load(thresh_path)
        else:
            self.thresholds = None
        
        print(f"Loaded {self.model_type} model with {len(self.label_cols)} emotion classes")
    
    def preprocess_text(self, text):
        """
        Apply the same preprocessing as during training.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed and padded sequence
        """
        # Apply same cleaning as in training
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)            # Remove HTML tags
        text = re.sub(r'[^a-z0-9\s]', '', text)      # Remove punctuation/special chars
        text = re.sub(r'\s+', ' ', text).strip()     # Collapse whitespace
        
        # Convert to sequence
        seq = [self.word_index.get(w, 1) for w in text.split()]  # 1 is OOV
        
        # Pad sequence
        padded_seq = (seq[:self.max_len] + [0] * self.max_len)[:self.max_len]
        
        # Convert to numpy array and reshape for model
        return np.array([padded_seq], dtype=np.int16)
    
    def predict(self, text, threshold=None, top_k=3):
        """
        Predict emotions for a text input.
        
        Args:
            text: Raw text string
            threshold: Classification threshold (use class-specific if None)
            top_k: Number of top emotions to return
            
        Returns:
            Dictionary with predicted emotions and scores
        """
        # Preprocess the text
        seq = self.preprocess_text(text)
        
        # Get model predictions
        pred_scores = self.model.predict(seq, verbose=0)[0]
        
        # Apply thresholds (either per-class or global)
        if threshold is None and self.thresholds is not None:
            # Use per-class optimized thresholds
            emotions_above_threshold = []
            for i, score in enumerate(pred_scores):
                if score >= self.thresholds[i]:
                    emotions_above_threshold.append((self.label_cols[i], float(score)))
        else:
            # Use fixed threshold (default 0.5 if not provided)
            threshold = threshold if threshold is not None else 0.5
            emotions_above_threshold = [(self.label_cols[i], float(score)) 
                                       for i, score in enumerate(pred_scores) 
                                       if score >= threshold]
        
        # Get top K emotions
        top_emotions = sorted([(self.label_cols[i], float(score)) 
                              for i, score in enumerate(pred_scores)], 
                             key=lambda x: x[1], reverse=True)[:top_k]
        
        return {
            "raw_scores": {self.label_cols[i]: float(score) for i, score in enumerate(pred_scores)},
            "emotions_above_threshold": emotions_above_threshold,
            "top_emotions": top_emotions
        }
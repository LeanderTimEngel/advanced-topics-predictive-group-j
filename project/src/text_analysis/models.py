"""
Text-based Sentiment Analysis Models
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class LSTMSentimentModel:
    """LSTM model for text sentiment analysis."""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=200):
        """Initialize the LSTM sentiment model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding layer
            max_length: Maximum length of input sequences
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.model = None
        
    def build_model(self):
        """Build the LSTM model architecture."""
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fit(self, texts, labels, validation_split=0.2, epochs=10, batch_size=32):
        """Train the model on the provided texts and labels.
        
        Args:
            texts: List of text strings
            labels: Binary sentiment labels (0 or 1)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
            
        # Prepare data
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
        
        # Train model
        history = self.model.fit(
            padded_sequences,
            labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def predict(self, texts):
        """Predict sentiment for the given texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Predicted sentiment probabilities
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
        
        return self.model.predict(padded_sequences) 
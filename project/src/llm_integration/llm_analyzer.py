"""
LLM Integration for Sentiment Analysis
"""
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class LLMSentimentAnalyzer:
    """Analyze sentiment using LLMs via LangChain."""
    
    def __init__(self, provider="openai", model_name=None):
        """Initialize LLM sentiment analyzer.
        
        Args:
            provider: LLM provider ('openai' or 'anthropic')
            model_name: Specific model to use (if None, uses default)
        """
        self.provider = provider
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.chain = self._create_chain()
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on provider."""
        if self.provider == "openai":
            model = self.model_name or "gpt-3.5-turbo"
            return ChatOpenAI(model_name=model, temperature=0)
        elif self.provider == "anthropic":
            model = self.model_name or "claude-3-sonnet-20240229"
            return ChatAnthropic(model=model, temperature=0)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _create_chain(self):
        """Create the sentiment analysis chain."""
        template = """
        You are a sentiment analysis expert. Analyze the following text and determine its sentiment.
        
        Text: {text}
        
        Provide your analysis as a JSON object with the following fields:
        - sentiment: The overall sentiment (positive, neutral, negative)
        - confidence: A number between 0 and 1 indicating your confidence
        - explanation: A brief explanation of your reasoning
        
        Your response should be JSON format only, no additional text.
        """
        
        prompt = PromptTemplate(template=template, input_variables=["text"])
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def analyze(self, text):
        """Analyze the sentiment of the provided text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        result = self.chain.invoke({"text": text})
        return result["text"]  # The result is returned in the 'text' field 
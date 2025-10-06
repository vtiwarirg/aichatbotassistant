import json
import logging
import random
import os
from datetime import datetime
from typing import Dict, Optional, Any, List
from config.config import Config
from services.nlp_processor import NLPProcessor
from services.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)

class MLChatbotService:
    """Advanced ML-powered chatbot using NLTK, spaCy, and scikit-learn"""
    
    def __init__(self):
        self.config = Config()
        logger.info("Initializing ML Chatbot Service")
        
        # Initialize NLP processor
        self.nlp_processor = NLPProcessor()
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier(self.config.CHATBOT_MODEL_PATH)
        
        # Load intents and responses
        self.intents_data = self._load_intents_data()
        self.responses_by_intent = self._build_response_mapping()
        
        # Initialize AI provider if enabled
        self.ai_enabled = self.config.CHATBOT_USE_AI
        self.ml_enabled = self.config.CHATBOT_USE_ML
        self.ai_provider = None
        
        if self.ai_enabled:
            self._initialize_ai_provider()
        
        # Train or load ML model
        if self.ml_enabled:
            self._setup_ml_model()
        
        logger.info("ML Chatbot Service initialized successfully")
    
    def _load_intents_data(self):
        """Load intents data from JSON file"""
        try:
            intents_path = self.config.CHATBOT_INTENTS_FILE
            if not os.path.exists(intents_path):
                logger.error(f"Intents file not found: {intents_path}")
                return None
            
            with open(intents_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            logger.info(f"Loaded {len(data.get('intents', []))} intents from {intents_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading intents data: {e}")
            return None
    
    def _build_response_mapping(self):
        """Build mapping from intent tags to responses"""
        if not self.intents_data:
            return {}
        
        response_mapping = {}
        for intent in self.intents_data.get('intents', []):
            tag = intent.get('tag')
            responses = intent.get('responses', [])
            if tag and responses:
                response_mapping[tag] = responses
        
        logger.info(f"Built response mapping for {len(response_mapping)} intents")
        return response_mapping
    
    def _setup_ml_model(self):
        """Setup ML model - train if needed or load existing"""
        try:
            # Try to load existing model first
            if not self.intent_classifier.load_model():
                logger.info("No existing model found. Training new model...")
                
                # Train new model
                success = self.intent_classifier.train_model(
                    self.config.CHATBOT_INTENTS_FILE,
                    self.config.CHATBOT_ML_MODEL_TYPE
                )
                
                if success:
                    logger.info("ML model trained successfully")
                else:
                    logger.error("Failed to train ML model")
                    self.ml_enabled = False
            else:
                logger.info("ML model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error setting up ML model: {e}")
            self.ml_enabled = False
    
    def _initialize_ai_provider(self):
        """Initialize AI provider if needed"""
        if not self.ai_enabled:
            return
        
        provider = self.config.CHATBOT_AI_PROVIDER.lower()
        print(f"Initializing AI provider: {provider}")
        try:
            if provider == 'openai' and self.config.OPENAI_API_KEY:
                self._initialize_openai()
            elif provider == 'ollama':
                self._initialize_ollama()
            elif provider == 'huggingface':
                self._initialize_huggingface()
            else:
                logger.warning(f"AI provider '{provider}' not available or not configured.")
                self.ai_enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize AI provider '{provider}': {e}")
            self.ai_enabled = False
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            openai.api_key = self.config.OPENAI_API_KEY
            self.ai_provider = 'openai'
            logger.info("OpenAI provider initialized")
        except ImportError:
            logger.warning("OpenAI library not installed")
            raise
    
    def _initialize_ollama(self):
        """Initialize Ollama client"""
        try:
            import requests
            response = requests.get(f"{self.config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                self.ai_provider = 'ollama'
                logger.info("Ollama provider initialized")
            else:
                raise ConnectionError("Ollama server not responding")
        except (ImportError, ConnectionError, requests.RequestException) as e:
            logger.warning(f"Ollama not available: {e}")
            raise
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face transformers"""
        try:
            from transformers import pipeline
            self.ai_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium")
            self.ai_provider = 'huggingface'
            logger.info("Hugging Face provider initialized")
        except ImportError:
            logger.warning("Transformers library not installed")
            raise
    
    def get_response(self, user_message: str, session_id: str = None, chat_history: List = None) -> Dict[str, Any]:
        """Main method to get chatbot response"""
        try:
            logger.info(f"Processing message: '{user_message}'")
            
            # Preprocess the message
            processed_message = self.nlp_processor.preprocess_text(user_message)
            
            # Try ML-based intent classification first
            if self.ml_enabled:
                response = self._get_ml_response(user_message, processed_message)
                if response:
                    return response
            
            # Fallback to rule-based response
            response = self._get_rule_based_response(user_message, processed_message)
            if response:
                return response
            
            # Final fallback to AI if available
            if self.ai_enabled:
                return self._get_ai_response(user_message, session_id, chat_history)
            
            # Ultimate fallback
            return self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"Error getting chatbot response: {e}")
            return self._get_fallback_response()
    
    def _get_ml_response(self, original_message: str, processed_message: str) -> Optional[Dict[str, Any]]:
        """Get response using ML intent classification"""
        try:
            # Predict intent
            predicted_intent, confidence = self.intent_classifier.predict_intent(
                original_message, 
                self.config.CHATBOT_CONFIDENCE_THRESHOLD
            )
            
            if predicted_intent and confidence >= self.config.CHATBOT_CONFIDENCE_THRESHOLD:
                # Get responses for this intent
                responses = self.responses_by_intent.get(predicted_intent, [])
                if responses:
                    response_text = random.choice(responses)
                    
                    # Add some entity extraction for personalization
                    entities = self.nlp_processor.extract_entities(original_message)
                    
                    return {
                        'response': response_text,
                        'type': f'ml_intent_{predicted_intent}',
                        'confidence': confidence,
                        'intent': predicted_intent,
                        'entities': entities,
                        'timestamp': datetime.now().isoformat()
                    }
            
            logger.info(f"ML prediction below threshold: intent={predicted_intent}, confidence={confidence}")
            return None
            
        except Exception as e:
            logger.error(f"Error in ML response generation: {e}")
            return None
    
    def _get_rule_based_response(self, original_message: str, processed_message: str) -> Optional[Dict[str, Any]]:
        """Enhanced rule-based responses as fallback"""
        message_lower = original_message.lower()
        
        # Enhanced keyword matching with similarity
        intent_keywords = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'goodbye': ['bye', 'goodbye', 'see you', 'thanks', 'thank you', 'farewell'],
            'services': ['service', 'services', 'what do you do', 'help with', 'offer', 'provide'],
            'contact': ['contact', 'email', 'phone', 'reach', 'get in touch', 'call'],
            'business_hours': ['hours', 'time', 'open', 'available', 'schedule', 'when'],
            'pricing': ['price', 'cost', 'pricing', 'how much', 'expensive', 'budget'],
            'timeline': ['timeline', 'how long', 'duration', 'time', 'delivery'],
            'portfolio': ['portfolio', 'examples', 'work', 'projects', 'showcase'],
            'technology': ['technology', 'tech', 'stack', 'framework', 'language'],
            'help': ['help', 'support', 'assistance', 'problem', 'issue']
        }
        
        # Find best matching intent
        best_intent = None
        best_score = 0
        
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    # Calculate similarity score
                    score = self.nlp_processor.get_text_similarity(processed_message, keyword)
                    if score > best_score:
                        best_score = score
                        best_intent = intent
        
        # Use intent if good enough match
        if best_intent and best_score > 0.1:  # Threshold for rule-based matching
            responses = self.responses_by_intent.get(best_intent, [])
            if responses:
                response_text = random.choice(responses)
                return {
                    'response': response_text,
                    'type': f'rule_based_{best_intent}',
                    'confidence': best_score,
                    'intent': best_intent,
                    'entities': [],
                    'timestamp': datetime.now().isoformat()
                }
        
        return None
    
    def _get_ai_response(self, user_message: str, session_id: str, chat_history: List) -> Dict[str, Any]:
        """Get AI-powered response"""
        try:
            if self.ai_provider == 'openai':
                return self._get_openai_response(user_message, chat_history)
            elif self.ai_provider == 'ollama':
                return self._get_ollama_response(user_message, chat_history)
            elif self.ai_provider == 'huggingface':
                return self._get_huggingface_response(user_message, chat_history)
        except Exception as e:
            logger.error(f"AI provider '{self.ai_provider}' error: {e}")
        
        return self._get_fallback_response()
    
    def _get_openai_response(self, user_message: str, chat_history: List) -> Dict[str, Any]:
        """Get response from OpenAI"""
        try:
            import openai
            
            messages = [
                {"role": "system", "content": "You are a helpful customer service assistant for a web development company. Keep responses concise and friendly."},
                {"role": "user", "content": user_message}
            ]
            
            response = openai.ChatCompletion.create(
                model=self.config.CHATBOT_MODEL,
                messages=messages,
                max_tokens=self.config.CHATBOT_MAX_TOKENS,
                temperature=self.config.CHATBOT_TEMPERATURE
            )
            
            return {
                'response': response.choices[0].message.content.strip(),
                'type': 'ai_openai',
                'confidence': 1.0,  # AI responses are considered confident
                'intent': 'ai_generated',
                'entities': [],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return self._get_fallback_response()
    
    def _get_ollama_response(self, user_message: str, chat_history: List) -> Dict[str, Any]:
        """Get response from Ollama"""
        try:
            import requests
            
            prompt = f"You are a helpful customer service assistant. User asks: {user_message}\nRespond helpfully and concisely:"
            
            response = requests.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'response': result.get('response', '').strip(),
                    'type': 'ai_ollama',
                    'confidence': 1.0,  # AI responses are considered confident
                    'intent': 'ai_generated',
                    'entities': [],
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Ollama error: {e}")
        
        return self._get_fallback_response()
    
    def _get_huggingface_response(self, user_message: str, chat_history: List) -> Dict[str, Any]:
        """Get response from Hugging Face model"""
        try:
            response = self.ai_pipeline(
                f"User: {user_message}\nBot:",
                max_length=100,
                num_return_sequences=1,
                temperature=0.7
            )
            
            generated_text = response[0]['generated_text']
            bot_response = generated_text.split("Bot:")[-1].strip()
            
            return {
                'response': bot_response,
                'type': 'ai_huggingface',
                'confidence': 1.0,  # AI responses are considered confident
                'intent': 'ai_generated',
                'entities': [],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Hugging Face error: {e}")
        
        return self._get_fallback_response()
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Ultimate fallback response"""
        fallback_responses = [
            "I'm here to help! Could you please rephrase your question or use our contact form for detailed assistance?",
            "I'd be happy to help you with that! For specific information, please contact us through our website.",
            "That's a great question! Please reach out through our contact form so our team can provide you with detailed assistance.",
            "I want to make sure I give you the best answer possible. Please use our contact form to get in touch with our team."
        ]
        
        return {
            'response': random.choice(fallback_responses),
            'type': 'fallback',
            'confidence': 1.0,  # Fallback responses are always confident
            'intent': 'fallback',
            'entities': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_intent_probabilities(self, user_message: str) -> Dict[str, float]:
        """Get probabilities for all intents (useful for debugging)"""
        if self.ml_enabled:
            return self.intent_classifier.get_all_intents_probabilities(user_message)
        return {}
    
    def add_feedback(self, user_message: str, correct_intent: str):
        """Add user feedback for model improvement"""
        if self.ml_enabled:
            self.intent_classifier.retrain_with_feedback(user_message, correct_intent)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'ml_enabled': self.ml_enabled,
            'ai_enabled': self.ai_enabled,
            'ai_provider': self.ai_provider,
            'model_path': self.config.CHATBOT_MODEL_PATH,
            'confidence_threshold': self.config.CHATBOT_CONFIDENCE_THRESHOLD,
            'intents_count': len(self.responses_by_intent),
            'available_intents': list(self.responses_by_intent.keys())
        }
    
    def is_available(self) -> bool:
        """Check if the chatbot service is available and ready to use"""
        return self.ml_enabled and bool(self.responses_by_intent)
"""
Complete ChatbotFacade implementation using ML tools
Uses NLTK, spaCy, scikit-learn for intelligent responses
"""
import logging
import json
import os
import csv
import pickle
import random
from datetime import datetime
from typing import Dict, Any, List
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

logger = logging.getLogger(__name__)

class ChatbotFacade:
    """Complete ML-powered chatbot facade"""
    
    def __init__(self):
        self._available = False
        self._initialized = False
        self.intents_file = 'data/chatbot_intents.json'
        self.model_file = 'models/chatbot_model.pkl'
        self.conversations_file = 'data/analytics/conversations.csv'
        self.intents_data = {'intents': []}
        self.ml_model = None
        self.nlp = None
        self._init_service()
    
    def _init_service(self):
        """Initialize complete ML chatbot service"""
        try:
            logger.info("Initializing Complete ML ChatbotFacade...")
            
            # Download required NLTK data
            self._download_nltk_data()
            
            # Load spaCy model
            self._load_spacy_model()
            
            # Load intents data
            self._load_intents_data()
            
            # Initialize or load ML model
            self._init_ml_model()
            
            # Setup CSV logging
            self._setup_csv_logging()
            
            self._available = True
            self._initialized = True
            logger.info("ChatbotFacade initialized successfully with ML capabilities")
            
        except Exception as e:
            logger.error(f"Error initializing ChatbotFacade: {e}")
            self._available = False
            self._initialized = False
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def _load_spacy_model(self):
        """Load spaCy model with fallback"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy en_core_web_sm not found, using basic processing")
            self.nlp = None
    
    def _load_intents_data(self):
        """Load intents data from JSON file"""
        if os.path.exists(self.intents_file):
            with open(self.intents_file, 'r', encoding='utf-8') as f:
                self.intents_data = json.load(f)
            logger.info(f"Loaded {len(self.intents_data.get('intents', []))} intents")
        else:
            logger.warning(f"Intents file not found: {self.intents_file}")
            # Create basic intents data
            self._create_basic_intents()
    
    def _create_basic_intents(self):
        """Create basic intents if file doesn't exist"""
        self.intents_data = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["hello", "hi", "hey", "good morning", "good afternoon"],
                    "responses": ["Hello! How can I help you today?", "Hi there! What can I do for you?"]
                },
                {
                    "tag": "time",
                    "patterns": ["what time is it", "current time", "tell me the time"],
                    "responses": ["The current time is {current_time}"]
                },
                {
                    "tag": "date", 
                    "patterns": ["what date is it", "current date", "today's date"],
                    "responses": ["Today's date is {current_date}"]
                }
            ]
        }
    
    def _init_ml_model(self):
        """Initialize or load ML model"""
        os.makedirs('models', exist_ok=True)
        
        if os.path.exists(self.model_file):
            self._load_model()
        else:
            self._train_model()
    
    def _load_model(self):
        """Load pre-trained model"""
        try:
            with open(self.model_file, 'rb') as f:
                self.ml_model = pickle.load(f)
            logger.info("ML model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._train_model()
    
    def _train_model(self):
        """Train ML model using scikit-learn"""
        try:
            logger.info("Training ML model...")
            
            # Prepare training data
            patterns = []
            labels = []
            
            for intent in self.intents_data.get('intents', []):
                tag = intent['tag']
                for pattern in intent.get('patterns', []):
                    patterns.append(self._preprocess_text(pattern))
                    labels.append(tag)
            
            if len(patterns) == 0:
                logger.warning("No training data found")
                return
            
            # Create ML pipeline
            self.ml_model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    lowercase=True
                )),
                ('classifier', LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ))
            ])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                patterns, labels, test_size=0.2, random_state=42
            )
            
            # Train model
            self.ml_model.fit(X_train, y_train)
            
            # Evaluate model
            if len(X_test) > 0:
                y_pred = self.ml_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"Model accuracy: {accuracy:.3f}")
            
            # Save model
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.ml_model, f)
            
            logger.info("ML model trained and saved successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            self.ml_model = None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text using NLTK and spaCy"""
        try:
            # Basic preprocessing
            text = text.lower().strip()
            
            # Use spaCy if available
            if self.nlp:
                doc = self.nlp(text)
                # Remove stop words and punctuation, lemmatize
                tokens = [token.lemma_ for token in doc 
                         if not token.is_stop and not token.is_punct and token.text.strip()]
                return ' '.join(tokens)
            else:
                # Fallback to basic NLTK processing
                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize
                
                stop_words = set(stopwords.words('english'))
                tokens = word_tokenize(text)
                tokens = [word for word in tokens if word.lower() not in stop_words]
                return ' '.join(tokens)
                
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text.lower()
    
    def _setup_csv_logging(self):
        """Setup CSV logging for conversations"""
        os.makedirs('data/analytics', exist_ok=True)
        
        if not os.path.exists(self.conversations_file):
            with open(self.conversations_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'session_id', 'user_message', 'bot_response', 
                               'intent', 'confidence', 'response_type'])
    
    def _log_conversation(self, user_message: str, response_data: Dict, session_id: str):
        """Log conversation to CSV"""
        try:
            with open(self.conversations_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    session_id,
                    user_message,
                    response_data.get('response', ''),
                    response_data.get('intent', ''),
                    response_data.get('confidence', 0),
                    response_data.get('response_type', '')
                ])
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
    
    def _get_dynamic_content(self, response_text: str) -> str:
        """Replace dynamic placeholders with actual content"""
        current_time = datetime.now().strftime("%I:%M %p")
        current_date = datetime.now().strftime("%B %d, %Y")
        current_day = datetime.now().strftime("%A")
        current_year = datetime.now().year
        
        # Get greeting based on time
        hour = datetime.now().hour
        if 5 <= hour < 12:
            greeting = "Good morning"
        elif 12 <= hour < 17:
            greeting = "Good afternoon"
        elif 17 <= hour < 21:
            greeting = "Good evening"
        else:
            greeting = "Good evening"
        
        replacements = {
            '{{DYNAMIC_TIME}}': current_time,
            '{{DYNAMIC_DATE}}': current_date,
            '{{DYNAMIC_DATETIME}}': f'{current_date} at {current_time}',
            '{{DYNAMIC_DAY}}': current_day,
            '{{DYNAMIC_YEAR}}': str(current_year),
            '{{DYNAMIC_GREETING}}': greeting,
            '{current_time}': current_time,
            '{current_date}': current_date,
            '{current_day}': current_day,
            '{current_year}': str(current_year),
            '{greeting}': greeting
        }
        
        for placeholder, value in replacements.items():
            response_text = response_text.replace(placeholder, value)
        
        return response_text
    
    def is_available(self) -> bool:
        """Check if chatbot service is available"""
        return self._available and self._initialized
    
    def get_response(self, user_message: str, session_id: str = 'default') -> Dict[str, Any]:
        """Get intelligent response using ML model"""
        try:
            # Preprocess user message
            processed_text = self._preprocess_text(user_message)
            
            # Try ML prediction first
            if self.ml_model and processed_text:
                try:
                    # Get prediction and probabilities
                    predicted_intent = self.ml_model.predict([processed_text])[0]
                    probabilities = self.ml_model.predict_proba([processed_text])[0]
                    max_confidence = max(probabilities)
                    
                    # Use ML prediction if confidence is high enough
                    if max_confidence > 0.3:
                        response_data = self._generate_response_for_intent(
                            predicted_intent, user_message, max_confidence
                        )
                        response_data['response_type'] = 'ml_prediction'
                    else:
                        # Fallback to rule-based response
                        response_data = self._get_rule_based_response(user_message)
                        response_data['response_type'] = 'rule_based'
                        
                except Exception as e:
                    logger.error(f"ML prediction error: {e}")
                    response_data = self._get_rule_based_response(user_message)
                    response_data['response_type'] = 'rule_based'
            else:
                # Fallback to rule-based response
                response_data = self._get_rule_based_response(user_message)
                response_data['response_type'] = 'rule_based'
            
            # Process dynamic content
            response_data['response'] = self._get_dynamic_content(response_data['response'])
            
            # Log conversation
            self._log_conversation(user_message, response_data, session_id)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return {
                'response': 'Sorry, I encountered an error. Please try again.',
                'intent': 'error',
                'confidence': 0.0,
                'success': False,
                'response_type': 'error'
            }
    
    def _generate_response_for_intent(self, intent: str, user_message: str, confidence: float) -> Dict[str, Any]:
        """Generate response for predicted intent"""
        try:
            # Find intent in data
            for intent_data in self.intents_data.get('intents', []):
                if intent_data['tag'] == intent:
                    responses = intent_data.get('responses', [])
                    if responses:
                        response_text = random.choice(responses)
                        return {
                            'response': response_text,
                            'intent': intent,
                            'confidence': float(confidence),
                            'success': True
                        }
            
            # If intent not found, use fallback
            return self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"Error generating intent response: {e}")
            return self._get_fallback_response()
    
    def _get_rule_based_response(self, user_message: str) -> Dict[str, Any]:
        """Get response using rule-based matching"""
        message_lower = user_message.lower()
        
        # Time queries
        if any(word in message_lower for word in ['time', 'clock']):
            current_time = datetime.now().strftime("%I:%M %p")
            return {
                'response': f'The current time is {current_time}',
                'intent': 'time',
                'confidence': 0.9,
                'success': True
            }
        
        # Date queries
        elif any(word in message_lower for word in ['date', 'today']):
            current_date = datetime.now().strftime("%B %d, %Y")
            return {
                'response': f'Today\'s date is {current_date}',
                'intent': 'date',
                'confidence': 0.9,
                'success': True
            }
        
        # Greetings
        elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            responses = [
                'Hello! Welcome to our AI assistant. How can I help you today?',
                'Hi there! I\'m here to assist you with various business needs.',
                'Welcome! I\'m ready to help with your questions.'
            ]
            return {
                'response': random.choice(responses),
                'intent': 'greeting',
                'confidence': 0.8,
                'success': True
            }
        
        # Services
        elif any(word in message_lower for word in ['services', 'help', 'what can you do', 'capabilities']):
            return {
                'response': 'I can help with: Customer Service inquiries, HR policy questions, IT support, account management, billing assistance, and general business information. What specific area do you need help with?',
                'intent': 'services',
                'confidence': 0.8,
                'success': True
            }
        
        # Weather
        elif any(word in message_lower for word in ['weather', 'forecast', 'temperature', 'rain']):
            return {
                'response': 'I don\'t have access to current weather data, but I recommend checking your local weather app or visiting weather.com for accurate, up-to-date weather information.',
                'intent': 'weather',
                'confidence': 0.8,
                'success': True
            }
        
        # Goodbye
        elif any(word in message_lower for word in ['bye', 'goodbye', 'thanks', 'thank you']):
            responses = [
                'Thank you for using our AI assistant! Feel free to reach out anytime.',
                'Goodbye! I\'m here whenever you need assistance.',
                'Thanks for chatting! Have a great day!'
            ]
            return {
                'response': random.choice(responses),
                'intent': 'goodbye',
                'confidence': 0.8,
                'success': True
            }
        
        # Fallback
        else:
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Get fallback response for unknown queries"""
        responses = [
            'I\'d be happy to help you with that! Could you please provide more specific details?',
            'I want to make sure I give you the most helpful response. Could you tell me more?',
            'I\'m here to help! Could you please rephrase your question or provide more details?'
        ]
        return {
            'response': random.choice(responses),
            'intent': 'fallback',
            'confidence': 0.5,
            'success': True
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        try:
            intents_count = len(self.intents_data.get('intents', []))
            has_ml_model = self.ml_model is not None
            has_spacy = self.nlp is not None
            
            return {
                'ml_enabled': has_ml_model,
                'intents_count': intents_count,
                'model_loaded': has_ml_model,
                'spacy_available': has_spacy,
                'conversation_count': self._get_conversation_count(),
                'service_type': 'complete_ml_facade'
            }
        except Exception:
            return {
                'ml_enabled': False,
                'intents_count': 0,
                'model_loaded': False,
                'spacy_available': False,
                'conversation_count': 0,
                'service_type': 'error'
            }
    
    def _get_conversation_count(self) -> int:
        """Get total conversation count from CSV"""
        try:
            if os.path.exists(self.conversations_file):
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    return sum(1 for _ in f) - 1  # Subtract header row
            return 0
        except Exception:
            return 0
    
    def get_service_status(self) -> Dict[str, bool]:
        """Get comprehensive service status"""
        return {
            'ChatbotFacade': self._initialized,
            'Intents File': os.path.exists(self.intents_file),
            'Intents Data': len(self.intents_data.get('intents', [])) > 0,
            'ML Model': self.ml_model is not None,
            'spaCy NLP': self.nlp is not None,
            'CSV Logging': os.path.exists(self.conversations_file),
            'File System': True
        }
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Get comprehensive analytics report"""
        try:
            # Read conversations from CSV
            conversations = []
            if os.path.exists(self.conversations_file):
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    conversations = list(reader)
            
            total_conversations = len(conversations)
            unique_sessions = len(set(conv.get('session_id', '') for conv in conversations))
            
            # Calculate average confidence
            confidences = [float(conv.get('confidence', 0)) for conv in conversations 
                          if conv.get('confidence')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Intent distribution
            intent_counts = {}
            for conv in conversations:
                intent = conv.get('intent', 'unknown')
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            return {
                'summary': {
                    'total_conversations': total_conversations,
                    'unique_sessions': unique_sessions,
                    'average_confidence': round(avg_confidence, 3),
                    'service_type': 'complete_ml_facade'
                },
                'intent_distribution': intent_counts,
                'top_intents': sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Error generating analytics: {e}")
            return {
                'summary': {
                    'total_conversations': 0,
                    'unique_sessions': 0,
                    'average_confidence': 0.0,
                    'service_type': 'error'
                },
                'error': str(e)
            }
    
    def retrain_model(self) -> bool:
        """Retrain the ML model with current data"""
        try:
            logger.info("Retraining ML model...")
            self._train_model()
            return self.ml_model is not None
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return False
    
    def get_intent_probabilities(self, user_message: str) -> Dict[str, float]:
        """Get probability scores for all intents"""
        try:
            if not self.ml_model:
                return {}
            
            processed_text = self._preprocess_text(user_message)
            probabilities = self.ml_model.predict_proba([processed_text])[0]
            classes = self.ml_model.classes_
            
            return dict(zip(classes, probabilities))
            
        except Exception as e:
            logger.error(f"Error getting probabilities: {e}")
            return {}
    
    def export_conversations_csv(self) -> str:
        """Export conversations to timestamped CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = f"data/analytics/conversations_export_{timestamp}.csv"
            
            if os.path.exists(self.conversations_file):
                import shutil
                shutil.copy2(self.conversations_file, export_file)
                return export_file
            
            return ""
        except Exception as e:
            logger.error(f"Error exporting conversations: {e}")
            return ""
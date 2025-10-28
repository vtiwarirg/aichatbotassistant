"""
Optimized ML Chatbot Service with clean architecture
Handles ML-powered conversation with proper separation of concerns
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Structured chat response data"""
    response: str
    intent: str
    confidence: float
    response_type: str
    entities: List = None
    sentiment: Dict = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'response': self.response,
            'intent': self.intent,
            'confidence': self.confidence,
            'response_type': self.response_type,
            'entities': self.entities or [],
            'sentiment': self.sentiment or {},
            'success': self.success
        }

class MLChatbotService:
    """
    Optimized ML-powered chatbot service with clean architecture
    Handles intent classification, response generation, and conversation logging
    """
    
    def __init__(self, intents_file: str = None):
        """Initialize the ML chatbot service"""
        self.intents_file = intents_file or 'data/chatbot_intents.json'
        self.confidence_threshold = 0.3
        self._initialized = False
        self._available = False
        
        # Service components
        self.intent_classifier = None
        self.nlp_processor = None
        self.response_generator = None
        self.conversation_logger = None
        self.dynamic_handler = None
        
        # Initialize the service
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize all service components"""
        try:
            logger.info("Initializing ML Chatbot Service...")
            
            # Initialize NLP processor
            from .nlp_processor import NLPProcessor
            self.nlp_processor = NLPProcessor()
            
            # Initialize intent classifier
            from .intent_classifier import IntentClassifier
            self.intent_classifier = IntentClassifier()
            
            # Load or train model
            if not self._load_or_train_model():
                raise Exception("Failed to load or train ML model")
            
            # Initialize response generator
            from .response_generator import ResponseGenerator
            self.response_generator = ResponseGenerator(self.intents_file)
            
            # Initialize conversation logger
            from .conversation_logger import ConversationLogger
            self.conversation_logger = ConversationLogger()
            
            # Initialize dynamic response handler
            from .dynamic_response_handler import DynamicResponseHandler
            self.dynamic_handler = DynamicResponseHandler()
            
            self._available = True
            self._initialized = True
            
            logger.info("ML Chatbot Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Chatbot Service: {e}")
            self._available = False
            self._initialized = False
    
    def _load_or_train_model(self) -> bool:
        """Load existing model or train new one"""
        try:
            # Try to load existing model
            if self.intent_classifier.load_model():
                logger.info("Existing ML model loaded successfully")
                return True
            
            # Train new model if none exists
            logger.info("Training new ML model...")
            if os.path.exists(self.intents_file):
                success = self.intent_classifier.train_model(
                    self.intents_file, 
                    model_type='logistic_regression'
                )
                if success:
                    logger.info("New ML model trained successfully")
                    return True
                else:
                    logger.error("Failed to train ML model")
                    return False
            else:
                logger.error(f"Intents file not found: {self.intents_file}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading or training model: {e}")
            return False
    
    def get_response(self, user_message: str, session_id: str = 'default') -> Dict[str, Any]:
        """
        Get chatbot response for user message
        
        Args:
            user_message: User's input message
            session_id: Session identifier for conversation tracking
            
        Returns:
            Dict containing response data
        """
        if not self.is_available():
            return self._get_fallback_response("Service unavailable").to_dict()
        
        try:
            # Validate input
            if not user_message or not user_message.strip():
                return self._get_fallback_response("Empty message").to_dict()
            
            # Process message with NLP
            processed_data = self._process_message(user_message.strip())
            
            # Classify intent
            intent_result = self._classify_intent(processed_data['processed_text'])
            
            # Generate response
            response = self._generate_response(
                user_message=user_message,
                processed_data=processed_data,
                intent_result=intent_result
            )
            
            # Log conversation
            self._log_conversation(user_message, response, session_id)
            
            return response.to_dict()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._get_error_response().to_dict()
    
    def _process_message(self, message: str) -> Dict[str, Any]:
        """Process message with NLP"""
        try:
            if self.nlp_processor:
                return self.nlp_processor.process_text(message)
            else:
                # Fallback processing
                return {
                    'processed_text': message.lower(),
                    'entities': [],
                    'sentiment': {'sentiment_score': 0, 'sentiment_label': 'neutral'}
                }
        except Exception as e:
            logger.error(f"Error in NLP processing: {e}")
            return {
                'processed_text': message.lower(),
                'entities': [],
                'sentiment': {'sentiment_score': 0, 'sentiment_label': 'neutral'}
            }
    
    def _classify_intent(self, processed_text: str) -> Dict[str, Any]:
        """Classify intent from processed text"""
        try:
            if self.intent_classifier:
                intent, confidence = self.intent_classifier.predict(processed_text)
                return {
                    'intent': intent,
                    'confidence': confidence,
                    'above_threshold': confidence >= self.confidence_threshold
                }
            else:
                return {
                    'intent': 'unknown',
                    'confidence': 0.0,
                    'above_threshold': False
                }
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'above_threshold': False
            }
    
    def _generate_response(self, user_message: str, processed_data: Dict, 
                          intent_result: Dict) -> ChatResponse:
        """Generate response based on intent and context"""
        try:
            intent = intent_result['intent']
            confidence = intent_result['confidence']
            
            # Try to get intent-based response
            if intent_result['above_threshold'] and self.response_generator:
                response_text = self.response_generator.get_response_for_intent(
                    intent=intent,
                    user_message=user_message,
                    entities=processed_data.get('entities', []),
                    sentiment=processed_data.get('sentiment', {})
                )
                
                if response_text:
                    # Process dynamic content
                    if self.dynamic_handler:
                        response_text = self.dynamic_handler.process_response(
                            response_text, user_message
                        )
                    
                    return ChatResponse(
                        response=response_text,
                        intent=intent,
                        confidence=confidence,
                        response_type='ml_intent',
                        entities=processed_data.get('entities', []),
                        sentiment=processed_data.get('sentiment', {})
                    )
            
            # Fallback to similarity-based response
            similarity_response = self._get_similarity_response(user_message, processed_data)
            if similarity_response:
                return similarity_response
            
            # Final fallback
            return self._get_fallback_response("No matching intent found")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_error_response()
    
    def _get_similarity_response(self, user_message: str, processed_data: Dict) -> Optional[ChatResponse]:
        """Get response based on text similarity (fallback method)"""
        try:
            # This could be enhanced with vector similarity
            # For now, return None to use fallback
            return None
        except Exception as e:
            logger.error(f"Error in similarity response: {e}")
            return None
    
    def _log_conversation(self, user_message: str, response: ChatResponse, session_id: str):
        """Log conversation to analytics"""
        try:
            if self.conversation_logger:
                self.conversation_logger.log_conversation(
                    user_message=user_message,
                    bot_response=response.response,
                    intent=response.intent,
                    confidence=response.confidence,
                    response_type=response.response_type,
                    entities=response.entities,
                    session_id=session_id
                )
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
    
    def _get_fallback_response(self, reason: str = "Unknown") -> ChatResponse:
        """Get fallback response when no intent matches"""
        fallback_responses = [
            "I'd be happy to help you with that! Could you please provide more specific details?",
            "I want to make sure I give you the most helpful response. Could you tell me more?",
            "I'm here to help! Could you please rephrase your question or provide more details?",
            "Let me assist you better. Could you provide more information about what you need?"
        ]
        
        import random
        response_text = random.choice(fallback_responses)
        
        return ChatResponse(
            response=response_text,
            intent='fallback',
            confidence=1.0,
            response_type='fallback',
            entities=[],
            sentiment={}
        )
    
    def _get_error_response(self) -> ChatResponse:
        """Get error response for system failures"""
        return ChatResponse(
            response="I'm experiencing some technical difficulties right now. Please try again in a moment.",
            intent='error',
            confidence=1.0,
            response_type='error',
            entities=[],
            sentiment={},
            success=False
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            info = {
                'ml_enabled': self._available,
                'intents_count': 0,
                'model_loaded': False,
                'conversation_count': 0
            }
            
            # Get intents count
            if os.path.exists(self.intents_file):
                try:
                    with open(self.intents_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        info['intents_count'] = len(data.get('intents', []))
                except Exception:
                    pass
            
            # Get model status
            if self.intent_classifier:
                info['model_loaded'] = True
            
            # Get conversation count
            if self.conversation_logger:
                analytics = self.conversation_logger.get_analytics_report()
                info['conversation_count'] = analytics.get('total_conversations', 0)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'ml_enabled': False, 'intents_count': 0, 'model_loaded': False}
    
    def get_service_status(self) -> Dict[str, bool]:
        """Get status of all service components"""
        return {
            'ML Chatbot Service': self._available,
            'Intent Classifier': self.intent_classifier is not None,
            'NLP Processor': self.nlp_processor is not None,
            'Response Generator': self.response_generator is not None,
            'Conversation Logger': self.conversation_logger is not None,
            'Dynamic Handler': self.dynamic_handler is not None
        }
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Get comprehensive analytics report"""
        try:
            if self.conversation_logger:
                report = self.conversation_logger.get_analytics_report()
                
                # Add summary section for easy access
                report['summary'] = {
                    'total_conversations': report.get('total_conversations', 0),
                    'unique_sessions': report.get('unique_sessions', 0),
                    'average_confidence': report.get('average_confidence', 0.0)
                }
                
                return report
            else:
                return {'error': 'Analytics not available', 'summary': {}}
                
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {'error': str(e), 'summary': {}}
    
    def add_user_feedback(self, user_message: str, bot_response: str, rating: int,
                         correct_intent: str = '', notes: str = '') -> bool:
        """Add user feedback for model improvement"""
        try:
            if self.conversation_logger:
                self.conversation_logger.log_feedback(
                    user_message=user_message,
                    bot_response=bot_response,
                    user_rating=rating,
                    correct_intent=correct_intent,
                    improvement_notes=notes
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if the service is available and functioning"""
        return self._available and self._initialized
    
    def retrain_model(self) -> bool:
        """Retrain the ML model with current data"""
        try:
            if self.intent_classifier and os.path.exists(self.intents_file):
                logger.info("Retraining ML model...")
                success = self.intent_classifier.train_model(
                    self.intents_file,
                    model_type='logistic_regression'
                )
                if success:
                    logger.info("Model retrained successfully")
                    return True
                else:
                    logger.error("Model retraining failed")
                    return False
            return False
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return False
    
    def get_intent_probabilities(self, user_message: str) -> Dict[str, float]:
        """Get probability scores for all intents"""
        try:
            if self.intent_classifier:
                processed_data = self._process_message(user_message)
                return self.intent_classifier.get_intent_probabilities(
                    processed_data['processed_text']
                )
            return {}
        except Exception as e:
            logger.error(f"Error getting intent probabilities: {e}")
            return {}
    
    def clear_conversation_history(self, session_id: str = None):
        """Clear conversation history for session or all sessions"""
        try:
            if self.conversation_logger:
                if hasattr(self.conversation_logger, 'clear_logs'):
                    self.conversation_logger.clear_logs(session_id)
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
from flask import Flask, render_template
import logging
import os
from routes.chatbot_routes import chatbot_bp
from config.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    logger.info("Creating Flask AI Chatbot application")
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure app
    app.config.from_object(config_class)
    
    # Register chatbot blueprint
    app.register_blueprint(chatbot_bp)
    logger.info("Chatbot blueprint registered")
    
    # Add context processors for chatbot
    @app.context_processor
    def inject_config():
        from services.ml_chatbot_service import MLChatbotService
        try:
            chatbot_service = MLChatbotService()
            chatbot_available = chatbot_service.is_available()
        except Exception as e:
            logger.error(f"Error checking chatbot availability: {e}")
            chatbot_available = False
            
        return {
            'current_year': Config.get_current_year(),
            'site_name': 'AI Chatbot Assistant',
            'chatbot_available': chatbot_available
        }
    
    # Add custom error handlers with chatbot theme
    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning(f"404 error: {error}")
        return render_template('error.html', 
                             error_code=404,
                             error_title='Page Not Found',
                             error_message='The page you are looking for does not exist. Our AI assistant is still here to help!'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 error: {error}")
        return render_template('error.html',
                             error_code=500,
                             error_title='Internal Server Error',
                             error_message='Something went wrong on our end. Please try again later or contact our AI assistant.'), 500
    
    # Add before/after request handlers
    @app.before_request
    def before_request():
        logger.debug("Processing chatbot request")
    
    @app.after_request
    def after_request(response):
        logger.debug(f"Chatbot request completed with status: {response.status_code}")
        return response
    
    # Add CLI commands for chatbot testing
    @app.cli.command()
    def test_chatbot():
        """Test ML chatbot functionality"""
        from services.ml_chatbot_service import MLChatbotService
        
        print("Testing ML Chatbot Service...")
        try:
            chatbot_service = MLChatbotService()
            
            # Test basic availability
            if chatbot_service.is_available():
                print("✓ Chatbot service is available")
            else:
                print("✗ Chatbot service is not available")
                return
            
            # Test sample queries
            test_queries = [
                "Hello there!",
                "What services do you offer?",
                "How can I contact you?",
                "What are your prices?",
                "I need technical support"
            ]
            
            print("\nTesting sample queries:")
            for query in test_queries:
                response = chatbot_service.get_response(query)
                print(f"Q: {query}")
                print(f"A: {response['response'][:100]}...")
                print(f"Confidence: {response['confidence']:.2f}")
                print("-" * 50)
                
        except Exception as e:
            print(f"✗ Chatbot test failed: {e}")
    
    @app.cli.command()
    def train_chatbot():
        """Train/retrain the ML chatbot model"""
        from services.intent_classifier import IntentClassifier
        
        print("Training ML Chatbot Model...")
        try:
            classifier = IntentClassifier()
            classifier.train_model(f"{Config.CHATBOT_INTENTS_FILE}")
            print("✓ Chatbot model trained successfully")
        except Exception as e:
            print(f"✗ Chatbot training failed: {e}")
    
    logger.info("AI Chatbot Flask application created successfully")
    return app

def create_development_app():
    logger.info("Creating development AI Chatbot application")
    
    # Development-specific configuration
    class DevelopmentConfig(Config):
        DEBUG = True
        TESTING = False
    
    return create_app(DevelopmentConfig)

def create_production_app():
    logger.info("Creating production AI Chatbot application")
    
    # Production-specific configuration
    class ProductionConfig(Config):
        DEBUG = False
        TESTING = False
        SECRET_KEY = os.environ.get('SECRET_KEY') or 'ai-chatbot-production-key'
    
    return create_app(ProductionConfig)
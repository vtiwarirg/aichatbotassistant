"""
Refactored Flask application with modular architecture
Main application factory and configuration
"""
from flask import Flask, render_template
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from routes import register_routes

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application with modular structure"""
    logger.info("Creating AI Chatbot application with refactored architecture")
    
    app = Flask(__name__)
    
    # Load configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
    app.config['WTF_CSRF_ENABLED'] = True
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Register all routes using the new modular structure
    register_routes(app)
    
    # Global context processor for template variables
    @app.context_processor
    def inject_globals():
        """Inject global variables into templates"""
        try:
            from services.chatbot_facade import ChatbotFacade
            chatbot = ChatbotFacade()
            chatbot_available = chatbot.is_available()
            
            # Get basic model info for templates
            if chatbot_available:
                model_info = chatbot.get_model_info()
                ml_enabled = model_info.get('ml_enabled', False)
                intents_count = model_info.get('intents_count', 0)
            else:
                ml_enabled = False
                intents_count = 0
                
        except Exception as e:
            logger.error(f"Error in context processor: {e}")
            chatbot_available = False
            ml_enabled = False
            intents_count = 0
            
        return {
            'site_name': 'AI Chatbot Assistant',
            'chatbot_available': chatbot_available,
            'ml_enabled': ml_enabled,
            'intents_count': intents_count,
            'current_year': datetime.now().year
        }
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return render_template('error.html', 
                             error_code=404,
                             error_message='Page not found'), 404
    
    @app.errorhandler(500)
    def server_error(error):
        logger.error(f"Server error: {error}")
        return render_template('error.html',
                             error_code=500,
                             error_message='Internal server error'), 500
    
    # CLI commands for development and maintenance
    @app.cli.command()
    def train_chatbot():
        """Train the ML chatbot model"""
        from services.chatbot_facade import ChatbotFacade
        print("🤖 Training chatbot model...")
        try:
            chatbot = ChatbotFacade()
            if chatbot.is_available():
                model_info = chatbot.get_model_info()
                print(f"✓ Chatbot services are running!")
                print(f"✓ Model loaded with {model_info.get('intents_count', 0)} intents")
                print(f"✓ ML enabled: {model_info.get('ml_enabled', False)}")
                
                # Get service status
                status = chatbot.get_service_status()
                print("\\nService Status:")
                for service, active in status.items():
                    status_symbol = "✓" if active else "✗"
                    print(f"  {status_symbol} {service}")
            else:
                print("✗ Chatbot services failed to initialize")
        except Exception as e:
            print(f"✗ Training failed: {e}")
    
    @app.cli.command()
    def status():
        """Check comprehensive application status"""
        from services.chatbot_facade import ChatbotFacade
        try:
            print("🔍 Checking application status...")
            chatbot = ChatbotFacade()
            
            # Service status
            status_info = chatbot.get_service_status()
            print("\\nService Status:")
            for service, status in status_info.items():
                status_symbol = "✓" if status else "✗"
                print(f"  {status_symbol} {service}")
            
            # Model information
            if chatbot.is_available():
                model_info = chatbot.get_model_info()
                print("\\nModel Information:")
                print(f"  📊 Intents loaded: {model_info.get('intents_count', 0)}")
                print(f"  🧠 ML enabled: {model_info.get('ml_enabled', False)}")
                print(f"  💬 Conversation count: {model_info.get('conversation_count', 0)}")
                
                # Analytics summary
                analytics = chatbot.get_analytics_report()
                if 'summary' in analytics:
                    summary = analytics['summary']
                    print("\\nAnalytics Summary:")
                    print(f"  💬 Total conversations: {summary.get('total_conversations', 0)}")
                    print(f"  👥 Unique sessions: {summary.get('unique_sessions', 0)}")
                    print(f"  📈 Average confidence: {summary.get('average_confidence', 0):.2f}")
            
        except Exception as e:
            print(f"✗ Status check failed: {e}")
    
    @app.cli.command()
    def clear_logs():
        """Clear conversation logs"""
        from services.chatbot_facade import ChatbotFacade
        try:
            print("🧹 Clearing conversation logs...")
            chatbot = ChatbotFacade()
            if hasattr(chatbot, 'conversation_logger'):
                chatbot.conversation_logger.clear_logs()
                print("✓ Logs cleared successfully")
            else:
                print("✗ Conversation logger not available")
        except Exception as e:
            print(f"✗ Failed to clear logs: {e}")
    
    @app.cli.command()
    def export_data():
        """Export training data"""
        from services.chatbot_facade import ChatbotFacade
        try:
            print("📤 Exporting training data...")
            chatbot = ChatbotFacade()
            if hasattr(chatbot, 'conversation_logger'):
                filepath = chatbot.conversation_logger.export_training_data()
                if filepath:
                    print(f"✓ Training data exported to: {filepath}")
                else:
                    print("✗ No data available for export")
            else:
                print("✗ Conversation logger not available")
        except Exception as e:
            print(f"✗ Export failed: {e}")
    
    @app.cli.command()
    def test_ml():
        """Test ML functionality"""
        from services.chatbot_facade import ChatbotFacade
        try:
            print("🧪 Testing ML chatbot functionality...")
            chatbot = ChatbotFacade()
            
            # Test basic functionality
            print(f"✓ Service available: {chatbot.is_available()}")
            
            # Get model info
            model_info = chatbot.get_model_info()
            print(f"✓ ML enabled: {model_info.get('ml_enabled', False)}")
            print(f"✓ spaCy available: {model_info.get('spacy_available', False)}")
            print(f"✓ Intents loaded: {model_info.get('intents_count', 0)}")
            
            # Test responses
            test_messages = [
                "Hello there!",
                "What time is it?",
                "What's today's date?",
                "What services do you offer?",
                "How's the weather?",
                "Thank you!"
            ]
            
            print("\\n🔍 Testing responses:")
            for message in test_messages:
                response = chatbot.get_response(message, 'test_session')
                print(f"  📩 '{message}' → {response.get('intent', 'unknown')} ({response.get('confidence', 0):.2f})")
            
            # Test analytics
            analytics = chatbot.get_analytics_report()
            if 'summary' in analytics:
                summary = analytics['summary']
                print(f"\\n📊 Analytics:")
                print(f"  💬 Total conversations: {summary.get('total_conversations', 0)}")
                print(f"  📈 Average confidence: {summary.get('average_confidence', 0):.2f}")
            
        except Exception as e:
            print(f"✗ ML test failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("AI Chatbot application created successfully with modular architecture")
    logger.info("Available routes: /, /chat, /about, /contact, /api/*, /health")
    return app

# Create the app instance
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    logger.info(f"Starting refactored AI Chatbot application")
    logger.info(f"Server: {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
    
    app.run(host=host, port=port, debug=debug)
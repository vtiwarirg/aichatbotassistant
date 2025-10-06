from flask import Blueprint, render_template, request, jsonify
import logging
from services.ml_chatbot_service import MLChatbotService
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
chatbot_bp = Blueprint('chatbot', __name__)

# Initialize ML chatbot service
try:
    chatbot_service = MLChatbotService()
    CHATBOT_AVAILABLE = True
    logger.info("ML Chatbot service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot service: {e}")
    chatbot_service = None
    CHATBOT_AVAILABLE = False

# Routes
@chatbot_bp.route('/')
def home():
    """Home page with chatbot introduction"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {e}")
        return render_template('error.html', 
                             error_code=500,
                             error_title='Server Error',
                             error_message='Unable to load the homepage'), 500

@chatbot_bp.route('/chat')
def chat_interface():
    """Chat interface page"""
    try:
        return render_template('chat.html', chatbot_available=CHATBOT_AVAILABLE)
    except Exception as e:
        logger.error(f"Error rendering chat page: {e}")
        return render_template('error.html',
                             error_code=500,
                             error_title='Chat Unavailable',
                             error_message='Unable to load the chat interface'), 500

@chatbot_bp.route('/about')
def about():
    """About page with chatbot information"""
    try:
        return render_template('about.html')
    except Exception as e:
        logger.error(f"Error rendering about page: {e}")
        return render_template('error.html',
                             error_code=500,
                             error_title='Page Error',
                             error_message='Unable to load the about page'), 500

# API Routes
@chatbot_bp.route('/api/chat', methods=['POST'])
def chat_api():
    """Main chat API endpoint"""
    try:
        if not CHATBOT_AVAILABLE or not chatbot_service:
            return jsonify({
                'success': False,
                'response': 'AI chatbot is currently unavailable. Please try again later.',
                'confidence': 0.0,
                'intent': 'error'
            }), 503

        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'response': 'Please provide a valid message.',
                'confidence': 0.0,
                'intent': 'error'
            }), 400

        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                'success': False,
                'response': 'Please enter a message.',
                'confidence': 0.0,
                'intent': 'error'
            }), 400

        # Get response from ML chatbot
        response_data = chatbot_service.get_response(user_message)
        
        return jsonify({
            'success': True,
            'response': response_data['response'],
            'confidence': response_data['confidence'],
            'intent': response_data.get('intent', 'unknown'),
            'entities': response_data.get('entities', [])
        })

    except Exception as e:
        logger.error(f"Error in chat API: {e}")
        return jsonify({
            'success': False,
            'response': 'An error occurred while processing your message. Please try again.',
            'confidence': 0.0,
            'intent': 'error'
        }), 500

@chatbot_bp.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        status = 'healthy' if CHATBOT_AVAILABLE else 'unhealthy'
        details = {
            'chatbot_available': CHATBOT_AVAILABLE,
            'service_status': 'online' if chatbot_service else 'offline'
        }
        
        if chatbot_service:
            details['model_loaded'] = chatbot_service.is_available()
        
        return jsonify({
            'status': status,
            'details': details
        }), 200 if CHATBOT_AVAILABLE else 503

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'error',
            'details': {'error': str(e)}
        }), 500

@chatbot_bp.route('/health')
def health_page():
    """Health check page"""
    try:
        health_data = health_check()
        return render_template('error.html',
                             error_code='Status',
                             error_title='System Health Check',
                             error_message=f"Chatbot Status: {'Available' if CHATBOT_AVAILABLE else 'Unavailable'}")
    except Exception as e:
        logger.error(f"Error rendering health page: {e}")
        return f"Health check error: {e}", 500

# Error handlers
@chatbot_bp.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('error.html',
                         error_code=404,
                         error_title='Page Not Found',
                         error_message='The page you are looking for does not exist'), 404

@chatbot_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html',
                         error_code=500,
                         error_title='Internal Server Error',
                         error_message='Something went wrong on our end'), 500
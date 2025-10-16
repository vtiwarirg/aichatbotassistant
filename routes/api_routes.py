"""
API routes for chatbot functionality
"""
from flask import Blueprint, request, jsonify
import logging
from datetime import datetime

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

@api_bp.route('/chat', methods=['POST'])
def chat_api():
    """Main chat API endpoint"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'response': 'Invalid request format',
                'confidence': 0.0
            }), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'response': 'Please enter a message.',
                'confidence': 0.0
            }), 400
        
        # Try to use full chatbot service
        try:
            from services.chatbot_facade import ChatbotFacade
            chatbot = ChatbotFacade()
            
            if chatbot.is_available():
                response_data = chatbot.get_response(user_message, session_id='web_session')
                return jsonify(response_data)
            else:
                # Fallback to basic responses if ML service unavailable
                from routes import get_basic_response
                response_data = get_basic_response(user_message)
                response_data['note'] = 'Using fallback mode - ML service unavailable'
                return jsonify(response_data)
        except ImportError as e:
            logger.info(f"ChatbotFacade not available: {e}, using fallback")
            # Fallback to basic responses
            from routes import get_basic_response
            response_data = get_basic_response(user_message)
            response_data['note'] = 'Using fallback mode - import error'
            return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat API: {e}")
        return jsonify({
            'success': False,
            'response': 'Sorry, I encountered an error. Please try again.',
            'confidence': 0.0,
            'intent': 'error'
        }), 500

@api_bp.route('/status', methods=['GET'])
def status_api():
    """API status endpoint"""
    try:
        # Try to get detailed status
        try:
            from services.chatbot_facade import ChatbotFacade
            chatbot = ChatbotFacade()
            
            return jsonify({
                'status': 'operational',
                'chatbot_available': chatbot.is_available(),
                'timestamp': datetime.now().isoformat(),
                'services': chatbot.get_service_status() if chatbot.is_available() else {}
            })
        except ImportError:
            return jsonify({
                'status': 'limited',
                'chatbot_available': False,
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'api': True,
                    'basic_responses': True,
                    'ml_chatbot': False
                },
                'mode': 'fallback'
            })
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@api_bp.route('/test', methods=['GET'])
def test_api():
    """Test endpoint for quick checks"""
    test_messages = [
        "Hello",
        "What time is it?",
        "What services do you offer?",
        "Thank you"
    ]
    
    results = []
    for message in test_messages:
        try:
            from routes import get_basic_response
            response = get_basic_response(message)
            results.append({
                'input': message,
                'output': response.get('response', ''),
                'intent': response.get('intent', 'unknown'),
                'confidence': response.get('confidence', 0)
            })
        except Exception as e:
            results.append({
                'input': message,
                'output': f'Error: {e}',
                'intent': 'error',
                'confidence': 0
            })
    
    return jsonify({
        'test_results': results,
        'timestamp': datetime.now().isoformat()
    })
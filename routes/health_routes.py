"""
Health check routes
"""
from flask import Blueprint, jsonify
from datetime import datetime
import os

health_bp = Blueprint('health', __name__)

@health_bp.route('/health')
def health():
    """Comprehensive health check"""
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'flask': True,
                'filesystem': True,
                'routes': True
            },
            'environment': {
                'flask_env': os.environ.get('FLASK_ENV', 'development'),
                'debug': os.environ.get('DEBUG', 'False'),
                'host': os.environ.get('HOST', '127.0.0.1'),
                'port': os.environ.get('PORT', '5000')
            }
        }
        
        # Check chatbot availability
        try:
            from services.chatbot_facade import ChatbotFacade
            chatbot = ChatbotFacade()
            health_data['services']['chatbot'] = chatbot.is_available()
            
            if chatbot.is_available():
                model_info = chatbot.get_model_info()
                health_data['model_status'] = {
                    'available': True,
                    'intents_loaded': model_info.get('intents_count', 0),
                    'ml_enabled': model_info.get('ml_enabled', False)
                }
        except ImportError:
            health_data['services']['chatbot'] = False
            health_data['model_status'] = {
                'available': False, 
                'reason': 'ChatbotFacade not available',
                'fallback_mode': True
            }
        
        # Check required directories
        required_dirs = ['data', 'models', 'logs', 'templates', 'static']
        missing_dirs = []
        for directory in required_dirs:
            if not os.path.exists(directory):
                health_data['services']['filesystem'] = False
                missing_dirs.append(directory)
        
        if missing_dirs:
            health_data['missing_directories'] = missing_dirs
        
        # Check required files
        required_files = ['data/chatbot_intents.json']
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            health_data['missing_files'] = missing_files
        
        return jsonify(health_data)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@health_bp.route('/ping')
def ping():
    """Simple ping endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })
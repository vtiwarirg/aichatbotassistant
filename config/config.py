import os
from datetime import datetime

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DEBUG = os.environ.get('DEBUG')
    # Chatbot Configuration (Local AI - No External APIs)
    CHATBOT_USE_AI = os.environ.get('CHATBOT_USE_AI', 'False').lower() == 'true'
    CHATBOT_USE_ML = os.environ.get('CHATBOT_USE_ML', 'True').lower() == 'true'
    CHATBOT_AI_PROVIDER = os.environ.get('CHATBOT_AI_PROVIDER', 'local')
    
    # ML Chatbot settings
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    CHATBOT_INTENTS_FILE = os.environ.get('CHATBOT_INTENTS_FILE', 'data/chatbot_intents.json')
    CHATBOT_MODEL_PATH = os.environ.get('CHATBOT_MODEL_PATH', 'models/intent_classifier.pkl')
    CHATBOT_CONFIDENCE_THRESHOLD = float(os.environ.get('CHATBOT_CONFIDENCE_THRESHOLD', '0.3'))
    CHATBOT_ML_MODEL_TYPE = os.environ.get('CHATBOT_ML_MODEL_TYPE', 'logistic_regression')
    
    # Session Management
    CHAT_SESSION_TIMEOUT = int(os.environ.get('CHAT_SESSION_TIMEOUT', '3600'))  # 1 hour
    MAX_CHAT_HISTORY = int(os.environ.get('MAX_CHAT_HISTORY', '10'))
    
    # Server Configuration
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', '5000'))
    
    # Template Configuration
    SITE_NAME = os.environ.get('SITE_NAME', 'AI Chatbot Assistant')
    SITE_DESCRIPTION = os.environ.get('SITE_DESCRIPTION', 'Intelligent AI chatbot powered by machine learning') 
    
    @staticmethod
    def get_current_year():
        return datetime.now().year
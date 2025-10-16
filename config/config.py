import os
from datetime import datetime

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    
    # Chatbot settings
    CHATBOT_USE_ML = True
    CHATBOT_USE_AI = False
    CHATBOT_INTENTS_FILE = 'data/chatbot_intents.json'
    CHATBOT_MODEL_PATH = 'models/intent_classifier.pkl'
    CHATBOT_CONFIDENCE_THRESHOLD = 0.3
    CHATBOT_ML_MODEL_TYPE = 'logistic_regression'
    
    # Session settings
    CHAT_SESSION_TIMEOUT = 3600
    MAX_CHAT_HISTORY = 10
    
    @staticmethod
    def get_current_year():
        return datetime.now().year

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set for production")

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    CHATBOT_CONFIDENCE_THRESHOLD = 0.1  # Lower threshold for testing
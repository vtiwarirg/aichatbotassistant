"""
Routes module initialization with error handling
"""
from flask import Blueprint, render_template, jsonify, request
from datetime import datetime
import logging
import random

logger = logging.getLogger(__name__)

def register_routes(app):
    """Register all route blueprints with fallback handling"""
    try:
        # Try to import and register all route modules
        from .main_routes import main_bp
        from .health_routes import health_bp
        
        app.register_blueprint(main_bp)
        app.register_blueprint(health_bp)
        
        # Register API routes from existing file
        from .api_routes import api_bp
        app.register_blueprint(api_bp, url_prefix='/api')
        
        logger.info("All route blueprints registered successfully")
        return True
        
    except ImportError as e:
        logger.warning(f"Route import failed: {e}. Registering fallback routes.")
        register_fallback_routes(app)
        return False

def register_fallback_routes(app):
    """Register minimal fallback routes when imports fail"""
    
    @app.route('/')
    def home():
        return render_template('index.html', 
                             site_name='AI Chatbot Assistant',
                             page_title='Home')
    
    @app.route('/chat')
    def chat():
        return render_template('chat.html', 
                             site_name='AI Chatbot Assistant',
                             page_title='Chat')
    
    @app.route('/about')
    def about():
        return render_template('about.html', 
                             site_name='AI Chatbot Assistant',
                             page_title='About')
    
    @app.route('/contact')
    def contact():
        return render_template('contact.html', 
                             site_name='AI Chatbot Assistant',
                             page_title='Contact')
    
    @app.route('/api/chat', methods=['POST'])
    def chat_api():
        try:
            data = request.get_json()
            user_message = data.get('message', '').strip() if data else ''
            
            if not user_message:
                return jsonify({
                    'success': False,
                    'response': 'Please enter a message.',
                    'confidence': 0.0
                }), 400
            
            # Basic response logic
            response = get_basic_response(user_message)
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return jsonify({
                'success': False,
                'response': 'Sorry, I encountered an error. Please try again.',
                'confidence': 0.0
            }), 500
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'mode': 'fallback',
            'services': {
                'flask': True,
                'routes': True,
                'chatbot': False
            }
        })
    
    logger.info("Fallback routes registered successfully")

def get_basic_response(user_message):
    """Get basic response without ML services"""
    
    message_lower = user_message.lower()
    
    # Time queries
    if any(word in message_lower for word in ['time', 'clock']):
        current_time = datetime.now().strftime("%I:%M %p")
        return {
            'response': f'The current time is {current_time}',
            'intent': 'time',
            'confidence': 0.9,
            'success': True,
            'response_type': 'dynamic'
        }
    
    # Date queries
    elif any(word in message_lower for word in ['date', 'today']):
        current_date = datetime.now().strftime("%B %d, %Y")
        return {
            'response': f'Today\'s date is {current_date}',
            'intent': 'date',
            'confidence': 0.9,
            'success': True,
            'response_type': 'dynamic'
        }
    
    # Greetings
    elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        responses = [
            'Hello! Welcome to our AI assistant. I can help with customer support, HR questions, IT issues, and general inquiries.',
            'Hi there! I\'m here to assist you with various business needs. How can I help you today?',
            'Welcome! I\'m ready to help with customer support, technical issues, and general questions.'
        ]
        return {
            'response': random.choice(responses),
            'intent': 'greeting',
            'confidence': 0.8,
            'success': True,
            'response_type': 'intent'
        }
    
    # Services
    elif any(word in message_lower for word in ['services', 'help', 'what can you do', 'capabilities']):
        return {
            'response': 'I can help with: Customer Service inquiries, HR policy questions, IT support, account management, billing assistance, and general business information. What specific area do you need help with?',
            'intent': 'services',
            'confidence': 0.8,
            'success': True,
            'response_type': 'intent'
        }
    
    # Contact
    elif any(word in message_lower for word in ['contact', 'phone', 'email', 'reach']):
        return {
            'response': 'You can reach our support team via Email: support@company.com, Phone: 1-800-SUPPORT, or through our online contact form. For urgent IT issues, use our priority support line.',
            'intent': 'contact',
            'confidence': 0.8,
            'success': True,
            'response_type': 'intent'
        }
    
    # Business hours
    elif any(word in message_lower for word in ['hours', 'open', 'available', 'when']):
        return {
            'response': 'Our support hours: Customer Service (24/7), IT Helpdesk (Mon-Fri 6AM-10PM EST), HR Department (Mon-Fri 8AM-6PM EST), Billing Support (Mon-Fri 7AM-7PM EST).',
            'intent': 'business_hours',
            'confidence': 0.8,
            'success': True,
            'response_type': 'intent'
        }
    
    # Goodbye
    elif any(word in message_lower for word in ['bye', 'goodbye', 'thanks', 'thank you']):
        responses = [
            'Thank you for using our AI assistant! Feel free to reach out anytime.',
            'Goodbye! Remember, I\'m available 24/7 for any business support needs.',
            'Thanks for chatting! Don\'t hesitate to return if you need assistance.'
        ]
        return {
            'response': random.choice(responses),
            'intent': 'goodbye',
            'confidence': 0.8,
            'success': True,
            'response_type': 'intent'
        }
    
    # Fallback
    else:
        responses = [
            'I\'d be happy to help you with that! Could you please provide more specific details?',
            'I want to make sure I give you the most helpful response. Could you tell me more?',
            'I\'m here to help! Could you please rephrase your question or provide more details?'
        ]
        return {
            'response': random.choice(responses),
            'intent': 'fallback',
            'confidence': 0.5,
            'success': True,
            'response_type': 'fallback'
        }
"""
Main application routes for pages
"""
from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    """Homepage with chatbot introduction"""
    return render_template('index.html', 
                         page_title='Home',
                         description='AI-powered chatbot assistant for business support')

@main_bp.route('/chat')
def chat():
    """Main chat interface"""
    return render_template('chat.html', 
                         page_title='Chat',
                         description='Interactive chat interface')

@main_bp.route('/about')
def about():
    """About page"""
    return render_template('about.html', 
                         page_title='About',
                         description='About our AI chatbot assistant')

@main_bp.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html', 
                         page_title='Contact',
                         description='Contact information and support')
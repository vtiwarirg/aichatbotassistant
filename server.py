#!/usr/bin/env python3
import os
import sys
import logging
from app import create_app, create_development_app, create_production_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run the Flask application"""
    
    # Determine environment
    environment = os.environ.get('FLASK_ENV', 'development').lower()
    
    logger.info(f"Starting Python Web App in {environment} mode")
    
    # Create appropriate app based on environment
    if environment == 'production':
        app = create_production_app()
        debug = False
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
    else:
        app = create_development_app()
        debug = True
        port = 5000
        host = '127.0.0.1'
    
    # Print startup information
    logger.info(f"Server starting on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    try:
        # Run the application
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
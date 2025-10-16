"""
Handles dynamic responses for real-time data like time, date, weather, etc.
"""
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DynamicResponseHandler:
    """Handles dynamic content in chatbot responses"""
    
    def __init__(self):
        self.dynamic_handlers = {
            'DYNAMIC_TIME': self._get_current_time,
            'DYNAMIC_DATE': self._get_current_date,
            'DYNAMIC_DATETIME': self._get_current_datetime,
            'DYNAMIC_DAY': self._get_current_day,
            'DYNAMIC_YEAR': self._get_current_year,
            'DYNAMIC_GREETING': self._get_greeting_by_time
        }
        logger.info("Dynamic Response Handler initialized")
    
    def process_response(self, response: str, user_message: str = None) -> str:
        """Process response and replace dynamic placeholders"""
        try:
            processed_response = response
            
            # Replace all dynamic placeholders
            for placeholder, handler in self.dynamic_handlers.items():
                pattern = f'{{{{{placeholder}}}}}'
                if pattern in processed_response:
                    dynamic_value = handler(user_message)
                    processed_response = processed_response.replace(pattern, dynamic_value)
                    logger.debug(f"Replaced {pattern} with {dynamic_value}")
            
            return processed_response
            
        except Exception as e:
            logger.error(f"Error processing dynamic response: {e}")
            return response  # Return original response if processing fails
    
    def _get_current_time(self, user_message: str = None) -> str:
        """Get current time formatted nicely"""
        now = datetime.now()
        return now.strftime("%I:%M %p")  # e.g., "02:30 PM"
    
    def _get_current_date(self, user_message: str = None) -> str:
        """Get current date formatted nicely"""
        now = datetime.now()
        return now.strftime("%B %d, %Y")  # e.g., "October 17, 2025"
    
    def _get_current_datetime(self, user_message: str = None) -> str:
        """Get current date and time"""
        now = datetime.now()
        return now.strftime("%B %d, %Y at %I:%M %p")  # e.g., "October 17, 2025 at 02:30 PM"
    
    def _get_current_day(self, user_message: str = None) -> str:
        """Get current day of week"""
        now = datetime.now()
        return now.strftime("%A")  # e.g., "Thursday"
    
    def _get_current_year(self, user_message: str = None) -> str:
        """Get current year"""
        now = datetime.now()
        return str(now.year)
    
    def _get_greeting_by_time(self, user_message: str = None) -> str:
        """Get time-appropriate greeting"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 21:
            return "Good evening"
        else:
            return "Good night"
    
    def has_dynamic_content(self, response: str) -> bool:
        """Check if response contains dynamic placeholders"""
        for placeholder in self.dynamic_handlers.keys():
            if f'{{{{{placeholder}}}}}' in response:
                return True
        return False
    
    def get_available_placeholders(self) -> Dict[str, str]:
        """Get list of available dynamic placeholders with descriptions"""
        return {
            'DYNAMIC_TIME': 'Current time (e.g., 02:30 PM)',
            'DYNAMIC_DATE': 'Current date (e.g., October 17, 2025)',
            'DYNAMIC_DATETIME': 'Current date and time combined',
            'DYNAMIC_DAY': 'Current day of week (e.g., Thursday)',
            'DYNAMIC_YEAR': 'Current year (e.g., 2025)',
            'DYNAMIC_GREETING': 'Time-appropriate greeting (Good morning/afternoon/evening/night)'
        }
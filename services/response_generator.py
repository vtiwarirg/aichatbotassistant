"""
Handles response generation logic and enhancement
"""
import json
import random
import logging
from typing import Dict, List, Any, Optional
from .dynamic_response_handler import DynamicResponseHandler

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles response generation from intents and enhancement logic"""
    
    def __init__(self, intents_file: str):
        self.intents_file = intents_file
        self.responses_by_intent = self._load_responses()
        self.fallback_responses = self._get_fallback_responses()
        self.dynamic_handler = DynamicResponseHandler()
    
    def _load_responses(self) -> Dict[str, List[str]]:
        """Load responses from intents JSON file"""
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            responses = {}
            for intent in data.get('intents', []):
                tag = intent.get('tag')
                intent_responses = intent.get('responses', [])
                if tag and intent_responses:
                    responses[tag] = intent_responses
            
            logger.info(f"Loaded responses for {len(responses)} intents")
            return responses
            
        except Exception as e:
            logger.error(f"Error loading responses from {self.intents_file}: {e}")
            return {}
    
    def get_response_for_intent(self, intent: str, entities: List = None, 
                               sentiment: Dict = None, user_message: str = '') -> Optional[str]:
        """Get response for given intent with enhancements"""
        responses = self.responses_by_intent.get(intent, [])
        if responses:
            base_response = random.choice(responses)
            
            # Process dynamic content first
            if self.dynamic_handler.has_dynamic_content(base_response):
                base_response = self.dynamic_handler.process_response(base_response, user_message)
            
            # Then apply other enhancements
            return self._enhance_response(base_response, entities, sentiment, user_message)
        return None
    
    def _enhance_response(self, response: str, entities: List = None, 
                         sentiment: Dict = None, user_message: str = '') -> str:
        """Enhance response with entities, sentiment, and context"""
        try:
            enhanced_response = response
            
            # Add personalization based on entities
            if entities:
                person_entities = [ent[0] for ent in entities if len(ent) > 1 and ent[1] == 'PERSON']
                org_entities = [ent[0] for ent in entities if len(ent) > 1 and ent[1] == 'ORG']
                
                if person_entities:
                    enhanced_response += f"\n\nI noticed you mentioned {', '.join(person_entities[:2])}."
                elif org_entities:
                    enhanced_response += f"\n\nI see you're asking about {', '.join(org_entities[:2])}."
            
            # Add sentiment-based enhancement
            if sentiment:
                sentiment_score = sentiment.get('sentiment_score', 0)
                
                if sentiment_score < -1:
                    enhanced_response += "\n\nI understand this might be frustrating. Let me help you resolve this quickly."
                elif sentiment_score > 1:
                    enhanced_response += "\n\nI'm glad to help with your inquiry!"
                
                # Add urgency indicators for negative sentiment
                if sentiment_score < -2:
                    enhanced_response += " If this is urgent, please don't hesitate to contact our priority support."
            
            # Add context-based enhancements
            enhanced_response = self._add_contextual_enhancements(enhanced_response, user_message)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return response
    
    def _add_contextual_enhancements(self, response: str, user_message: str) -> str:
        """Add contextual information based on user message patterns"""
        try:
            user_lower = user_message.lower()
            
            # Check for time-sensitive keywords
            if any(word in user_lower for word in ['urgent', 'asap', 'immediately', 'emergency']):
                if 'urgent' not in response.lower():
                    response += "\n\nI understand this is urgent. Let me prioritize your request."
            
            # Check for specific technical terms
            if any(word in user_lower for word in ['error', 'bug', 'crash', 'broken', 'not working']):
                if 'technical' not in response.lower():
                    response += "\n\nFor technical issues, you can also check our status page or submit a detailed support ticket."
            
            # Check for business hours inquiries
            if any(word in user_lower for word in ['hours', 'open', 'available', 'schedule']):
                response += "\n\nYou can also find our current operating hours on our website."
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding contextual enhancements: {e}")
            return response
    
    def get_fallback_response(self, sentiment: Dict = None, user_message: str = '') -> str:
        """Get fallback response based on sentiment and context"""
        try:
            base_responses = self.fallback_responses
            
            if sentiment:
                sentiment_score = sentiment.get('sentiment_score', 0)
                
                if sentiment_score < -1:
                    # Negative sentiment responses
                    base_responses = [
                        "I understand you might be experiencing some frustration. Let me connect you with the right support to help resolve this quickly.",
                        "I can see this is important to you. Please let me help you find the best solution for your concern.",
                        "I want to make sure we address your concern properly. Could you provide more details so I can assist you better?"
                    ]
                elif sentiment_score > 1:
                    # Positive sentiment responses
                    base_responses = [
                        "Thank you for your positive energy! I'm here to help you with whatever you need.",
                        "It's great to chat with you! How can I assist you today?",
                        "I appreciate your enthusiasm! Let me know how I can help you."
                    ]
            
            # Select response and enhance it
            base_response = random.choice(base_responses)
            
            # Process dynamic content if present
            if self.dynamic_handler.has_dynamic_content(base_response):
                base_response = self.dynamic_handler.process_response(base_response, user_message)
            
            # Add suggestions based on common patterns
            if user_message:
                suggestions = self._get_response_suggestions(user_message)
                if suggestions:
                    base_response += f"\n\nYou might also try: {', '.join(suggestions)}"
            
            return base_response
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return "I'm here to help! Could you please rephrase your question or provide more details?"
    
    def _get_fallback_responses(self) -> List[str]:
        """Get default fallback responses"""
        return [
            "I'd be happy to help you with that! Could you please provide more specific details about what you're looking for?",
            "I want to make sure I give you the most helpful response. Could you tell me more about what you need assistance with?",
            "I'm here to help! Please let me know what specific information or support you're looking for.",
            "Could you please rephrase your question or provide more context? I want to give you the best possible assistance.",
            "I'm not quite sure I understand. Could you tell me more about what you're trying to accomplish?"
        ]
    
    def _get_response_suggestions(self, user_message: str) -> List[str]:
        """Get suggested follow-up actions based on user message"""
        suggestions = []
        user_lower = user_message.lower()
        
        try:
            # Suggest based on keywords
            if any(word in user_lower for word in ['help', 'support']):
                suggestions.extend(['Browse our FAQ', 'Contact live support', 'Submit a ticket'])
            
            if any(word in user_lower for word in ['price', 'cost', 'pricing']):
                suggestions.extend(['View pricing page', 'Get a quote', 'Contact sales'])
            
            if any(word in user_lower for word in ['account', 'login', 'password']):
                suggestions.extend(['Reset password', 'Account recovery', 'Contact account support'])
            
            if any(word in user_lower for word in ['technical', 'error', 'bug']):
                suggestions.extend(['Check system status', 'Submit bug report', 'Contact technical support'])
            
            # Limit suggestions
            return suggestions[:3]
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []
    
    def get_intent_suggestions(self, intent: str) -> List[str]:
        """Get suggested follow-up actions based on predicted intent"""
        intent_suggestions = {
            'greeting': ['Ask about our services', 'Get help with account', 'Contact support'],
            'services': ['Learn about pricing', 'Contact sales', 'View documentation'],
            'technical_support': ['Submit a ticket', 'View status page', 'Contact IT'],
            'billing': ['View account', 'Update payment', 'Contact billing'],
            'contact': ['Send email', 'Call support', 'Submit form'],
            'account': ['Reset password', 'Update profile', 'Security settings'],
            'hr_policies': ['View handbook', 'Contact HR', 'Submit request'],
            'help': ['Browse FAQ', 'Contact support', 'Live chat'],
            'goodbye': ['Rate conversation', 'Subscribe to updates', 'Visit website'],
            'complaints': ['Submit formal complaint', 'Speak with manager', 'Escalate issue']
        }
        
        return intent_suggestions.get(intent, ['Contact support', 'Browse FAQ', 'Try again'])
    
    def reload_responses(self):
        """Reload responses from file (useful for dynamic updates)"""
        try:
            logger.info("Reloading responses from intents file...")
            self.responses_by_intent = self._load_responses()
            logger.info("Responses reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading responses: {e}")
    
    def get_response_stats(self) -> Dict[str, Any]:
        """Get statistics about available responses"""
        total_responses = sum(len(responses) for responses in self.responses_by_intent.values())
        
        return {
            'total_intents': len(self.responses_by_intent),
            'total_responses': total_responses,
            'average_responses_per_intent': total_responses / len(self.responses_by_intent) if self.responses_by_intent else 0,
            'intents_with_multiple_responses': sum(1 for responses in self.responses_by_intent.values() if len(responses) > 1),
            'fallback_responses_available': len(self.fallback_responses)
        }
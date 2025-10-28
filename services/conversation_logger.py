"""
Handles conversation logging and analytics using CSV storage
"""
import csv
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ConversationLogger:
    """Handles CSV-based conversation logging and analytics"""
    
    def __init__(self):
        self.setup_csv_storage()
    
    def setup_csv_storage(self):
        """Setup CSV files for data storage and analytics"""
        try:
            # Create analytics directory
            os.makedirs('data/analytics', exist_ok=True)
            
            # Conversation logs CSV
            self.conversation_csv = 'data/analytics/conversations.csv'
            if not os.path.exists(self.conversation_csv):
                headers = [
                    'timestamp', 'session_id', 'user_message', 'bot_response', 
                    'intent', 'confidence', 'response_type', 'entities'
                ]
                self._create_csv_file(self.conversation_csv, headers)
            
            # Feedback logs CSV
            self.feedback_csv = 'data/analytics/feedback.csv'
            if not os.path.exists(self.feedback_csv):
                headers = [
                    'timestamp', 'user_message', 'bot_response', 'user_rating',
                    'correct_intent', 'improvement_notes'
                ]
                self._create_csv_file(self.feedback_csv, headers)
            
            # Analytics summary CSV
            self.analytics_csv = 'data/analytics/intent_analytics.csv'
            if not os.path.exists(self.analytics_csv):
                headers = [
                    'date', 'intent', 'total_requests', 'avg_confidence',
                    'successful_responses', 'feedback_count'
                ]
                self._create_csv_file(self.analytics_csv, headers)
            
            logger.info("CSV storage setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up CSV storage: {e}")
            raise
    
    def _create_csv_file(self, filepath: str, headers: List[str]):
        """Create CSV file with headers"""
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    def log_conversation(self, user_message: str, bot_response: str, intent: str, 
                        confidence: float, response_type: str, entities: List, 
                        session_id: str = 'default'):
        """Log conversation interaction to CSV"""
        try:
            with open(self.conversation_csv, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().isoformat(),
                    session_id,
                    user_message,
                    bot_response,
                    intent,
                    confidence,
                    response_type,
                    json.dumps(entities) if entities else '[]'
                ])
            logger.debug(f"Logged conversation: {intent} with confidence {confidence}")
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
    
    def log_feedback(self, user_message: str, bot_response: str, user_rating: int,
                    correct_intent: str, improvement_notes: str = ''):
        """Log user feedback to CSV"""
        try:
            with open(self.feedback_csv, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().isoformat(),
                    user_message,
                    bot_response,
                    user_rating,
                    correct_intent,
                    improvement_notes
                ])
            logger.info(f"Logged feedback: rating={user_rating}, intent={correct_intent}")
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report from CSV data"""
        try:
            # Read CSV files
            conversations_df = self._safe_read_csv(self.conversation_csv)
            feedback_df = self._safe_read_csv(self.feedback_csv)
            
            # Basic statistics
            total_conversations = len(conversations_df) if not conversations_df.empty else 0
            unique_sessions = conversations_df['session_id'].nunique() if 'session_id' in conversations_df.columns else 0
            
            # Intent analysis
            intent_distribution = {}
            avg_confidence = 0.0
            if not conversations_df.empty and 'intent' in conversations_df.columns:
                intent_distribution = conversations_df['intent'].value_counts().to_dict()
                if 'confidence' in conversations_df.columns:
                    avg_confidence = float(conversations_df['confidence'].mean())
            
            # Response type analysis
            response_type_distribution = {}
            if not conversations_df.empty and 'response_type' in conversations_df.columns:
                response_type_distribution = conversations_df['response_type'].value_counts().to_dict()
            
            # Feedback analysis
            feedback_stats = {}
            if not feedback_df.empty:
                if 'user_rating' in feedback_df.columns:
                    feedback_stats = {
                        'total_feedback': len(feedback_df),
                        'average_rating': float(feedback_df['user_rating'].mean()),
                        'rating_distribution': feedback_df['user_rating'].value_counts().to_dict()
                    }
            
            # Time-based analysis
            daily_conversations = {}
            if not conversations_df.empty and 'timestamp' in conversations_df.columns:
                try:
                    conversations_df['timestamp'] = pd.to_datetime(conversations_df['timestamp'])
                    conversations_df['date'] = conversations_df['timestamp'].dt.date
                    daily_counts = conversations_df['date'].value_counts().to_dict()
                    # Convert date keys to strings for JSON serialization
                    daily_conversations = {str(k): v for k, v in daily_counts.items()}
                except Exception as e:
                    logger.warning(f"Error processing time-based analysis: {e}")
            
            report = {
                'report_generated': datetime.now().isoformat(),
                'summary': {
                    'total_conversations': total_conversations,
                    'unique_sessions': unique_sessions,
                    'average_confidence': avg_confidence
                },
                'intent_analysis': {
                    'distribution': intent_distribution,
                    'total_intents': len(intent_distribution)
                },
                'response_analysis': {
                    'type_distribution': response_type_distribution
                },
                'feedback_analysis': feedback_stats,
                'temporal_analysis': {
                    'daily_conversation_counts': daily_conversations
                },
                'data_quality': {
                    'conversations_logged': total_conversations,
                    'feedback_entries': len(feedback_df) if not feedback_df.empty else 0,
                    'data_files_status': self._check_data_files()
                }
            }
            
            logger.info(f"Generated analytics report: {total_conversations} conversations analyzed")
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            return {
                'error': f"Analytics generation failed: {str(e)}",
                'report_generated': datetime.now().isoformat()
            }
    
    def _safe_read_csv(self, filepath: str) -> pd.DataFrame:
        """Safely read CSV file, return empty DataFrame if file doesn't exist or is empty"""
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                return pd.read_csv(filepath)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Could not read {filepath}: {e}")
            return pd.DataFrame()
    
    def _check_data_files(self) -> Dict[str, bool]:
        """Check status of data files"""
        return {
            'conversations_csv': os.path.exists(self.conversation_csv),
            'feedback_csv': os.path.exists(self.feedback_csv),
            'analytics_csv': os.path.exists(self.analytics_csv)
        }
    
    def export_training_data(self, filename: str = None) -> str:
        """Export conversation data as training data CSV"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"training_data_{timestamp}.csv"
            
            conversations_df = self._safe_read_csv(self.conversation_csv)
            
            if conversations_df.empty:
                logger.warning("No conversation data available for export")
                return ""
            
            # Prepare training data
            training_data = []
            for _, row in conversations_df.iterrows():
                if row.get('intent') and row.get('intent') != 'fallback':
                    training_data.append({
                        'text': row['user_message'],
                        'intent': row['intent'],
                        'confidence': row.get('confidence', 0),
                        'response_type': row.get('response_type', ''),
                        'timestamp': row.get('timestamp', '')
                    })
            
            if not training_data:
                logger.warning("No suitable training data found")
                return ""
            
            # Create DataFrame and save
            training_df = pd.DataFrame(training_data)
            filepath = f"data/analytics/{filename}"
            training_df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"Training data exported: {len(training_data)} samples to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return ""
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get quick conversation statistics"""
        try:
            conversations_df = self._safe_read_csv(self.conversation_csv)
            
            if conversations_df.empty:
                return {'total': 0, 'sessions': 0, 'avg_confidence': 0}
            
            stats = {
                'total_conversations': len(conversations_df),
                'unique_sessions': conversations_df['session_id'].nunique() if 'session_id' in conversations_df.columns else 0,
                'average_confidence': float(conversations_df['confidence'].mean()) if 'confidence' in conversations_df.columns else 0,
                'most_common_intent': conversations_df['intent'].mode()[0] if 'intent' in conversations_df.columns and len(conversations_df) > 0 else 'unknown'
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return {'error': str(e)}
    
    def clear_logs(self, older_than_days: int = None):
        """Clear conversation logs (optionally older than specified days)"""
        try:
            if older_than_days:
                # Implementation for date-based clearing would go here
                logger.info(f"Clearing logs older than {older_than_days} days (not implemented)")
            else:
                # Clear all logs by recreating files
                self.setup_csv_storage()
                logger.info("All conversation logs cleared")
        except Exception as e:
            logger.error(f"Error clearing logs: {e}")
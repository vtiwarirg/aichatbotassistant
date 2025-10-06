import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Machine Learning based intent classifier using scikit-learn"""
    
    def __init__(self, model_path='models/intent_classifier.pkl'):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.is_trained = False
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        logger.info("Intent Classifier initialized")
    
    def load_training_data(self, json_path):
        """Load and prepare training data from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Prepare training data
            texts = []
            labels = []
            
            for intent in data['intents']:
                tag = intent['tag']
                patterns = intent['patterns']
                
                for pattern in patterns:
                    texts.append(pattern)
                    labels.append(tag)
            
            # Create label encoding
            unique_labels = list(set(labels))
            self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
            
            # Convert labels to numbers
            encoded_labels = [self.label_encoder[label] for label in labels]
            
            logger.info(f"Loaded {len(texts)} training samples with {len(unique_labels)} classes")
            return texts, encoded_labels, unique_labels
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None, None, None
    
    def train_model(self, json_path, model_type='logistic_regression'):
        """Train the intent classification model"""
        # Load training data
        texts, labels, unique_labels = self.load_training_data(json_path)
        
        if texts is None:
            logger.error("Failed to load training data")
            return False
        
        try:
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True,
                min_df=1,
                max_df=0.95
            )
            
            # Choose model based on model_type
            if model_type == 'naive_bayes':
                classifier = MultinomialNB(alpha=0.1)
            elif model_type == 'random_forest':
                classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # default to logistic regression
                classifier = LogisticRegression(random_state=42, max_iter=1000)
            
            # Create pipeline
            self.model = Pipeline([
                ('tfidf', self.vectorizer),
                ('classifier', classifier)
            ])
            
            # Train the model
            self.model.fit(texts, labels)
            
            # Evaluate on training data (for basic validation)
            train_predictions = self.model.predict(texts)
            accuracy = accuracy_score(labels, train_predictions)
            
            logger.info(f"Model trained successfully with accuracy: {accuracy:.3f}")
            logger.info(f"Model type: {model_type}")
            logger.info(f"Classes: {unique_labels}")
            
            self.is_trained = True
            
            # Save the model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def save_model(self):
        """Save the trained model and encoders"""
        try:
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'reverse_label_encoder': self.reverse_label_encoder,
                'is_trained': self.is_trained
            }
            
            with open(self.model_path, 'wb') as file:
                pickle.dump(model_data, file)
            
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load a pre-trained model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as file:
                    model_data = pickle.load(file)
                
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']
                self.reverse_label_encoder = model_data['reverse_label_encoder']
                self.is_trained = model_data['is_trained']
                
                logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_intent(self, text, confidence_threshold=0.3):
        """Predict intent for given text"""
        if not self.is_trained:
            logger.warning("Model not trained. Attempting to load saved model...")
            if not self.load_model():
                return None, 0.0
        
        try:
            # Get prediction probabilities
            probabilities = self.model.predict_proba([text])[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                logger.info(f"Low confidence prediction: {confidence:.3f} < {confidence_threshold}")
                return None, confidence
            
            # Get predicted intent
            predicted_intent = self.reverse_label_encoder[predicted_class_idx]
            
            logger.info(f"Predicted intent: {predicted_intent} with confidence: {confidence:.3f}")
            return predicted_intent, confidence
            
        except Exception as e:
            logger.error(f"Error predicting intent: {e}")
            return None, 0.0
    
    def get_all_intents_probabilities(self, text):
        """Get probabilities for all intents"""
        if not self.is_trained:
            if not self.load_model():
                return {}
        
        try:
            probabilities = self.model.predict_proba([text])[0]
            
            # Create intent-probability mapping
            intent_probs = {}
            for idx, prob in enumerate(probabilities):
                intent = self.reverse_label_encoder[idx]
                intent_probs[intent] = prob
            
            # Sort by probability
            sorted_intents = sorted(intent_probs.items(), key=lambda x: x[1], reverse=True)
            
            return dict(sorted_intents)
            
        except Exception as e:
            logger.error(f"Error getting intent probabilities: {e}")
            return {}
    
    def retrain_with_feedback(self, text, correct_intent):
        """Add new training example and retrain (simple online learning simulation)"""
        try:
            # This is a simplified approach - in production, you'd want more sophisticated online learning
            logger.info(f"Adding feedback: '{text}' -> '{correct_intent}'")
            
            # For now, we'll just log the feedback
            # In a full implementation, you'd store this and periodically retrain
            feedback_data = {
                'text': text,
                'intent': correct_intent,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Save feedback to a file for future retraining
            feedback_file = 'data/feedback_data.jsonl'
            os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
            
            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_data) + '\n')
            
            logger.info("Feedback saved for future retraining")
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def evaluate_model(self, test_texts, test_labels):
        """Evaluate model performance"""
        if not self.is_trained:
            return None
        
        try:
            predictions = self.model.predict(test_texts)
            accuracy = accuracy_score(test_labels, predictions)
            
            # Get class names for the report
            target_names = [self.reverse_label_encoder[i] for i in sorted(self.reverse_label_encoder.keys())]
            report = classification_report(test_labels, predictions, target_names=target_names)
            
            logger.info(f"Model evaluation - Accuracy: {accuracy:.3f}")
            logger.info(f"Classification report:\n{report}")
            
            return {
                'accuracy': accuracy,
                'classification_report': report
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
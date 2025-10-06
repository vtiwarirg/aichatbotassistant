import nltk
import spacy
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

logger = logging.getLogger(__name__)

class NLPProcessor:
    """Advanced NLP processor using NLTK and spaCy for text preprocessing and feature extraction"""
    
    def __init__(self):
        try:
            # Load spaCy model
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found. Using NLTK only.")
            self.nlp = None
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not found. Using default set.")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # TF-IDF vectorizer for feature extraction
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            lowercase=True,
            tokenizer=self.custom_tokenizer
        )
        
        logger.info("NLP Processor initialized")
    
    def custom_tokenizer(self, text):
        """Custom tokenizer that uses spaCy if available, otherwise NLTK"""
        if self.nlp:
            # Use spaCy for tokenization and lemmatization
            doc = self.nlp(text)
            tokens = [token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct and token.is_alpha]
        else:
            # Use NLTK for tokenization and lemmatization
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and token not in string.punctuation and token.isalpha()]
        
        return tokens
    
    def preprocess_text(self, text):
        """Comprehensive text preprocessing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\?\!\.]', '', text)
        
        if self.nlp:
            # Use spaCy for advanced preprocessing
            doc = self.nlp(text)
            # Extract lemmatized tokens, excluding stop words and punctuation
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and token.is_alpha]
        else:
            # Use NLTK for preprocessing
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and token not in string.punctuation and token.isalpha()]
        
        return ' '.join(tokens)
    
    def extract_features(self, texts):
        """Extract TF-IDF features from text data"""
        try:
            # Preprocess all texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Fit and transform texts to TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            logger.info(f"Extracted features with shape: {tfidf_matrix.shape}")
            return tfidf_matrix
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def transform_text(self, text):
        """Transform a single text using fitted vectorizer"""
        try:
            processed_text = self.preprocess_text(text)
            return self.vectorizer.transform([processed_text])
        except Exception as e:
            logger.error(f"Error transforming text: {e}")
            return None
    
    def get_text_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        try:
            processed_texts = [self.preprocess_text(text1), self.preprocess_text(text2)]
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            if tfidf_matrix.shape[0] >= 2:
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return similarity
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def get_pos_tags(self, text):
        """Get part-of-speech tags"""
        if self.nlp:
            # Use spaCy
            doc = self.nlp(text)
            return [(token.text, token.pos_) for token in doc]
        else:
            # Use NLTK
            try:
                import nltk
                tokens = word_tokenize(text)
                return nltk.pos_tag(tokens)
            except Exception as e:
                logger.error(f"Error getting POS tags: {e}")
                return []
    
    def get_sentiment_features(self, text):
        """Extract basic sentiment features"""
        # Simple sentiment analysis based on word patterns
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'best'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'dislike', 'worst', 'horrible', 'poor'}
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'sentiment_score': positive_count - negative_count
        }
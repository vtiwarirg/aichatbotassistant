# AI Chatbot Assistant - Privacy-Focused ML Chatbot

A Flask-based AI chatbot application powered by machine learning and natural language processing. Built with NLTK, spaCy, and scikit-learn for completely local, privacy-focused intelligent conversations without external API dependencies.

## 🤖 Key Features

- **Complete Privacy**: All processing happens locally, no external API calls
- **Machine Learning Powered**: Uses scikit-learn for intent classification
- **Advanced NLP**: NLTK and spaCy for text processing and entity recognition
- **High Accuracy**: TF-IDF vectorization with Logistic Regression classifier
- **Fast Response**: Optimized local processing for instant replies
- **Open Source**: Built entirely with transparent, open-source technologies

## 📁 Project Structure

```
pythonwebapp/
│
├── 📁 config/
│   ├── __init__.py
│   └── config.py              # ML chatbot configuration
│
├── 📁 services/
│   ├── __init__.py
│   ├── ml_chatbot_service.py  # Main chatbot service
│   ├── intent_classifier.py   # ML intent classification
│   └── nlp_processor.py       # NLP text processing
│
├── 📁 routes/
│   ├── __init__.py
│   └── chatbot_routes.py      # Chatbot API endpoints
│
├── 📁 data/
│   └── chatbot_intents.json   # Training data & responses
│
├── 📁 models/
│   └── intent_classifier.pkl  # Trained ML model
│
├── 📁 templates/              # Bootstrap UI templates
│   ├── base.html              # Template inheritance
│   ├── index.html             # Homepage
│   ├── chat.html              # Chat interface
│   ├── about.html             # About page
│   ├── error.html             # Error pages
│   └── includes/
│       ├── header.html
│       └── footer.html
│
├── 📁 static/                 # CSS, JS, images
│   ├── css/styles.css         # AI-themed styling
│   └── js/chat.js            # Chat interface
│
├── app.py                     # Flask application factory
├── requirements.txt           # ML/NLP dependencies
└── README.md                 # This file
```

## 🧠 AI Technology Stack

### Natural Language Processing
- **NLTK 3.8.1**: Text tokenization, stopword removal, lemmatization
- **spaCy 3.7.0**: Advanced NLP pipeline, entity recognition, linguistic analysis
- **Custom Preprocessing**: Optimized text cleaning and normalization

### Machine Learning
- **scikit-learn 1.3.0**: Intent classification pipeline
- **TF-IDF Vectorization**: Feature extraction from text
- **Logistic Regression**: High-accuracy intent prediction
- **Confidence Scoring**: Response quality assessment

### Web Framework
- **Flask 2.3.3**: Lightweight web framework
- **Bootstrap 5**: Responsive UI with AI-themed design
- **AJAX**: Real-time chat interface

## � Installation & Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### 1. Clone/Download the Project
```bash
cd pythonwebapp
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Train the ML Model (Optional - pre-trained model included)
```bash
# Option 1: Using Flask CLI
python -m flask --app app.py train-chatbot

# Option 2: Using management script (recommended)
python manage.py train
```

### 6. Run the Application
```bash
# Option 1: Using main app file
python app.py

# Option 2: Using run script
python run.py

# Option 3: Using management script (recommended)
python manage.py start
```

The AI chatbot will be available at: `http://127.0.0.1:5000`

## 🧪 Testing & Validation

### Test Chatbot Functionality
```bash
# Option 1: Using Flask CLI
python -m flask --app app.py test-ml

# Option 2: Using management script (recommended)
python manage.py test-ml
```

### Test Sample Conversations
The chatbot is trained on 11 intent categories:
- **Greeting**: Hello, hi, good morning
- **Services**: What do you offer, your services
- **Contact**: How to reach you, contact information
- **Pricing**: Costs, fees, payment information
- **Support**: Technical help, troubleshooting
- **Account**: Login, registration, profile
- **Booking**: Appointments, scheduling
- **Information**: General inquiries
- **Feedback**: Reviews, suggestions
- **Goodbye**: Farewell, end conversation
- **Fallback**: Unknown or unclear queries

### Health Check
Visit: `http://127.0.0.1:5000/api/health`

## 🔧 Configuration

### ML Chatbot Settings (config/config.py)
```python
CHATBOT_CONFIG = {
    'confidence_threshold': 0.3,
    'model_path': 'models/intent_classifier.pkl',
    'intents_file': 'data/chatbot_intents.json',
    'use_spacy': True,
    'use_nltk': True
}
```

### Customizing Responses
Edit `data/chatbot_intents.json` to:
- Add new intent categories
- Modify response templates
- Include additional training patterns
- Customize entity recognition

### Retraining the Model
After modifying intents:
```bash
# Option 1: Using Flask CLI
python -m flask --app app.py train-chatbot

# Option 2: Using management script (recommended)
python manage.py train
```

## 🛠 API Endpoints

### Chat API
```bash
POST /api/chat
Content-Type: application/json

{
    "message": "Hello, what services do you offer?"
}
```

Response:
```json
{
    "response": "Hello! We offer various services including...",
    "confidence": 0.95,
    "intent": "services",
    "entities": []
}
```

### Health Check
```bash
GET /api/health
```

## 🧠 How It Works

### 1. Text Preprocessing
- Tokenization and normalization
- Stopword removal and lemmatization
- Entity recognition and extraction

### 2. Intent Classification
- TF-IDF feature extraction
- ML model prediction with confidence scoring
- Fallback handling for low-confidence responses

### 3. Response Generation
- Intent-based response selection
- Context-aware reply formatting
- Entity incorporation when relevant

## 🎨 UI Features

- **Modern AI Theme**: Gradient designs and robot iconography
- **Responsive Chat Interface**: Real-time messaging with typing indicators
- **Status Indicators**: Live chatbot availability monitoring
- **Error Handling**: Graceful degradation with helpful error pages
- **Accessibility**: Bootstrap-based responsive design

## 🔒 Privacy & Security

- **No External Dependencies**: All processing happens locally
- **No Data Collection**: Conversations are not stored or transmitted
- **Open Source Transparency**: Full access to all algorithms and models
- **Local Model Training**: Complete control over AI behavior
- **No API Keys Required**: Zero external service dependencies

## 📈 Performance

- **Fast Response Times**: Local processing eliminates network latency
- **High Accuracy**: Trained model achieves >90% intent classification accuracy
- **Low Resource Usage**: Optimized for efficient local computation
- **Scalable Architecture**: Modular design supports easy enhancements

## 🔧 Development

### Adding New Intents
1. Edit `data/chatbot_intents.json`
2. Add patterns and responses for new intent
3. Retrain model: `python manage.py train`
4. Test with new queries: `python manage.py test-ml`

### Customizing NLP Pipeline
Modify `services/nlp_processor.py` to:
- Add custom text preprocessing
- Include domain-specific entity recognition
- Implement advanced linguistic analysis

### Enhancing ML Model
Update `services/intent_classifier.py` to:
- Try different ML algorithms
- Implement ensemble methods
- Add cross-validation
- Include hyperparameter tuning

## 🚀 Deployment

### Production Configuration
```bash
export FLASK_ENV=production
export SECRET_KEY=your-production-secret-key
export DEBUG=False
```

### Using Production WSGI Server
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_production_app()"
```

## 📝 Logging

Comprehensive logging includes:
- NLP processing steps
- ML model predictions
- Intent classification confidence
- Response generation logic
- Error handling and debugging

## 🤝 Contributing

1. Follow the modular architecture
2. Add comprehensive logging
3. Include unit tests for new features
4. Document ML model changes
5. Test with diverse conversation scenarios

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**No OpenAI API keys required. No external dependencies. Complete privacy protection.**#   a i c h a t b o t a s s i s t a n t 
 
 
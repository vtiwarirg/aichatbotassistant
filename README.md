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
aichatbotassistant/
│
├── 📁 config/
│   ├── __init__.py
│   └── config.py              # ML chatbot configuration
│
├── 📁 services/
│   ├── __init__.py
│   ├── chatbot_facade.py      # Main chatbot facade
│   ├── ml_chatbot_service.py  # ML chatbot service
│   ├── intent_classifier.py   # ML intent classification
│   ├── nlp_processor.py       # NLP text processing
│   ├── response_generator.py  # Response generation
│   └── conversation_logger.py # Conversation logging
│
├── 📁 routes/
│   ├── __init__.py
│   ├── main_routes.py         # Main web routes
│   ├── api_routes.py          # API endpoints
│   └── health_routes.py       # Health check routes
│
├── 📁 data/
│   └── chatbot_intents.json   # Training data & responses
│
├── 📁 models/                 # Auto-created for trained models
│
├── 📁 logs/                   # Auto-created for application logs
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
│   └── js/                    # JavaScript files
│       ├── chat.js            # Chat interface
│       └── scripts.js         # Additional scripts
│
├── app.py                     # Flask application factory
├── run.py                     # Alternative entry point
├── manage.py                  # Management script (recommended)
├── requirements.txt           # ML/NLP dependencies
├── COMMAND_REFERENCE.md       # Command reference guide
└── README.md                  # This file
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

## 🛠 Installation & Setup (Windows 11)

### Quick Setup (Recommended)
For the easiest setup experience, use the automated setup scripts:

```powershell
# Option 1: PowerShell Script (Recommended)
powershell -ExecutionPolicy Bypass -File setup_windows.ps1

# Option 2: Batch Script
setup_windows.bat
```

### Manual Setup
If you prefer manual setup or the automated scripts fail:

### Prerequisites
- **Python 3.8+** (Download from [python.org](https://python.org))
- **pip** (included with Python)
- **Git** (optional, for cloning)

### 1. Download/Clone the Project
```powershell
# Option 1: Clone from Git
git clone https://github.com/vtiwarirg/aichatbotassistant.git
cd aichatbotassistant

# Option 2: Download ZIP and extract
# Navigate to the extracted folder
cd aichatbotassistant
```

### 2. Create Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell/CMD)
.venv\Scripts\activate

# For Windows PowerShell (if execution policy issues)
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# .venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Fix any compatibility issues (if needed)
pip install --upgrade setuptools wheel
```

### 4. Download spaCy Language Model
```powershell
# Download English language model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded successfully!')"
```

### 5. Initialize the Application
```powershell
# Train the ML model (using management script - recommended)
python manage.py train

# Alternative: using Flask CLI
python -m flask --app app.py train-chatbot

# Verify setup
python manage.py status
```

### 6. Run the Application
```powershell
# Option 1: Using management script (recommended)
python manage.py start

# Option 2: Using main app file
python app.py

# Option 3: Using run script
python run.py
```

The AI chatbot will be available at: `http://127.0.0.1:5000`

### ✅ Verify Setup Success
After completing setup, verify everything works:

```powershell
# Check system status (should show all ✓ green checkmarks)
python manage.py status

# Test ML functionality
python manage.py test-ml

# Quick conversation test
python -c "
from services.chatbot_facade import ChatbotFacade
chatbot = ChatbotFacade()
print('🤖 Chatbot says:', chatbot.get_response('Hello!', 'test')['response'])
"
```

If you see all green checkmarks (✓) and the chatbot responds to "Hello!", your setup is successful!

## 🧪 Testing & Validation

### Test Chatbot Functionality
```bash
# Option 1: Using Flask CLI
python -m flask --app app.py test-ml

# Option 2: Using management script (recommended)
python manage.py test-ml
```

### Test Sample Conversations
The chatbot is trained on 20 intent categories:
- **Greeting**: Hello, hi, good morning
- **Services**: What do you offer, your services
- **Time**: Current time, what time is it
- **Date**: Current date, today's date
- **Weather**: Weather conditions, forecast
- **Contact**: How to reach you, contact information
- **Pricing**: Costs, fees, payment information
- **Support**: Technical help, troubleshooting
- **Account**: Login, registration, profile
- **Booking**: Appointments, scheduling
- **Information**: General inquiries
- **Feedback**: Reviews, suggestions
- **Business Hours**: When are you open, operating hours
- **Location**: Where are you located, address
- **Emergency**: Urgent help, immediate assistance
- **Goodbye**: Farewell, end conversation
- **Help**: Need assistance, help me
- **About**: Information about the service
- **Features**: Available features, capabilities
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

### Production Configuration (Windows)
```powershell
# Set environment variables
$env:FLASK_ENV="production"
$env:SECRET_KEY="your-production-secret-key"
$env:DEBUG="False"
```

### Using Production WSGI Server
```powershell
# Install gunicorn (works on Windows with WSL or use waitress for native Windows)
pip install waitress

# Run with waitress (Windows-compatible)
waitress-serve --host=127.0.0.1 --port=5000 app:app

# Alternative: Using gunicorn with WSL
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "app:app"
```

## ⚠️ Windows 11 Troubleshooting

### Common Setup Issues & Solutions

#### 1. PowerShell Execution Policy Error
**Error**: `cannot be loaded because running scripts is disabled on this system`
**Solution**:
```powershell
# Run PowerShell as Administrator and execute:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate virtual environment:
.venv\Scripts\Activate.ps1
```

#### 2. Python Not Found Error
**Error**: `'python' is not recognized as an internal or external command`
**Solution**:
```powershell
# Check if Python is installed
py --version

# Use 'py' instead of 'python' if needed:
py -m venv .venv
py -m pip install -r requirements.txt
py manage.py status
```

#### 3. spaCy Model Download Fails
**Error**: `Can't find model 'en_core_web_sm'`
**Solution**:
```powershell
# Try different download methods:
python -m spacy download en_core_web_sm --user
# or
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl
```

#### 4. Permission Errors During Installation
**Error**: `PermissionError: [WinError 5] Access is denied`
**Solution**:
```powershell
# Install packages for current user only:
pip install -r requirements.txt --user

# Or run Command Prompt/PowerShell as Administrator
```

#### 5. Port 5000 Already in Use
**Error**: `Address already in use`
**Solution**:
```powershell
# Check what's using port 5000:
netstat -ano | findstr :5000

# Use different port:
$env:PORT="5001"
python manage.py start

# Or kill the process using the port:
taskkill /PID <process_id> /F
```

#### 6. Virtual Environment Issues
**Error**: Virtual environment not activating
**Solution**:
```powershell
# Remove existing .venv and recreate:
rmdir /s .venv
python -m venv .venv

# Use full path to activate:
C:\path\to\your\project\.venv\Scripts\activate

# Or use PowerShell specific activation:
.venv\Scripts\Activate.ps1
```

#### 7. Flask Commands Not Working
**Error**: `'flask' is not recognized`
**Solution**:
```powershell
# Use the management script instead:
python manage.py train
python manage.py status
python manage.py test-ml

# Or use full Flask CLI syntax:
python -m flask --app app.py train-chatbot
```

#### 8. Import Errors
**Error**: `ModuleNotFoundError: No module named 'services'`
**Solution**:
```powershell
# Make sure you're in the correct directory:
cd aichatbotassistant
pwd  # Should show: C:\path\to\aichatbotassistant

# Ensure virtual environment is activated:
.venv\Scripts\activate

# Reinstall requirements:
pip install -r requirements.txt
```

### Quick Windows 11 Setup Verification

Run this command to verify everything is working:
```powershell
# Complete verification script
python manage.py status && echo "✅ Setup Complete!"
```

If you see all green checkmarks (✓), your setup is successful!

### Getting Help

1. **Check the Command Reference**: See `COMMAND_REFERENCE.md`
2. **Run Status Check**: `python manage.py status`
3. **Test ML Functionality**: `python manage.py test-ml`
4. **View Logs**: Check the `logs/` directory for error details

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
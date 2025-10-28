# AI Chatbot Assistant - Command Reference

## 🚀 Quick Start Commands

### Management Script (Recommended)
The easiest way to manage your chatbot:

```bash
# Show all available commands
python manage.py

# Start the Flask development server
python manage.py start

# Train the chatbot model
python manage.py train

# Check application status
python manage.py status

# Test ML functionality
python manage.py test-ml

# Clear conversation logs
python manage.py clear-logs

# Export training data
python manage.py export-data
```

### Direct Flask CLI Commands
Alternative way using Flask CLI:

```bash
# Train the chatbot model
python -m flask --app app.py train-chatbot

# Check application status
python -m flask --app app.py status

# Test ML functionality
python -m flask --app app.py test-ml

# Clear conversation logs
python -m flask --app app.py clear-logs

# Export training data
python -m flask --app app.py export-data
```

### Direct Python Commands
Running the application directly:

```bash
# Start with main app file
python app.py

# Start with run script
python run.py
```

## ❌ Common Errors & Solutions

### Error: `flask : The term 'flask' is not recognized`
**Problem**: Trying to run `flask train-chatbot` directly
**Solution**: Use one of these correct methods:
```bash
# ✅ Correct: Using manage.py (recommended)
python manage.py train

# ✅ Correct: Using Flask CLI
python -m flask --app app.py train-chatbot
```

### Error: `No module named 'flask'`
**Problem**: Flask not installed
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Error: `ModuleNotFoundError: No module named 'services'`
**Problem**: Running from wrong directory
**Solution**: Make sure you're in the project root directory
```bash
cd d:\aichatbotassistant
python manage.py status
```

### Error: `spaCy model 'en_core_web_sm' not found`
**Problem**: spaCy language model not downloaded
**Solution**: Download the model
```bash
python -m spacy download en_core_web_sm
```

## 🔧 Development Workflow

### 1. Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Train initial model
python manage.py train

# Check status
python manage.py status
```

### 2. Daily Development
```bash
# Start development server
python manage.py start

# In another terminal - test changes
python manage.py test-ml

# Check system status
python manage.py status
```

### 3. After Making Changes
```bash
# If you modified intents or training data
python manage.py train

# Test the changes
python manage.py test-ml

# Clear logs if needed
python manage.py clear-logs
```

## 📊 Monitoring Commands

### Check System Health
```bash
# Full system status
python manage.py status

# Quick ML test
python manage.py test-ml

# Web health check (when server is running)
curl http://127.0.0.1:5000/api/health
```

### Analytics & Data
```bash
# Export conversation data
python manage.py export-data

# Clear old logs
python manage.py clear-logs

# Check status for analytics summary
python manage.py status
```

## 🆘 Troubleshooting

### Flask Commands Not Working?
1. Make sure you're in the project directory: `d:\aichatbotassistant`
2. Use the full Flask CLI syntax: `python -m flask --app app.py <command>`
3. Or use the management script: `python manage.py <command>`

### Server Won't Start?
1. Check if another process is using port 5000
2. Try different port: `set PORT=5001` then `python manage.py start`
3. Check logs in `logs/` directory

### ML Model Issues?
1. Retrain the model: `python manage.py train`
2. Check intents file: `data/chatbot_intents.json`
3. Verify spaCy model: `python -m spacy info en_core_web_sm`

### Dependencies Issues?
1. Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
2. Check Python version: `python --version` (needs 3.7+)
3. Update pip: `python -m pip install --upgrade pip`

---

**Remember**: Always use `python manage.py <command>` for the easiest experience!
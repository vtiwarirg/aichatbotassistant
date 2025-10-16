#!/usr/bin/env python3
"""
Management script for AI Chatbot Assistant
Provides easy access to Flask CLI commands
"""
import sys
import subprocess
import os

def run_flask_command(command):
    """Run a Flask CLI command"""
    cmd = ["python", "-m", "flask", "--app", "app.py", command]
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Python not found. Make sure Python is installed and in PATH.")
        return False

def main():
    """Main management function"""
    if len(sys.argv) < 2:
        print("AI Chatbot Management Script")
        print("Usage: python manage.py <command>")
        print()
        print("Available commands:")
        print("  start            - Start the Flask development server")
        print("  train            - Train the chatbot model")
        print("  status           - Check application status")
        print("  test-ml          - Test ML functionality")
        print("  clear-logs       - Clear conversation logs")
        print("  export-data      - Export training data")
        print()
        print("Examples:")
        print("  python manage.py start")
        print("  python manage.py train")
        print("  python manage.py status")
        return

    command = sys.argv[1].lower()
    
    if command in ['start', 'run', 'server']:
        print("🚀 Starting AI Chatbot server...")
        subprocess.run(["python", "run.py"])
    
    elif command in ['train', 'train-chatbot']:
        print("🤖 Training chatbot model...")
        run_flask_command("train-chatbot")
    
    elif command in ['status', 'info']:
        print("🔍 Checking application status...")
        run_flask_command("status")
    
    elif command in ['test', 'test-ml']:
        print("🧪 Testing ML functionality...")
        run_flask_command("test-ml")
    
    elif command in ['clear', 'clear-logs']:
        print("🧹 Clearing conversation logs...")
        run_flask_command("clear-logs")
    
    elif command in ['export', 'export-data']:
        print("📤 Exporting training data...")
        run_flask_command("export-data")
    
    elif command in ['help', '--help', '-h']:
        main()  # Show help
    
    else:
        print(f"Unknown command: {command}")
        print("Run 'python manage.py' to see available commands.")

if __name__ == '__main__':
    main()
from app import create_app

if __name__ == '__main__':
    app = create_app()
    print("🤖 AI Chatbot Assistant starting...")
    print("🌐 Visit: http://127.0.0.1:5000")
    print("💬 Chat interface: http://127.0.0.1:5000/chat")
    print("📝 About: http://127.0.0.1:5000/about")
    print("❤️ Health check: http://127.0.0.1:5000/health")
    app.run(host='127.0.0.1', port=5000, debug=True)
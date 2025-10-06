class ChatBot {
    constructor() {
        this.isTyping = false;
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.initializeChat();
    }
    
    bindEvents() {
        const sendButton = document.getElementById('sendButton');
        const messageInput = document.getElementById('messageInput');
        const clearChatBtn = document.getElementById('clearChatBtn');
        
        if (sendButton) {
            sendButton.addEventListener('click', () => this.sendMessage());
        }
        
        if (messageInput) {
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendMessage();
                }
            });
        }
        
        if (clearChatBtn) {
            clearChatBtn.addEventListener('click', () => this.clearChat());
        }
    }
    
    initializeChat() {
        // Add welcome message
        this.addMessage("Hello! I'm your AI assistant. How can I help you today?", 'bot');
        this.updateStatus('online');
    }
    
    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message || this.isTyping) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        messageInput.value = '';
        
        // Show typing indicator
        this.showTyping();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message
                })
            });
            
            const data = await response.json();
            this.hideTyping();
            
            if (data.success) {
                this.addMessage(data.response, 'bot');
                
                // Show confidence if it's low
                if (data.confidence < 0.5) {
                    this.addSystemMessage(`Confidence: ${(data.confidence * 100).toFixed(1)}% - I might not have understood perfectly.`);
                }
            } else {
                this.addMessage(data.response || 'Sorry, I encountered an error. Please try again.', 'bot');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTyping();
            this.addMessage('Sorry, I\'m having trouble connecting. Please try again.', 'bot');
        }
    }
    
    addMessage(message, type) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const now = new Date();
        const timeString = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        messageDiv.innerHTML = `
            <div class="message-content">
                ${this.escapeHtml(message)}
                <div class="message-time">${timeString}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    addSystemMessage(message) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <small class="text-muted">
                    <i class="fas fa-info-circle me-1"></i>
                    ${this.escapeHtml(message)}
                </small>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    showTyping() {
        this.isTyping = true;
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.style.display = 'flex';
        }
        const sendButton = document.getElementById('sendButton');
        if (sendButton) {
            sendButton.disabled = true;
        }
    }
    
    hideTyping() {
        this.isTyping = false;
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.style.display = 'none';
        }
        const sendButton = document.getElementById('sendButton');
        if (sendButton) {
            sendButton.disabled = false;
        }
    }
    
    clearChat() {
        if (confirm('Are you sure you want to clear the chat?')) {
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                chatMessages.innerHTML = '';
            }
            this.initializeChat();
        }
    }
    
    updateStatus(status) {
        const statusElement = document.getElementById('chatStatus');
        if (statusElement) {
            if (status === 'online') {
                statusElement.textContent = 'Online';
                statusElement.className = 'badge bg-success ms-2';
            } else {
                statusElement.textContent = 'Offline';
                statusElement.className = 'badge bg-secondary ms-2';
            }
        }
    }
    
    showError(message) {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${message}
                </div>
            `;
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize chatbot when page loads
document.addEventListener('DOMContentLoaded', function() {
    new ChatBot();
});
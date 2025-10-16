# Manual Chatbot Testing Guide

## Quick Test Prompts for Each Intent Category

### 1. **GREETING TESTS** 🤝
```
- "Hello"
- "Hi there"
- "Good morning"
- "Hey"
- "What's up?"
```
**Expected:** Welcome message mentioning assistance with business needs, customer support, HR, IT, etc.

### 2. **SERVICES TESTS** 🔧
```
- "What services do you offer?"
- "What can you help me with?"
- "Tell me about your capabilities"
- "What do you do?"
```
**Expected:** List of services including Customer Service, HR, IT, Billing, Product Support, etc.

### 3. **CONTACT TESTS** 📞
```
- "How can I contact you?"
- "What's your phone number?"
- "Contact information please"
- "Email address?"
```
**Expected:** Contact details with email addresses, phone numbers, support channels.

### 4. **BUSINESS HOURS TESTS** 🕐
```
- "What are your business hours?"
- "When are you open?"
- "Support hours?"
- "When can I reach you?"
```
**Expected:** Hours for different departments (24/7 for some, business hours for others).

### 5. **TECHNICAL SUPPORT TESTS** 💻
```
- "I have a technical issue"
- "System not working"
- "Software problem"
- "IT support needed"
```
**Expected:** Troubleshooting steps, IT contact information, technical assistance guidance.

### 6. **BILLING TESTS** 💰
```
- "Billing questions"
- "Invoice issues"
- "Payment problems"
- "Refund request"
```
**Expected:** Billing support information, account portal references, billing contact details.

### 7. **ACCOUNT TESTS** 🔐
```
- "Account issues"
- "Login problems"
- "Password reset"
- "Can't log in"
```
**Expected:** Account troubleshooting steps, password reset guidance, IT support contact.

### 8. **HR POLICIES TESTS** 👥
```
- "HR policies"
- "Employee handbook"
- "Benefits information"
- "Vacation time"
```
**Expected:** HR resources, benefits portal, policy information, HR contact details.

### 9. **PRODUCT SUPPORT TESTS** 📦
```
- "Product issues"
- "How to use product"
- "User manual"
- "Product features"
```
**Expected:** Product documentation, tutorials, support resources, specialist contact.

### 10. **SHIPPING TESTS** 🚚
```
- "Shipping information"
- "Track order"
- "Delivery status"
- "Shipping cost"
```
**Expected:** Tracking information, shipping options, delivery details.

### 11. **RETURNS TESTS** ↩️
```
- "Return policy"
- "Return item"
- "Refund process"
- "Exchange product"
```
**Expected:** Return procedures, time limits, refund information.

### 12. **SECURITY TESTS** 🛡️
```
- "Security concerns"
- "Data security"
- "Privacy policy"
- "Account security"
```
**Expected:** Security information, privacy policy details, data protection measures.

### 13. **COMPLAINTS TESTS** 😞
```
- "File complaint"
- "Unsatisfied with service"
- "Poor experience"
- "Escalate issue"
```
**Expected:** Complaint procedures, manager contact, escalation process.

### 14. **EMERGENCY TESTS** 🚨
```
- "Emergency"
- "Urgent issue"
- "Critical problem"
- "System down"
```
**Expected:** Emergency contact information, 24/7 support, escalation procedures.

### 15. **PRICING TESTS** 💵
```
- "How much does it cost?"
- "What are your prices?"
- "Pricing information"
- "Cost estimate"
```
**Expected:** Pricing information, quote processes, cost details.

### 16. **HELP TESTS** ❓
```
- "Help"
- "I need help"
- "Can you help me?"
- "Support"
```
**Expected:** General assistance message listing available support areas.

### 17. **TIME TESTS** ⏰ (Dynamic Content)
```
- "What time is it?"
- "Current time"
- "Tell me the time"
- "Time now"
```
**Expected:** Actual current time (e.g., "Right now it's 01:52 AM")

### 18. **DATE TESTS** 📅 (Dynamic Content)
```
- "What date is it?"
- "Current date"
- "What's today's date?"
- "Today's date"
```
**Expected:** Actual current date (e.g., "Today's date is October 17, 2025")

### 19. **WEATHER TESTS** 🌤️
```
- "What's the weather?"
- "How's the weather today?"
- "Weather forecast"
- "Is it raining?"
```
**Expected:** Guidance to check weather apps/websites (no real-time weather data).

### 20. **GOODBYE TESTS** 👋
```
- "Goodbye"
- "Bye"
- "Thanks"
- "Thank you"
```
**Expected:** Farewell message with availability information.

### 21. **EDGE CASE TESTS** 🎯
```
- "" (empty message)
- "asdfghjkl" (random text)
- "How much wood would a woodchuck chuck?" (unrelated)
- "!@#$%^&*()" (special characters)
```
**Expected:** Fallback responses asking for clarification or offering general help.

---

## Testing Checklist

### ✅ **Basic Functionality**
- [ ] Greeting responses are welcoming and informative
- [ ] Help requests provide comprehensive service overview
- [ ] Goodbye messages are polite and mention availability

### ✅ **Service Categories**
- [ ] Each department (HR, IT, Billing, etc.) provides relevant information
- [ ] Contact information is provided when appropriate
- [ ] Business hours are clearly communicated

### ✅ **Dynamic Content**
- [ ] Time queries return actual current time
- [ ] Date queries return actual current date
- [ ] Weather queries provide helpful guidance

### ✅ **Error Handling**
- [ ] Empty messages are handled gracefully
- [ ] Nonsensical input gets appropriate fallback responses
- [ ] System maintains helpful tone even for unclear requests

### ✅ **Response Quality**
- [ ] Responses are relevant to the query
- [ ] Confidence levels seem appropriate
- [ ] Language is professional and helpful
- [ ] Information is accurate and useful

---

## Quick Manual Test Commands

### Start the chatbot:
```bash
cd d:\aichatbotassistant
python app.py
```

### Run automated tests:
```bash
cd d:\aichatbotassistant
python test_chatbot_functionality.py
```

### Test individual queries via web interface:
1. Open browser to `http://127.0.0.1:5000`
2. Navigate to chat interface
3. Type test prompts and verify responses

### Test via API directly:
```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it?"}'
```
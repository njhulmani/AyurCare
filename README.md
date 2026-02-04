# AyurCare

AyurCare is a modern, AI-powered Ayurvedic chatbot web application that blends traditional Ayurvedic wisdom with advanced technology. It offers multilingual support, user authentication, persistent chat history, and a visually appealing, professional user interface.

## 🌿 Project Overview

AyurCare is designed to make Ayurvedic knowledge accessible to everyone. Users can interact with the chatbot in English, Hindi, or Kannada, receive personalized remedies (with images), and manage their chat history securely. The application is built with Flask, SQLAlchemy, and a neural network for intent detection.

## ✨ Features

- **Multilingual Support:** Communicate in English, Hindi, or Kannada. The chatbot translates queries and responses for seamless interaction.
- **Voice Interaction:** Speak to the bot and receive spoken responses (including Hindi and Kannada).
- **Visual Remedies:** Suggested remedies include images and names for better understanding.
- **User Authentication:** Secure login, registration, and logout with session management.
- **Persistent Chat History:** Each user's chat sessions are saved, can be reviewed, managed (delete/download sessions).
- **Modern UI:** Responsive, attractive design consistent across landing, login, and chat pages.
- **Session Management:** Sidebar for switching between chat sessions, with remedy images and names displayed per message.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/vvnserrao/AYURCARE.git
   cd AyurBot
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download NLTK data (if not already present):**
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('punkt')
   ```
5. **Set up environment variables:**
   - Create a `.env` file in the root directory and add your API keys and secret key:
     ```env
     GOOGLE_API_KEY=your-google-api-key
     SEARCH_ENGINE_ID=your-search-engine-id
     OPENAI_API_KEY=your-openai-api-key
     ```
6. **Run the application:**
   ```bash
   python app.py
   ```

## 🖥️ Usage
- Visit the landing page to learn about AyurCare.
- Register or log in to access the chatbot.
- Start a new chat or review previous sessions in the sidebar.
- Download or delete chat sessions as needed.

## 🛠️ Tech Stack
- **Backend:** Flask, Flask-Login, Flask-SQLAlchemy, Keras, NLTK, OpenAI API
- **Frontend:** HTML5, CSS3, JavaScript
- **Database:** SQLite
- **Other:** Google Custom Search API, Deep Translator

## 📁 Folder Structure
```
AyurBot/
├── app.py                # Main Flask application
├── models.py             # Database models
├── requirements.txt      # Python dependencies
├── static/               # Static files (CSS, images)
├── templates/            # HTML templates
├── training.py           # Model training script
├── model.h5              # Trained neural network model (ignored in git)
├── data.json             # Intents and remedies data (ignored in git)
├── labels.pkl, texts.pkl # Model data (ignored in git)
├── nltk_data/            # NLTK data (ignored in git)
├── instance/             # Runtime files (ignored in git)
├── venv/                 # Virtual environment (ignored in git)
└── ...
```

## 📸 Screenshots

<img width="1727" alt="Home Page" src="https://github.com/user-attachments/assets/d9ffb5d1-4bdf-4fce-8b01-349fd6fde369" />

<br><br>
<img width="1727" alt="data set response" src="https://github.com/user-attachments/assets/2a70fce8-8971-44cf-916f-04eb9a3bc2aa" />

<br><br>
<img width="1728" alt="llama reposne" src="https://github.com/user-attachments/assets/f7840514-a011-4cac-a8a7-92656ae03e25" />


<br><br>
<img width="1728" alt="Screenshot 2025-06-23 at 5 17 37 PM" src="https://github.com/user-attachments/assets/3825632d-6937-4d10-b5c0-124a4a39d4de" />

<br>

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.


## 🙏 Acknowledgements
- [NLTK](https://www.nltk.org/), [Flask](https://flask.palletsprojects.com/), [Keras](https://keras.io/), [OpenAI](https://openai.com/), [Deep Translator](https://pypi.org/project/deep-translator/)

---
© 2026 AyurCare - Developed by Nagaraj Hulmani
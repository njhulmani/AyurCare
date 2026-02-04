from dotenv import load_dotenv
load_dotenv()

import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import random
import json
from keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
import requests
import openai
import os
import ollama
from translate import Translator
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, Chat, ChatSession, Feedback
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ayurmate.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize NLTK and download required data
try:
    # Set NLTK data path to a local directory
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)

    # Try to load the data, download if not available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    
    lemmatizer = WordNetLemmatizer()
    logger.info("Successfully initialized NLTK and environment")
except Exception as e:
    logger.error(f"Error initializing NLTK: {str(e)}")
    raise

# API keys from .env with validation
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID") or os.getenv("GOOGLE_CX")  # Support both names
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY  # Set OpenAI API key

if not all([GOOGLE_API_KEY, SEARCH_ENGINE_ID]):
    logger.warning("Google API keys are missing from .env file")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key is missing (only needed for DALL-E image generation)")

try:
    # Load model and data
    logger.info("Loading model and data files...")
    model = load_model('model.h5', compile=False)  # Skip compilation for faster loading
    with open('data.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    with open('texts.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('labels.pkl', 'rb') as f:
        classes = pickle.load(f)
    logger.info("Successfully loaded model and data files")
except Exception as e:
    logger.error(f"Error loading model or data files: {str(e)}")
    raise

# Cache for known ingredients and images
known_ingredients = set(
    r.lower() for intent in intents['intents'] 
    for r in intent.get('remedy', []) 
    if r.lower() != "unknown"
)
image_cache = {}
logger.info(f"Loaded {len(known_ingredients)} known ingredients")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('landing.html')


@app.route('/remedies')
@app.route('/remedies.html')
def remedies():
    return render_template('remedies.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(email=email).first()
        
        if not user:
            flash('No account found with this email. Please register first.', 'error')
            return redirect(url_for('login'))
        
        if not user.check_password(password):
            flash('Incorrect password. Please try again.', 'error')
            return redirect(url_for('login'))
        
        login_user(user)
        return redirect(url_for('chat'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please login instead.', 'error')
            return redirect(url_for('login'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/get_chat_sessions')
@login_required
def get_chat_sessions():
    sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.updated_at.desc()).all()
    return jsonify([{
        'id': session.id,
        'title': session.title,
        'created_at': session.created_at.isoformat(),
        'updated_at': session.updated_at.isoformat(),
        'preview': session.messages[-1].message if session.messages else None
    } for session in sessions])

@app.route('/get_chat_session/<int:session_id>')
@login_required
def get_chat_session(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    return jsonify({
        'id': session.id,
        'title': session.title,
        'messages': [{
            'id': msg.id,
            'message': msg.message,
            'response': msg.response,
            'timestamp': msg.timestamp.isoformat(),
            'language': msg.language,
            'images': msg.images,
            'remedy_names': msg.remedy_names
        } for msg in session.messages]
    })

@app.route('/delete_chat_session/<int:session_id>', methods=['DELETE'])
@login_required
def delete_chat_session(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    try:
        db.session.delete(session)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/create_chat_session', methods=['POST'])
@login_required
def create_chat_session():
    data = request.get_json()
    session = ChatSession(
        user_id=current_user.id,
        title=data.get('title', 'New Chat')
    )
    db.session.add(session)
    db.session.commit()
    return jsonify({'session_id': session.id})

@app.route('/get_bot_response', methods=['POST'])
@login_required
def get_bot_response():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        selected_language = data.get('language', 'en')
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
            
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
        
        # Store original text for response translation
        original_text = user_message
        
        # Translate user query to English if not already in English
        if selected_language != 'en':
            logger.info(f"Translating from {selected_language} to en: {user_message}")
            translated_message = translate_text(user_message, 'en')
            logger.info(f"Translated message: {translated_message}")
            # Only use translated if it's different from original (translation succeeded)
            if translated_message and translated_message != user_message:
                user_message = translated_message
            else:
                logger.warning(f"Translation from {selected_language} to en failed or returned same text")

        # Get bot response using the existing chatbot logic
        response_text, remedies = chatbot_response(user_message)
        logger.info(f"Bot response: {response_text}")
        logger.info(f"Remedies found: {remedies}")
        
        # Get images for remedies
        image_urls = []
        remedy_names = []
        if remedies and "unknown" not in remedies:
            logger.info(f"Fetching images for {len(remedies)} remedies")
            for remedy in remedies:
                logger.info(f"Processing remedy: {remedy}")
                images = get_remedy_with_image(remedy)
                logger.info(f"Images returned for {remedy}: {images}")
                if images:
                    image_urls.append(images[0])
                    formatted_name = remedy.replace('_', ' ').title()
                    remedy_names.append(formatted_name)
        
        logger.info(f"Final image_urls: {image_urls}")
        logger.info(f"Final remedy_names: {remedy_names}")
        
        # Translate response back to user's language if not English
        if selected_language != 'en':
            logger.info(f"Translating response to {selected_language}: {response_text}")
            translated_response = translate_text(response_text, selected_language)
            logger.info(f"Translated response: {translated_response}")
            if translated_response and translated_response != response_text:
                response_text = translated_response
            else:
                logger.warning(f"Translation to {selected_language} failed, using original English response")
        
        # Save the chat to database with images and remedy names
        chat = Chat(
            session_id=session_id,
            user_id=current_user.id,
            message=original_text,
            response=response_text,
            language=selected_language,
            timestamp=datetime.utcnow(),
            images=json.dumps(image_urls),
            remedy_names=json.dumps(remedy_names)
        )
        db.session.add(chat)
        
        # Update session title if it's the first message
        if len(session.messages) == 0:
            session.title = original_text[:50] + ('...' if len(original_text) > 50 else '')
        
        session.updated_at = datetime.utcnow()
        db.session.commit()
        
        response_data = {
            "text": response_text,
            "images": image_urls if image_urls else [],
            "remedyNames": remedy_names if remedy_names else []
        }
        logger.info(f"Returning response with {len(image_urls)} images")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in get_bot_response: {str(e)}")
        return jsonify({
            "text": "I apologize, but I encountered an error. Please try asking your health-related question again.",
            "images": [],
            "remedyNames": []
        })

@app.route("/translate", methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        text = data.get('text')
        target_language = data.get('target_language')
        
        if not text or not target_language:
            return jsonify({'error': 'Missing text or target language'}), 400
            
        # Create translator for the target language
        translator = Translator(to_lang=target_language)
        
        # Split text into smaller chunks if it's too long
        max_chunk_size = 500
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        # Translate each chunk
        translated_chunks = []
        for chunk in chunks:
            try:
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
            except Exception as e:
                logger.error(f"Error translating chunk: {str(e)}")
                translated_chunks.append(chunk)  # Keep original if translation fails
        
        # Combine translated chunks
        translated_text = ' '.join(translated_chunks)
        
        return jsonify({
            'translated_text': translated_text
        })
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({'error': 'Translation failed'}), 500

def get_ayurvedic_image(query):
    """Fetch Ayurvedic image using Google Custom Search API with caching."""
    if query in image_cache:
        return image_cache[query]

    try:
        logger.debug(f"Fetching image for query: {query}")
        url = f"https://www.googleapis.com/customsearch/v1"
        
        # Enhance search query to focus on natural ingredients
        search_terms = {
            'ginger': 'fresh ginger root',
            'tulsi': 'fresh tulsi leaves plant',
            'holy basil': 'fresh tulsi holy basil leaves',
            'honey': 'pure natural honey comb',
            'turmeric': 'fresh turmeric root',
            'cinnamon': 'natural cinnamon bark',
            'pepper': 'black pepper seeds',
            'cardamom': 'green cardamom pods',
            'mint': 'fresh mint leaves',
            'lemon': 'fresh lemon fruit',
            'garlic': 'fresh garlic cloves',
            'neem': 'fresh neem leaves',
            'aloe vera': 'fresh aloe vera leaf',
            'amla': 'fresh amla fruit',
            'ashwagandha': 'ashwagandha root natural',
            'cumin': 'cumin seeds natural',
            'fennel': 'fennel seeds natural',
            'fenugreek': 'fenugreek seeds natural',
            'shatavari': 'shatavari root ayurvedic',
            'shilajit': 'shilajit resin natural',
            'safed musli': 'safed musli root',
            'ashoka': 'ashoka bark ayurvedic',
            'almonds': 'almonds nuts natural',
            'dates': 'dates fruit natural'
        }
        
        # Get specific search term if available, otherwise use generic format
        search_query = query.lower().strip()
        found_match = False
        for key, value in search_terms.items():
            if key.lower() in search_query:
                search_query = value
                found_match = True
                break
        
        if not found_match:
            # Clean up the query and make it more specific
            search_query = f"natural fresh {query} ayurvedic herb ingredient -bottle -package -product -supplement"
        
        params = {
            "q": search_query,
            "cx": SEARCH_ENGINE_ID,
            "searchType": "image",
            "key": GOOGLE_API_KEY,
            "num": 1,
            "imgSize": "LARGE",
            "imgType": "photo",
            "safe": "active"
        }
        
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        data = response.json()
        
        if "items" in data:
            image_url = data["items"][0]["link"]
            image_cache[query] = image_url
            logger.debug(f"Successfully found image for {query}")
            return image_url
            
        logger.debug(f"No images found for {query}")
        return None
    except Exception as e:
        logger.error(f"Error fetching image for {query}: {str(e)}")
        return None

def generate_ayurvedic_image(query):
    """Generate Ayurvedic image using OpenAI's DALL-E."""
    try:
        response = openai.Image.create(
            prompt=f"High-quality image of {query} used in Ayurveda.",
            n=1,
            size="512x512"
        )
        return response["data"][0]["url"]
    except Exception as e:
        logger.error(f"Error generating image with DALL-E: {str(e)}")
        return None

def get_remedy_with_image(remedy_name):
    """Get images for a valid remedy with proper validation."""
    logger.info(f"get_remedy_with_image called with: {remedy_name}")
    logger.info(f"Known ingredients: {list(known_ingredients)[:10]}...")  # Log first 10
    
    if not remedy_name:
        logger.warning(f"Empty remedy_name provided")
        return []
    
    if remedy_name.lower() not in known_ingredients:
        logger.warning(f"Remedy '{remedy_name}' not in known_ingredients")
        return []

    try:
        image_urls = []
        google_image = get_ayurvedic_image(remedy_name)
        logger.info(f"Google image for {remedy_name}: {google_image}")
        if google_image:
            image_urls.append(google_image)
        
        if not image_urls:
            logger.info(f"No Google image found, trying AI image for {remedy_name}")
            ai_image = generate_ayurvedic_image(remedy_name)
            if ai_image:
                image_urls.append(ai_image)
        
        logger.info(f"Final image_urls for {remedy_name}: {image_urls}")
        return image_urls[:3]  # Limit to 3 images max
    except Exception as e:
        logger.error(f"Error in get_remedy_with_image for {remedy_name}: {str(e)}")
        return []

def clean_up_sentence(sentence):
    """Clean and lemmatize input sentence."""
    try:
        words = sentence.lower().split()
        return [lemmatizer.lemmatize(word) for word in words]
    except Exception as e:
        logger.error(f"Error in clean_up_sentence: {str(e)}")
        return []

def bow(sentence, words, show_details=False):
    """Convert sentence to bag of words."""
    try:
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        logger.debug(f"found in bag: {w}")
        return np.array(bag)
    except Exception as e:
        logger.error(f"Error in bow: {str(e)}")
        return np.zeros(len(words))

def predict_class(sentence, model):
    """Predict intent class for input sentence."""
    try:
        logger.debug(f"Predicting class for: {sentence}")
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
        logger.debug(f"Prediction results: {return_list}")
        return return_list
    except Exception as e:
        logger.error(f"Error in predict_class: {str(e)}")
        return []

def getResponse(ints, intents_json):
    """Get response based on predicted intent."""
    try:
        if not ints:
            logger.debug("No intents found")
            return "I'm not sure how to help with that. Could you please ask about a specific health concern?", ["unknown"]
        
        tag = ints[0]['intent']
        logger.debug(f"Getting response for intent: {tag}")
        list_of_intents = intents_json['intents']
        
        for i in list_of_intents:
            if i['tag'].lower() == tag.lower():  # Case-insensitive comparison
                response = random.choice(i['responses'])
                remedies = i.get('remedy', ["unknown"])
                
                # Format remedies list for better display
                if remedies != ["unknown"]:
                    remedy_list = ", ".join([r.title() for r in remedies[:-1]])
                    if len(remedies) > 1:
                        remedy_list += f" and {remedies[-1].title()}"
                    else:
                        remedy_list = remedies[0].title()
                    
                    # Add remedy names to response if not already mentioned
                    if not any(remedy.lower() in response.lower() for remedy in remedies):
                        response += f"\n\nKey ingredients: {remedy_list}"
                
                logger.debug(f"Found response: {response} with remedies: {remedies}")
                return response, remedies
        
        logger.debug(f"No response found for intent: {tag}")
        return "I'm not sure how to help with that specific concern. Could you please provide more details about your health issue?", ["unknown"]
    except Exception as e:
        logger.error(f"Error in getResponse: {str(e)}")
        return "I apologize, but I encountered an error. Please try rephrasing your health concern.", ["unknown"]

def get_llama_response(message):
    """Get response from LLaMA model with proper error handling."""
    try:
        # Check if it's a health-related query
        health_keywords = [
            'health', 'pain', 'ache', 'disease', 'remedy', 'cure', 'treatment',
            'medicine', 'ayurvedic', 'ayurveda', 'body', 'immunity', 'weight',
            'sleep', 'digestion', 'stress', 'anxiety', 'tired', 'fatigue',
            'energy', 'wellness', 'healing', 'natural', 'herbs', 'herbal',
            'throat', 'cough', 'cold', 'fever', 'headache', 'stomach'
        ]
        
        is_health_related = any(keyword in message.lower() for keyword in health_keywords)
        
        if not is_health_related:
            return "I am an Ayurvedic health assistant. I can only help you with health-related concerns. Please feel free to ask me about any health issues you're experiencing."

        # Simplified prompt for natural responses
        prompt = f"""You are an Ayurvedic expert. The user is experiencing {message}. 
Provide a simple Ayurvedic remedy in a natural, conversational way. Follow this example format:

"Take 1 teaspoon of fresh ginger juice and 1 teaspoon of lemon juice. Add a pinch of black salt and ½ teaspoon of zeera powder. Take this mixture 3 times after meals. This will help with indigestion."

Keep your response:
1. In simple, clear sentences
2. Include ingredients and their quantities
3. Include preparation method
4. Include dosage and timing
5. End with the benefit
6. Keep it brief (2-3 sentences)
7. Do not use labels like 'Remedy:', 'Ingredients:', etc.
8. Write in a natural, flowing way

Remember to keep the response very brief and easy to understand."""

        response = ollama.chat(
            model='llama3.2:latest',
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {"role": "user", "content": message}
            ],
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 512
            }
        )
        
        # Clean up the response to ensure it's in the right format
        response_text = response['message']['content'].strip()
        
        # Remove any remaining labels if they exist
        response_text = response_text.replace('Remedy:', '').replace('Ingredients:', '').replace('Method:', '').replace('Dosage:', '')
        
        # Ensure it starts with a capital letter and ends with a period
        response_text = response_text[0].upper() + response_text[1:]
        if not response_text.endswith('.'):
            response_text += '.'
            
        return response_text
    except Exception as e:
        logger.error(f"LLaMA Error: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try asking about a specific health concern."

def translate_text(text, target_language):
    """Translate text to target language with proper error handling and multi-language support."""
    try:
        if not text:
            logger.warning("Empty text provided to translate_text")
            return text
            
        # Define common Hindi-to-English medical phrase mappings
        hindi_to_english = {
            'पेट दर्द': 'stomach pain',
            'पेट': 'stomach',
            'दर्द': 'pain',
            'सिर दर्द': 'headache',
            'सिरदर्द': 'headache',
            'सिर': 'head',
            'बुखार': 'fever',
            'खांसी': 'cough',
            'सर्दी': 'cold',
            'जुकाम': 'cold',
            'गले': 'throat',
            'में दर्द': 'pain',
            'है': '',
            'का': '',
            'मुझे': 'I have',
            'बुखार है': 'fever',
            'थकान': 'fatigue',
            'कमजोरी': 'weakness',
            'उल्टी': 'vomiting',
            'दस्त': 'diarrhea',
            'अपच': 'indigestion',
            'गैस': 'gas',
            'एसिडिटी': 'acidity',
            'कब्ज': 'constipation',
            'पीठ दर्द': 'back pain',
            'जोड़ों में दर्द': 'joint pain',
            'घुटने में दर्द': 'knee pain',
        }
        
        # Kannada-to-English medical phrase mappings
        kannada_to_english = {
            'ಹೊಟ್ಟೆ ನೋವು': 'stomach pain',
            'ಹೊಟ್ಟೆ': 'stomach',
            'ನೋವು': 'pain',
            'ತಲೆ ನೋವು': 'headache',
            'ತಲೆನೋವು': 'headache',
            'ತಲೆ': 'head',
            'ಜ್ವರ': 'fever',
            'ಕೆಮ್ಮು': 'cough',
            'ಶೀತ': 'cold',
            'ಗಂಟಲು': 'throat',
            'ಇದೆ': '',
            'ನನಗೆ': 'I have',
            'ಆಯಾಸ': 'fatigue',
            'ದೌರ್ಬಲ್ಯ': 'weakness',
            'ವಾಂತಿ': 'vomiting',
            'ಅತಿಸಾರ': 'diarrhea',
            'ಅಜೀರ್ಣ': 'indigestion',
            'ಅನಿಲ': 'gas',
            'ಆಮ್ಲತೆ': 'acidity',
            'ಮಲಬದ್ಧತೆ': 'constipation',
            'ಬೆನ್ನು ನೋವು': 'back pain',
            'ಕೀಲು ನೋವು': 'joint pain',
            'ಮೊಣಕಾಲು ನೋವು': 'knee pain',
        }
        
        # Telugu-to-English medical phrase mappings
        telugu_to_english = {
            'కడుపు నొప్పి': 'stomach pain',
            'కడుపు': 'stomach',
            'నొప్పి': 'pain',
            'తల నొప్పి': 'headache',
            'తలనొప్పి': 'headache',
            'తల': 'head',
            'జ్వరం': 'fever',
            'దగ్గు': 'cough',
            'జలుబు': 'cold',
            'చలి': 'cold',
            'గొంతు': 'throat',
            'నాకు': 'I have',
            'ఉంది': '',
            'అలసట': 'fatigue',
            'బలహీనత': 'weakness',
            'వాంతులు': 'vomiting',
            'విరేచనాలు': 'diarrhea',
            'అజీర్ణం': 'indigestion',
            'గ్యాస్': 'gas',
            'ఆమ్లత్వం': 'acidity',
            'మలబద్ధకం': 'constipation',
            'వెన్ను నొప్పి': 'back pain',
            'కీళ్ల నొప్పి': 'joint pain',
            'మోకాలి నొప్పి': 'knee pain',
        }
        
        # Tamil-to-English medical phrase mappings
        tamil_to_english = {
            'வயிற்று வலி': 'stomach pain',
            'வயிறு': 'stomach',
            'வலி': 'pain',
            'தலை வலி': 'headache',
            'தலைவலி': 'headache',
            'தலை': 'head',
            'காய்ச்சல்': 'fever',
            'இருமல்': 'cough',
            'சளி': 'cold',
            'தொண்டை': 'throat',
            'எனக்கு': 'I have',
            'உள்ளது': '',
            'சோர்வு': 'fatigue',
            'பலவீனம்': 'weakness',
            'வாந்தி': 'vomiting',
            'வயிற்றுப்போக்கு': 'diarrhea',
            'அஜீரணம்': 'indigestion',
            'வாயு': 'gas',
            'அமிலத்தன்மை': 'acidity',
            'மலச்சிக்கல்': 'constipation',
            'முதுகு வலி': 'back pain',
            'மூட்டு வலி': 'joint pain',
            'முழங்கால் வலி': 'knee pain',
        }
        
        english_to_hindi = {
            'stomach pain': 'पेट दर्द',
            'headache': 'सिर दर्द',
            'fever': 'बुखार',
            'cough': 'खांसी',
            'cold': 'सर्दी',
            'throat pain': 'गले में दर्द',
            'fatigue': 'थकान',
            'weakness': 'कमजोरी',
            'vomiting': 'उल्टी',
            'diarrhea': 'दस्त',
            'indigestion': 'अपच',
            'gas': 'गैस',
            'acidity': 'एसिडिटी',
            'constipation': 'कब्ज',
            'back pain': 'पीठ दर्द',
            'joint pain': 'जोड़ों में दर्द',
            'knee pain': 'घुटने में दर्द',
            'take': 'लें',
            'ginger': 'अदरक',
            'honey': 'शहद',
            'lemon': 'नींबू',
            'turmeric': 'हल्दी',
            'water': 'पानी',
            'times': 'बार',
            'daily': 'रोज',
            'this will help': 'यह मदद करेगा',
        }
        
        english_to_kannada = {
            'stomach pain': 'ಹೊಟ್ಟೆ ನೋವು',
            'headache': 'ತಲೆ ನೋವು',
            'fever': 'ಜ್ವರ',
            'cough': 'ಕೆಮ್ಮು',
            'cold': 'ಶೀತ',
            'throat pain': 'ಗಂಟಲು ನೋವು',
            'fatigue': 'ಆಯಾಸ',
            'weakness': 'ದೌರ್ಬಲ್ಯ',
            'vomiting': 'ವಾಂತಿ',
            'diarrhea': 'ಅತಿಸಾರ',
            'indigestion': 'ಅಜೀರ್ಣ',
            'gas': 'ಅನಿಲ',
            'acidity': 'ಆಮ್ಲತೆ',
            'constipation': 'ಮಲಬದ್ಧತೆ',
            'back pain': 'ಬೆನ್ನು ನೋವು',
            'joint pain': 'ಕೀಲು ನೋವು',
            'knee pain': 'ಮೊಣಕಾಲು ನೋವು',
            'take': 'ತೆಗೆದುಕೊಳ್ಳಿ',
            'ginger': 'ಶುಂಠಿ',
            'honey': 'ಜೇನು',
            'lemon': 'ನಿಂಬೆ',
            'turmeric': 'ಅರಿಶಿನ',
            'water': 'ನೀರು',
            'times': 'ಬಾರಿ',
            'daily': 'ದಿನಂಪ್ರತಿ',
            'this will help': 'ಇದು ಸಹಾಯ ಮಾಡುತ್ತದೆ',
        }
        
        english_to_telugu = {
            'stomach pain': 'కడుపు నొప్పి',
            'headache': 'తల నొప్పి',
            'fever': 'జ్వరం',
            'cough': 'దగ్గు',
            'cold': 'జలుబు',
            'throat pain': 'గొంతు నొప్పి',
            'fatigue': 'అలసట',
            'weakness': 'బలహీనత',
            'vomiting': 'వాంతులు',
            'diarrhea': 'విరేచనాలు',
            'indigestion': 'అజీర్ణం',
            'gas': 'గ్యాస్',
            'acidity': 'ఆమ్లత్వం',
            'constipation': 'మలబద్ధకం',
            'back pain': 'వెన్ను నొప్పి',
            'joint pain': 'కీళ్ల నొప్పి',
            'knee pain': 'మోకాలి నొప్పి',
            'take': 'తీసుకోండి',
            'ginger': 'అల్లం',
            'honey': 'తేనె',
            'lemon': 'నిమ్మకాయ',
            'turmeric': 'పసుపు',
            'water': 'నీరు',
            'times': 'సార్లు',
            'daily': 'ప్రతిరోజు',
            'this will help': 'ఇది సహాయం చేస్తుంది',
        }
        
        english_to_tamil = {
            'stomach pain': 'வயிற்று வலி',
            'headache': 'தலை வலி',
            'fever': 'காய்ச்சல்',
            'cough': 'இருமல்',
            'cold': 'சளி',
            'throat pain': 'தொண்டை வலி',
            'fatigue': 'சோர்வு',
            'weakness': 'பலவீனம்',
            'vomiting': 'வாந்தி',
            'diarrhea': 'வயிற்றுப்போக்கு',
            'indigestion': 'அஜீரணம்',
            'gas': 'வாயு',
            'acidity': 'அமிலத்தன்மை',
            'constipation': 'மலச்சிக்கல்',
            'back pain': 'முதுகு வலி',
            'joint pain': 'மூட்டு வலி',
            'knee pain': 'முழங்கால் வலி',
            'take': 'எடுத்துக்கொள்ளுங்கள்',
            'ginger': 'இஞ்சி',
            'honey': 'தேன்',
            'lemon': 'எலுமிச்சை',
            'turmeric': 'மஞ்சள்',
            'water': 'தண்ணீர்',
            'times': 'முறை',
            'daily': 'தினசரி',
            'this will help': 'இது உதவும்',
        }
        
        target_lang_code = target_language
        text_to_translate = text.strip()
        
        # If translating FROM Hindi/Kannada/Telugu/Tamil TO English
        if target_lang_code == 'en':
            text_lower = text_to_translate.lower()
            
            # Detect script/language by Unicode ranges
            has_kannada = any(ord(char) >= 0x0C80 and ord(char) <= 0x0CFF for char in text_to_translate)
            has_telugu = any(ord(char) >= 0x0C00 and ord(char) <= 0x0C7F for char in text_to_translate)
            has_tamil = any(ord(char) >= 0x0B80 and ord(char) <= 0x0BFF for char in text_to_translate)
            has_hindi = any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in text_to_translate)
            
            if has_kannada:
                logger.info(f"Detected Kannada text, translating to English: '{text_to_translate}'")
                for kannada, english in kannada_to_english.items():
                    if kannada in text_to_translate:
                        text_lower = text_lower.replace(kannada, english)
                        logger.info(f"Matched Kannada phrase: '{kannada}' -> '{english}'")
            elif has_telugu:
                logger.info(f"Detected Telugu text, translating to English: '{text_to_translate}'")
                for telugu, english in telugu_to_english.items():
                    if telugu in text_to_translate:
                        text_lower = text_lower.replace(telugu, english)
                        logger.info(f"Matched Telugu phrase: '{telugu}' -> '{english}'")
            elif has_tamil:
                logger.info(f"Detected Tamil text, translating to English: '{text_to_translate}'")
                for tamil, english in tamil_to_english.items():
                    if tamil in text_to_translate:
                        text_lower = text_lower.replace(tamil, english)
                        logger.info(f"Matched Tamil phrase: '{tamil}' -> '{english}'")
            elif has_hindi:
                logger.info(f"Detected Hindi text, translating to English: '{text_to_translate}'")
                for hindi, english in hindi_to_english.items():
                    if hindi in text_to_translate:
                        text_lower = text_lower.replace(hindi, english)
                        logger.info(f"Matched Hindi phrase: '{hindi}' -> '{english}'")
            
            # Clean up extra spaces
            text_lower = ' '.join(text_lower.split())
            if text_lower != text_to_translate.lower() and text_lower.strip():
                logger.info(f"Direct translation successful: '{text_lower}'")
                return text_lower
        
        # If translating FROM English TO Hindi
        elif target_lang_code == 'hi':
            logger.info(f"Translating English to Hindi: '{text_to_translate[:50]}...'")
            # Try direct phrase matching for common terms
            result = text_to_translate
            for english, hindi in english_to_hindi.items():
                if english.lower() in result.lower():
                    # Case-insensitive replacement
                    import re
                    result = re.sub(re.escape(english), hindi, result, flags=re.IGNORECASE)
            
            if result != text_to_translate:
                logger.info(f"Partial Hindi translation applied")
                # Fall through to use translate library for remaining text
                text_to_translate = result
        
        # If translating FROM English TO Kannada
        elif target_lang_code == 'kn':
            logger.info(f"Translating English to Kannada: '{text_to_translate[:50]}...'")
            # Try direct phrase matching for common terms
            result = text_to_translate
            for english, kannada in english_to_kannada.items():
                if english.lower() in result.lower():
                    # Case-insensitive replacement
                    import re
                    result = re.sub(re.escape(english), kannada, result, flags=re.IGNORECASE)
            
            if result != text_to_translate:
                logger.info(f"Partial Kannada translation applied")
                # Fall through to use translate library for remaining text
                text_to_translate = result
        
        # If translating FROM English TO Telugu
        elif target_lang_code == 'te':
            logger.info(f"Translating English to Telugu: '{text_to_translate[:50]}...'")
            # Try direct phrase matching for common terms
            result = text_to_translate
            for english, telugu in english_to_telugu.items():
                if english.lower() in result.lower():
                    # Case-insensitive replacement
                    import re
                    result = re.sub(re.escape(english), telugu, result, flags=re.IGNORECASE)
            
            if result != text_to_translate:
                logger.info(f"Partial Telugu translation applied")
                # Fall through to use translate library for remaining text
                text_to_translate = result
        
        # If translating FROM English TO Tamil
        elif target_lang_code == 'ta':
            logger.info(f"Translating English to Tamil: '{text_to_translate[:50]}...'")
            # Try direct phrase matching for common terms
            result = text_to_translate
            for english, tamil in english_to_tamil.items():
                if english.lower() in result.lower():
                    # Case-insensitive replacement
                    import re
                    result = re.sub(re.escape(english), tamil, result, flags=re.IGNORECASE)
            
            if result != text_to_translate:
                logger.info(f"Partial Tamil translation applied")
                # Fall through to use translate library for remaining text
                text_to_translate = result
        
        if len(text_to_translate) > 500:  # Limit text length for translation
            logger.warning(f"Text too long for translation ({len(text_to_translate)} chars), truncating")
            text_to_translate = text_to_translate[:500]
        
        # Use translate library for additional translation
        logger.info(f"Using translate library for language code: {target_lang_code}")
        translator = Translator(to_lang=target_lang_code)
        translated = translator.translate(text_to_translate)
        
        if not translated or not translated.strip():
            logger.warning(f"Translation returned empty result, using original or partially translated")
            return text_to_translate if target_lang_code == 'en' else text
        
        # Clean the translated text
        translated = translated.strip()
        logger.info(f"Translation successful: '{translated[:50]}...'")
        return translated
    except Exception as e:
        logger.error(f"Translation error for language '{target_language}': {str(e)}")
        logger.error(f"Original text: {text[:100] if len(text) > 100 else text}")
        # Return partially translated text if available, otherwise original
        return text_to_translate if 'text_to_translate' in locals() else text

def extract_remedies_from_text(text):
    """Extract known remedies from text."""
    if not text:
        return ["unknown"]
    found = [
        ingredient for ingredient in known_ingredients 
        if f" {ingredient.lower()} " in f" {text.lower()} "
    ]
    return found if found else ["unknown"]

def chatbot_response(msg):
    """Generate chatbot response with proper logging."""
    try:
        logger.info(f"Processing message: {msg}")
        
        # Check if user is asking for alternative remedies
        if "alternative" in msg.lower() or "not satisfied" in msg.lower() or "another" in msg.lower():
            logger.info("User requesting alternative remedy, using LLaMA")
            gpt_reply = get_llama_response(msg)
            extracted_remedies = extract_remedies_from_text(gpt_reply)
            return gpt_reply, extracted_remedies

        # Check for multiple symptoms
        symptoms = ["fever", "headache", "cough", "cold", "stomach", "pain", "digestion"]
        found_symptoms = [s for s in symptoms if s in msg.lower()]
        
        if len(found_symptoms) > 1:
            logger.info(f"Multiple symptoms detected: {found_symptoms}, using LLaMA")
            gpt_reply = get_llama_response(msg)
            extracted_remedies = extract_remedies_from_text(gpt_reply)
            return gpt_reply, extracted_remedies

        # For single symptoms, try the model first
        ints = predict_class(msg, model)
        logger.info(f"Intent prediction: {ints}")

        if not ints or float(ints[0]['probability']) < 0.85:
            logger.info("Using LLaMA fallback")
            gpt_reply = get_llama_response(msg)
            extracted_remedies = extract_remedies_from_text(gpt_reply)
            return gpt_reply, extracted_remedies

        response, remedies = getResponse(ints, intents)
        return response, remedies
    except Exception as e:
        logger.error(f"Error in chatbot_response: {str(e)}")
        return "I apologize, but I'm having trouble understanding. Could you rephrase that?", ["unknown"]

@app.route('/update_chat_title/<int:session_id>', methods=['POST'])
@login_required
def update_chat_title(session_id):
    try:
        data = request.get_json()
        new_title = data.get('title')
        
        if not new_title:
            return jsonify({'success': False, 'error': 'No title provided'}), 400
            
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
        session.title = new_title
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating chat title: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    session_id = data.get('session_id')
    message_id = data.get('message_id')
    feedback = data.get('feedback')
    rating = data.get('rating')
    
    try:
        fb = Feedback(
            session_id=session_id,
            message_id=message_id,
            feedback=feedback,
            rating=rating
        )
        db.session.add(fb)
        db.session.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Add feedback table to database
def init_db():
    # Use SQLAlchemy to create missing tables. This will create the `feedback` table
    # based on the `Feedback` model defined in `models.py`.
    with app.app_context():
        db.create_all()

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    with app.app_context():
        db.create_all()
    app.run(debug=True)

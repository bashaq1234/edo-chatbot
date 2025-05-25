import streamlit as st
import json
import random
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download NLTK data (only needed once)
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load data and model files from current directory
try:
    with open("intents.json", encoding="utf-8") as f:
        intents = json.load(f)

    model = load_model("chatbot_model.h5")
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.isalnum()]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    threshold = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I couldn't understand. Please try again."
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Streamlit UI setup
st.set_page_config(page_title="AskEdo1.o", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
        body { background-color: #f5fff5; }
        .header-title {
            font-size: 32px;
            font-weight: bold;
            color: #0c4b14;
        }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 8])
with col1:
    st.image("logoedsg-removebg-preview.png", width=60)
with col2:
    st.markdown('<div class="header-title">AskEdo1.o</div>', unsafe_allow_html=True)

user_input = st.text_input("Ask a question about Edo State Government:")

if st.button("Get Response"):
    if user_input:
        try:
            ints = predict_class(user_input)
            res = get_response(ints, intents)
            st.success(f"🤖 {res}")
        except Exception as e:
            st.error(f"Error during response generation: {e}")
    else:
        st.warning("Please enter your question first.")

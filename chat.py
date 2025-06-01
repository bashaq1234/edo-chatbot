import json
import random
import pickle
import numpy as np
import nltk
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

try:
    print("✅ Starting chatbot...")
    nltk.download('punkt')
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()

    print("✅ Loading model and files...")
    model = load_model("streamlit_app/chatbot_model.h5")
    intents = json.load(open("streamlit_app/intents.json", encoding="utf-8"))
    words = pickle.load(open("streamlit_app/words.pkl", "rb"))
    classes = pickle.load(open("streamlit_app/classes.pkl", "rb"))
    print("✅ Model and data loaded successfully.\n")

except Exception as e:
    print("❌ Error occurred:", e)
    exit()

# Clean up user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.isalnum()]
    return sentence_words

# Convert input to Bag of Words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict class
def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    threshold = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Get response
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I couldn't find that office. Please try again."
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Chat loop
print("🚀 Edo State Address Chatbot is ready! Type 'quit' to exit.\n")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("👋 Goodbye!")
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("AskEdo:", res)

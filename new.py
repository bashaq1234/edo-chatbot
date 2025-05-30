import nltk
nltk.download("punkt", download_dir="./nltk_data")
nltk.download("wordnet", download_dir="./nltk_data")
nltk.download("omw-1.4", download_dir="./nltk_data")
import json
import random
import numpy as np
import nltk
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import pickle

# === Ensure NLTK data is downloaded ===
nltk.download('punkt')
nltk.download('wordnet')

# === Prepare folders ===
data_dir = "streamlit_app"
intents_file = os.path.join(data_dir, "intents.json")
model_file = os.path.join(data_dir, "chatbot_model.h5")
words_file = os.path.join(data_dir, "words.pkl")
classes_file = os.path.join(data_dir, "classes.pkl")

# === NLP Setup ===
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []

# === Load intents.json ===
with open(intents_file, encoding='utf-8') as file:
    intents = json.load(file)

# === Tokenization and Lemmatization ===
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# === Clean up words and sort ===
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w.isalnum()]))
classes = sorted(set(classes))

# === Create Training Data ===
training = []
output_empty = [0] * len(classes)

for doc in documents:
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = output_empty.copy()
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# === Shuffle & Convert to numpy arrays ===
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# === Sanity check ===
if len(train_x) == 0 or len(train_y) == 0:
    print("❌ Error: No training data found.")
    exit()

# === Model Architecture ===
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# === Train Model ===
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# === Save Model and Data ===
model.save(model_file)
pickle.dump(words, open(words_file, "wb"))
pickle.dump(classes, open(classes_file, "wb"))

print("✅ Model training complete. Files saved to streamlit_app/")
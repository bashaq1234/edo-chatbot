import os
import json
import random
import numpy as np
import nltk
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import pickle

# === Setup NLTK data path and download required resources ===
nltk_data_dir = "./nltk_data"
nltk.data.path.append(nltk_data_dir)

for resource in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# === Define file paths ===
data_dir = "streamlit_app"
intents_file = os.path.join(data_dir, "intents.json")
model_file = os.path.join(data_dir, "chatbot_model.h5")
words_file = os.path.join(data_dir, "words.pkl")
classes_file = os.path.join(data_dir, "classes.pkl")

# === Initialize NLP tools ===
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []

# === Load intents JSON ===
with open(intents_file, encoding="utf-8") as file:
    intents = json.load(file)

# === Tokenize patterns and build documents/classes lists ===
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent["tag"]))
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# === Clean and sort words and classes ===
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w.isalnum()]))
classes = sorted(set(classes))

# === Create training data ===
training = []
output_empty = [0] * len(classes)

for doc in documents:
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = output_empty.copy()
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# === Shuffle and convert to numpy arrays ===
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# === Sanity check ===
if len(train_x) == 0 or len(train_y) == 0:
    print("❌ Error: No training data found.")
    exit()

# === Build the model ===
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(train_y[0]), activation="softmax"),
])

# === Compile the model ===
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# === Train the model ===
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# === Save model and data ===
model.save(model_file)
model.save("chatbot_model_v2.h5", include_optimizer=False)
pickle.dump(words, open(words_file, "wb"))
pickle.dump(classes, open(classes_file, "wb"))

print("✅ Model training complete. Files saved to streamlit_app/")

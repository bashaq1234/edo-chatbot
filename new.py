import json
import random
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []

# Load intents
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Tokenize & lemmatize patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern.strip())
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Preprocess words
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w.isalnum()]))
classes = sorted(set(classes))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    bag = [1 if w in pattern_words else 0 for w in words]

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Debugging info
print(f"Training samples: {train_x.shape}")
print(f"Output labels: {train_y.shape}")

# Sanity check
if len(train_x) == 0 or len(train_y) == 0:
    print("❌ Error: Training data is empty. Please check your intents.json patterns.")
    exit()

# Build and compile model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model and assets
model.save("chatbot_model.h5")
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

print("✅ Model training complete. Files saved successfully.")

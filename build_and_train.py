# build_and_train.py
import json
import os
import pickle
import random

import numpy as np
import nltk

from nltk.stem import PorterStemmer
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Configuration
# ----------------------------
DATA_FILE = "data.json"
MODEL_FILE = "model.h5"
WORDS_PKL = "words.pkl"     # list of unique stemmed words
CLASSES_PKL = "classes.pkl" # list of tags
DOCUMENTS_PKL = "documents.pkl" # list of (tokenized_words, tag)
TEXTS_PKL = "texts.pkl"     # legacy name used by your app (we'll store documents)
LABELS_PKL = "labels.pkl"   # label encoder object (or list of labels)
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")  # local nltk data dir

# Create nltk data folder and download required corpora if not present
if not os.path.exists(NLTK_DIR):
    os.makedirs(NLTK_DIR, exist_ok=True)

nltk.data.path.append(NLTK_DIR)

# Download required NLTK resources (will save to nltk_data dir)
for pkg in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        print(f"Downloading NLTK package: {pkg}")
        nltk.download(pkg, download_dir=NLTK_DIR)

# ----------------------------
# Load data.json
# ----------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Create it before running this script.")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ----------------------------
# Preprocess
# ----------------------------
stemmer = PorterStemmer()

words = []
classes = []
documents = []  # (tokenized_stemmed_words, tag)

for intent in data.get("intents", []):
    tag = intent.get("tag")
    patterns = intent.get("patterns", [])
    for pat in patterns:
        # tokenize
        tokens = nltk.word_tokenize(pat)
        # lower and stem
        st_tokens = [stemmer.stem(w.lower()) for w in tokens if w.isalnum()]
        words.extend(st_tokens)
        documents.append((st_tokens, tag))
    if tag not in classes:
        classes.append(tag)

# remove duplicates & sort
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"# words: {len(words)}, # classes: {len(classes)}, # documents: {len(documents)}")

# Save words, classes, documents, texts & labels pickles
with open(WORDS_PKL, "wb") as f:
    pickle.dump(words, f)
with open(CLASSES_PKL, "wb") as f:
    pickle.dump(classes, f)
with open(DOCUMENTS_PKL, "wb") as f:
    pickle.dump(documents, f)

# For compatibility with your app: texts.pkl = documents, labels.pkl = classes
with open(TEXTS_PKL, "wb") as f:
    pickle.dump(documents, f)
with open(LABELS_PKL, "wb") as f:
    pickle.dump(classes, f)

# ----------------------------
# Create training data (bag-of-words)
# ----------------------------
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# ----------------------------
# Build model (simple feed-forward)
# ----------------------------
input_shape = len(train_x[0])
output_shape = len(train_y[0])

model = Sequential()
model.add(Dense(128, input_shape=(input_shape,), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# Train
print("Training model...")
history = model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=1)

# Save model & data
model.save(MODEL_FILE)
print(f"Saved model to {MODEL_FILE}")

# Also save words/classes for runtime
with open("words.pkl", "wb") as f:
    pickle.dump(words, f)
with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

print("Saved words.pkl, classes.pkl, texts.pkl, labels.pkl, documents.pkl")

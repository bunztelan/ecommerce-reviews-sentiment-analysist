import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords  # Import stopwords library
from tensorflow.keras import callbacks as keras
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.optimizers import Adam  
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import chardet
import encodings
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
print(tf.__version__)
def clean_text(text):
    import re
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip().lower()  # Trim leading/trailing spaces and lowercase
    return text
print("Load Data...")
# Load data
reviews = pd.read_csv('lazada_reviews/20191002-reviews.csv')
print("Data Loaded...")
# Preprocess data
print("Preprocessing...")
print("Preprocessing - drop original rating...")
reviews = reviews.drop('originalRating', axis=1)

print("Preprocessing - Fill empty value...")
reviews['reviewContent'] = reviews['reviewContent'].fillna('unknown')
reviews['reviewTitle'] = reviews['reviewTitle'].fillna('unknown')

print("Preprocessing - Cleaning Review Text...")
reviews['reviewContent'] = reviews['reviewContent'].apply(clean_text)
reviews['reviewTitle'] = reviews['reviewTitle'].apply(clean_text)

# Define sentiment
print("Preprocessing - defining sentiment...")
reviews['sentiment'] = reviews['rating'].apply(lambda x: 'negative' if x < 4 else 'positive')

# Tokenization and sequence padding
X = reviews['reviewContent']
y = reviews['sentiment'].values
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences, maxlen=400, padding='post', truncating='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42, stratify=y)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
# Model definition with Regularization
print(" Defining Model...")
# Define the LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128),  # Use **kwargs for max_length
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping (optional)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Train the model with class weights
print("Training Model with Class Weights...")
print(X_train[:5])  # Check the first few training sequences
print(y_train_encoded[:5]) 
model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_test, y_test_encoded), class_weight=class_weight_dict, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save the model
model.save('sentiment_analysis_model_class_weights.h5')

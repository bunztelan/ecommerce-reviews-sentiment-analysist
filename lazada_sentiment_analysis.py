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


def clean_text(text):
    import re
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip().lower()  # Trim leading/trailing spaces and lowercase
    return text

# Load data
reviews = pd.read_csv('lazada_reviews/20191002-reviews.csv')

# Preprocess data
reviews = reviews.drop('originalRating', axis=1)

reviews['reviewContent'] = reviews['reviewContent'].fillna('unknown')
reviews['reviewTitle'] = reviews['reviewTitle'].fillna('unknown')

reviews['reviewContent'] = reviews['reviewContent'].apply(clean_text)
reviews['reviewTitle'] = reviews['reviewTitle'].apply(clean_text)

# Define sentiment
reviews['sentiment'] = reviews['rating'].apply(lambda x: 'negative' if x < 4 else 'positive')

# Feature Engineering (optional)
reviews['word_count'] = reviews['reviewContent'].apply(lambda x: len(x.split()))
# Define sentiment based on rating
tfidf_vectorizer_title = TfidfVectorizer(max_features=2500)
X_title_tfidf = tfidf_vectorizer_title.fit_transform(reviews['reviewTitle']).toarray()

tfidf_vectorizer_content = TfidfVectorizer(max_features=5000)
X_content_tfidf = tfidf_vectorizer_content.fit_transform(reviews['reviewContent']).toarray()

# Concatenate the features
X_combined = np.concatenate((X_title_tfidf, X_content_tfidf), axis=1)
# Separate features (text columns) and target (sentiment)
X = reviews[['reviewTitle', 'reviewContent']]
y = reviews['sentiment']

# Apply SMOTE for oversampling the minority class
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X_combined, y)

# Combine features and target after resampling
reviews_resampled = pd.DataFrame({'reviewTitle': X_resampled[:, 0], 'reviewContent': X_resampled[:, 1], 'sentiment': y_resampled})

# Split data
X_train, X_test, y_train, y_test = train_test_split(reviews_resampled['reviewContent'], reviews_resampled['sentiment'], test_size=0.2, random_state=42, stratify=reviews_resampled['rating'])  # Stratify for balanced classes

# Tokenization
max_length = 400  # Adjust based on analysis
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

padded_sequences_train = pad_sequences(sequences_train, maxlen=max_length, truncating='post', padding='post')
padded_sequences_test = pad_sequences(sequences_test, maxlen=max_length, truncating='post', padding='post')

# Model definition with Regularization
model = Sequential([
  Embedding(input_dim=5000, output_dim=128, input_length=max_length),
  Bidirectional(LSTM(64, return_sequences=True)),
  Dropout(0.4),  # Increased dropout
  LSTM(32),
  Dropout(0.5),  # Increased dropout
  Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping (optional)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Train model
model.fit(padded_sequences_train, y_train, epochs=10, validation_data=(padded_sequences_test, y_test), callbacks=[early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(padded_sequences_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model
model.save('sentiment_analysis_model_v2.h5')

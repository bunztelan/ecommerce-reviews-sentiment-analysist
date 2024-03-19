import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords  # Import stopwords library
from tensorflow.keras import callbacks as keras
from tensorflow.keras.optimizers import Adam  
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def clean_text(text):
    import re
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip().lower()  # Trim leading/trailing spaces and lowercase

    text = stemmer.stem(text)
    return text
# Load data
reviews = pd.read_csv('lazada_reviews/20191002-reviews.csv')

# Preprocess data
reviews = reviews.drop('originalRating', axis=1)
reviews = reviews.dropna(subset=['reviewContent'])  # Remove rows where reviewContent is null
reviews = reviews[reviews['reviewContent'].str.strip().astype(bool)]  # Remove rows with empty strings
reviews['reviewContent'] = reviews['reviewContent'].apply(clean_text)

# Define sentiment
reviews['sentiment'] = reviews['rating'].apply(lambda x: 'negative' if x < 4 else 'positive')

# Tokenization and sequence padding
X = reviews['reviewContent']
y = reviews['sentiment'].values
tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42, stratify=y)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Define the LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Early stopping (optional)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# Train the model with class weights
model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_test, y_test_encoded), class_weight=class_weight_dict, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save the model
model.save('my_model.keras')

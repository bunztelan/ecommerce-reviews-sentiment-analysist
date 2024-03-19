from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import json

app = Flask(__name__)

# Load your trained model
model = load_model('my_model.keras')

# Assuming you have your tokenizer saved, load it here. For demonstration, we will initialize it afresh.
# In a real scenario, you would save and load your tokenizer to ensure consistency.
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')

# Test conversion
def clean_text(text):
    import re
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip().lower()  # Trim leading/trailing spaces and lowercase
    return text

@app.route('/predict', methods=['POST'])
def predict():
   
    data = request.get_json(force=True)
    review = clean_text(data['review'])
    print(review)
    tokenizer.fit_on_texts(review)
    if not review:
        return jsonify({'error': 'Review text is missing.'}), 400
    
    # Tokenize and pad the review
    tokenizer.fit_on_texts([review])
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)
    print(prediction)
    sentiment = 'positive' if prediction[0][0] > 0.7 else 'negative'
    
    # Return the result
    return jsonify({'review': review, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(port=8002, debug=True)

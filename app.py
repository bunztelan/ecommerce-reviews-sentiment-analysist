from flask import Flask, request, jsonify, current_app
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import json

app = Flask(__name__)

# Load your trained model
model = load_model('sentiment_analysis_model.h5')

# Assuming you have your tokenizer saved, load it here. For demonstration, we will initialize it afresh.
# In a real scenario, you would save and load your tokenizer to ensure consistency.
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')

# Test conversion


@app.route('/predict', methods=['POST'])
def predict():
   
    data = request.get_json(force=True)
    review = data['review']
    print(review)
    tokenizer.fit_on_texts(review)
    if not review:
        return jsonify({'error': 'Review text is missing.'}), 400
    
    # Tokenize and pad the review
    sequences = tokenizer.texts_to_sequences([review])
    sequences = [sublist for sublist in sequences if None not in sublist]
    padded = pad_sequences(sequences, maxlen=400)  # Ensure this matches the maxlen used during training
    
    # Make prediction
    # print(padded)
    prediction = model.predict(padded)
    print(prediction)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    
    # Return the result
    return jsonify({'review': review, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(port=8000, debug=True)

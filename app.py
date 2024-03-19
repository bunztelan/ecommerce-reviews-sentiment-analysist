from flask import Flask, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

model = tf.keras.models.load_model('my_model.keras')
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
@app.route('/')
def home():
    return '''
        <form action="/predict" method="post">
            <textarea name="text" rows="10" cols="30"></textarea><br>
            <input type="submit" value="Predict Sentiment">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    preprocessed_text = preprocess_input(input_text)
    prediction = model.predict(preprocessed_text)
    readable_prediction = format_prediction(prediction)
    return f'Prediction: {readable_prediction}'

def clean_text(text):
    import re
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip().lower()  # Trim leading/trailing spaces and lowercase
    return text

def preprocess_input(text):
    text = clean_text(text)
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')
    return padded_sequences

def format_prediction(prediction):
    prediction = model.predict(prediction)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment

if __name__ == '__main__':
    app.run(debug=True)

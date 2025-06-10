from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import streamlit as st

#loading model and tokenizer
def load_spam_model():
    model = load_model('models/spam_detector.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

#extracting text-based features 
def extract_features(text):
    features = {}
    text_lower = text.lower()
    suspicious_words = ['urgent', 'verify', 'suspend', 'account', 'security', 'update',
                        'confirm', 'click here', 'login', 'password', 'suspended',
                        'winner', 'free', 'cash', 'prize', 'call now', 'text to']
    features['suspicious_word_count'] = sum(1 for word in suspicious_words if word in text_lower)
    features['url_count'] = len(re.findall(r'http[s]?://', text))
    features['exclamation_count'] = text.count('!')
    return features

#predicting spam/phishing
def predict_spam(text, model, tokenizer):
    if not text.strip():
        return 0.0, {}
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded, verbose=0)[0][0]
    features = extract_features(text)
    return prediction, features


st.title('AI Spam/Phishing Detector')
option = st.selectbox(
    'What would you like to analyze?',
    ('Suspicious Email', 'Suspicious SMS')
)


model, tokenizer = load_spam_model() #loading the model


if option == "Suspicious Email": #if user selects to analyze a suspiscious email
    sender = st.text_input("Enter Sender: ")
    subject = st.text_input("Enter Subject Line: ")
    fb = st.text_area("Enter Body: ")

    full_message = f"{sender} {subject} {fb}"
    
    if st.button("Analyze"): #analyzing the email through the trained model
        prediction, features = predict_spam(full_message, model, tokenizer)
        st.subheader(f"Phishing Confidence: {prediction:.2f}") #displaying confidence score
        st.subheader("Extracted Features:") #displaying features
        st.write(features)

if option == "Suspicious SMS":
    sender = st.text_input("Enter Sender: ")
    fb = st.text_area("Enter Message: ")

    full_message = f"{sender} {fb}"
    
    if st.button("Analyze"): #analyzing the email through the trained model
        prediction, features = predict_spam(full_message, model, tokenizer)
        st.subheader(f"Spam Confidence: {prediction:.2f}") #displaying confidence score
        st.subheader("Extracted Features:") #displaying features
        st.write(features)
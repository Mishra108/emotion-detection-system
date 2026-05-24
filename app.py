import streamlit as st
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# ---------------- LOAD FILES ---------------- #

model = joblib.load("emotion_model.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")


# ---------------- NLP SETUP ---------------- #

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

negation_words = {'not', 'no', 'never'}

stop_words = stop_words - negation_words


# ---------------- REVERSE MAPPING ---------------- #

reverse_emotion_numbers = {
    0:'sadness',
    1:'joy',
    2:'love',
    3:'anger',
    4:'fear',
    5:'surprise'
}


# ---------------- CLEAN FUNCTION ---------------- #

def remove(txt):

    words = txt.split()

    cleaned = [word for word in words if word not in stop_words]

    return " ".join(cleaned)


# ---------------- PREDICTION FUNCTION ---------------- #

def predict_emotion(text):

    text = text.lower()

    text = remove(text)

    text = " ".join(
        [lemmatizer.lemmatize(word) for word in text.split()]
    )

    vectorized_text = tfidf.transform([text])

    prediction = model.predict(vectorized_text)

    return reverse_emotion_numbers[prediction[0]]


# ---------------- STREAMLIT UI ---------------- #

st.title("Emotion Detection System")

st.write("Enter text to detect emotion")


user_input = st.text_area("Enter Text")


if st.button("Predict Emotion"):

    result = predict_emotion(user_input)

    st.success(f"Predicted Emotion: {result}")
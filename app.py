import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model & vectorizer
model = pickle.load(open("models/spam_svm_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# Preprocessing function
nltk.download('stopwords')
ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# Streamlit UI
st.title("📧 Spam Email Classifier (SVM)")
st.write("Enter an email below to check if it's **Spam** or **Ham**")

user_input = st.text_area("✉️ Paste email text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        processed_text = preprocess_text(user_input)
        vector = tfidf.transform([processed_text]).toarray()
        prediction = model.predict(vector)[0]
        
        if prediction == 1:
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **HAM (Safe)**")
    else:
        st.warning("Please enter some text to classify.")

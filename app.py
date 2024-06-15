import streamlit as st
import pickle
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# Function to preprocess text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Perform stemming
    text = [ps.stem(i) for i in text]

    # Join the list back to a string
    return " ".join(text)


# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict using the pre-trained model
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model=joblib.load("xgboost_model.pkl")
cv=joblib.load("count_vectorizer.pkl")

ps = PorterStemmer()

def prediction(title,text,model,count_vectorizer):
    message=title+text
    data=re.sub("[^a-zA-Z]"," ",message)
    review=data.lower()
    words=data.split()
    sentence=[ps.stem(word) for word in words if word not in set(stopwords.words("english"))]
    stem_word=" ".join(sentence)
    count_vc=count_vectorizer.transform([stem_word]).toarray()
    pred=model.predict(count_vc)
    if pred[0]==1:
        return "Fake News"
    else:
        return "Real News"


# Creating Streamlit app 
st.set_page_config(page_title="Fake News Classifier", layout="wide") # Title
st.markdown("""
    <style>
    /* Tabs container */
    .stTabs {
        display: flex;
        justify-content: center;
    }

    </style>
    """, unsafe_allow_html=True)
html_temp = """
<div style="
    display: flex;
    justify-content: center;
    margin-top: 30px;
">
    <div style="
        background-color:#706C26;
        padding:20px;
        width:80%;
        height:100px;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius: 10px;
    ">
        <h2 style="color:white;text-align:center;">
            Fake News Classifier
        </h2>
    </div>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

title=st.text_input("Title",placeholder="Head Lines")
 # creating tabs

text=st.text_area(
    "Text",
    placeholder="Paste the news article here...",
    height=200)

grade=st.button("Submit")

if grade:
    st.write(prediction(title,text,model,cv))


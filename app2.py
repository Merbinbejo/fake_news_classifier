import streamlit as st
import joblib
import re
import torch
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification

model_path="./fake_classifier"

tokenizer=DistilBertTokenizerFast.from_pretrained(model_path)
model=DistilBertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def prediction(title, text, model, tokenizer):
    message = title + " " + text

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, pred_id = torch.max(probs, dim=1)

    pred_id = pred_id.item()
    confidence = confidence.item()

    # Safe label mapping
    label = model.config.id2label.get(pred_id, str(pred_id))

    if label in ["LABEL_1", "FAKE", "Fake"]:
        return "Fake News", confidence
    else:
        return "Real News", confidence


# Creating Streamlit app 
st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="wide"
)
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
    with st.spinner("🔍 Analyzing article..."):
        label, confidence = prediction(title, text, model, tokenizer)

    st.markdown("### 🧠 Prediction Result")

    if label == "Fake News":
        st.error(f"🚨 **Fake News Detected**\n\nConfidence: **{confidence:.2%}**")
    else:
        st.success(f"✅ **Real News**\n\nConfidence: **{confidence:.2%}**")


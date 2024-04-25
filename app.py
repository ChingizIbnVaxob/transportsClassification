import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import platform
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Title
st.title("Transportni klassifikatsiya qiluvchi model")

# Upload image
file = st.file_uploader("Rasm yuklash", type=['png', 'jpeg', 'img'])

# Load model
model = load_learner(str("transport_model.pkl"))

# Prediction
if file is not None:
    # Create PIL image
    img = PILImage.create(file)

    # Display uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Perform prediction
    pred, pred_id, probs = model.predict(img)

    # Display prediction result
    st.success(pred)

    #Display prediction accuracy
    st.info(f'Accuracy: {probs[pred_id]*100:.1f}')

    #display figure
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

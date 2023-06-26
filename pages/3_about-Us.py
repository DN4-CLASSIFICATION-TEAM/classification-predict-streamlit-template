import streamlit as st
from PIL import Image


st.title("About Us")
image = Image.open("./resources/imgs/dn4logo.png")
st.sidebar.image(image, width=200)

# About us code here
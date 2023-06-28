import streamlit as st
from matplotlib.animation import ImageMagickWriter
import streamlit as st
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import pandas as pd
from nltk.corpus import stopwords
import sys


st.write("Whether  you are a business looking to improve your marketing, a non-profit organisation trying to engage with your audience, or a government body interested in public opinion for policy making, Sense Solution is here to help.")
st.write("Sense Solutions is known for giving great insigts in data by developing advanced machine learning models. We specialize in analyzing social media data, particularly on platforms like Twitter. We believe that data holds the key to unlocking important insights and driving successful strategies.")
st.write("Our team team is filled with data enthusiastic individuals. Our team works closely with clients to understand their specific needs and provide ongoing support.")

st.subheader("The Team")
gha = Image.open("resources/imgs/Ghaalib.jpeg")
olwe = Image.open("resources/imgs/Olwethu.jpeg")
pet = Image.open("resources/imgs/Pertunia.jpeg")

col1 ,col2, col3 = st.columns(3)

with col1:
	st.image(olwe,use_column_width=False, clamp=False, width = 150, output_format="JPEG")
	st.write("Project Manager")
	st.write("Olwethu Magadla")
	st.write("olwethumagadla@sensesolutions.com")
with col2:
	st.image(pet, use_column_width=False, clamp=False, width=150, output_format="JPEG")
	st.write("Data Scientist")
	st.write("Pertunia Nhlapo")
	st.write("pertunianhlapo@sensesolutions.com")
with col3:
	st.image(gha, use_column_width=False, clamp=True, width=150, output_format="JPEG")
	st.write("Lead Data Scientist")
	st.write("Ghaalib van der Ross")
	st.write("ghaalibvanderross@sensesolutionse.com")

imi = Image.open("./resources/imgs/Imi.jpeg")
leh = Image.open("./resources/imgs/Lehlohonolo.jpeg")
ori = Image.open("./resources/imgs/Orifuna.jpeg")
col4, col5, col6 = st.columns(3)

with col4:
	st.image(imi, use_column_width=False, clamp=True, width=150, output_format="JPEG")
	st.write("Data Engineer")
	st.write("Ntokozo Bingwe")
	st.write("ntokozobingwe@sensesolutions.com")
with col5:
	st.image(leh, use_column_width=False, clamp=True, width=150, output_format="JPEG")
	st.write("Technical Lead")
	st.write("Lehlohonolo Radebe")
	st.write("lehlohonoloradebe@sensesolutions.com")
with col6:
	st.image(ori, use_column_width=False, clamp=True, width=150, output_format="JPEG")
	st.write("Data Engineer")
	st.write("Orifuna Nemusombori")
	st.write("orifunanemusombori@sensesolutions.com")

st.write("Discover the power of data with Sense Solutions- Your partner for unlocking insights from social media.")
st.write("Contact us today to learn more about our services and how we can help your organization thrive in digital age.")

st.write("Our website - sensesolution.com")

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

#st.info("General Information")
st.write("This exciting App can help your organisation make better decisions using Twitter. We understand the importance of knowing what people are thinking and want to make it easy to filter tweets. We use technology to study tweets classify them.")

# You can read a markdown file from supporting resources folder
st.subheader("We provide a simple dashboard and tools that make it easy for you to understand the data. You don't need to be an expert to make sense of it and make informed choices.")

twitter = Image.open("resources/imgs/twitter_logo.jpeg")
emojis = Image.open("resources/imgs/emojis.jpeg")
interaction = Image.open("resources/imgs/interaction.jpeg")
col10, col12, col11=st.columns(3)

with col10:
	st.image(twitter,use_column_width=False, clamp=False, width = 200, output_format="JPEG")
	#st.write("* Social media is websites and applications that enable users to create and share content or to participate in social networking. ")
	st.write("* Social media is forever growning, a way for people to express their view with the world and connect with others. We find social media helpful also in finding some useful information about products, news affecting our work or country.")
	st.write("* As the climate crisis intensifies and natural disasters become more frequent and powerful, scientists are increasingly turning to social media as a way to assess the damage and impact. ")
with col12:
	st.image(emojis,use_column_width=False, clamp=False, width = 200, output_format="JPEG")
	st.write("We analyse a sentiments data that has that will inform us which class a tweet falls under. We have 4 different classe explained below.")
	st.write("* 2 News: the tweet links to factual news about climate change")
	st.write("* 1 Pro: the tweet supports the belief of man-made climate change")
	st.write("* 0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change")
	st.write("* -1 Anti: the tweet does not believe in man-made climate change Variable definitions")
	#st.write("*")
with col11:
	st.image(interaction,use_column_width=False, clamp=False, width = 200, output_format="JPEG")
	st.write("* We used a linear regresion model to describe the relationship between our sentiment classes and the tweets.")
	st.write("* Our model gives you important information to help you understand trends and make smarter choices.")
	st.write("* Our model can show you what specific groups of people or areas of the world are saying. This helps you create messages and campaigns that connect with your target audience.")
		

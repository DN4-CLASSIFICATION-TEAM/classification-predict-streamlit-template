"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
from PIL import Image
import joblib


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	image = Image.open("./resources/imgs/dn4logo.png")
	st.image(image, caption="Image Caption")
    
	# image = Image.open("./resources/imgs/chatbot.png")
	# st.image(image, caption="Image Caption")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()

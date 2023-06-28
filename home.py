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
# from pages import home, modeling, aboutUsupdate, aboutTheApp

# Create a dictionary to map page names to their respective modules
pages = {
    "Home": home,
    "About the App":aboutTheApp,
    "Modeling": modeling,
    "about Us": aboutUsupdate,
}

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    image = Image.open("./resources/imgs/dn4logo.png")
    st.sidebar.image(image, width=200)
    st.title("Welcome to Sense Solutions!")
    st.markdown("<h2 style='text-align: center;'>Let's Classify</h2>", unsafe_allow_html=True)
    wel = Image.open("resources/imgs/chatBot.png")
    st.image(wel,use_column_width=True, clamp=False, width = 200, output_format="PNG")
    st.markdown("#### We are delighted to present our state-of-the-art web application designed for tweet classification. With our innovative technology, you can effortlessly analyze and categorize tweets based on their sentiments, providing you with valuable insights into public opinions")
    st.write("Stand Out from the Competition: By using our sentiment analysis model, you'll have an advantage over others because you can respond quickly to public sentiment, address concerns, and align your efforts with what people are saying.")
    st.write("Support for Decision Making: Our model acts as a reliable tool to help you make decisions based on data. You can use the insights to guide your marketing strategies, improve your products, and handle any crises that may arise.")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()

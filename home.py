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
# from pages import home, modeling, aboutUsupdate

# Create a dictionary to map page names to their respective modules
# pages = {
#     "Home": home,
#     "Modeling": modeling,
#     "about Us": aboutUsupdate,
# }

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    # Add a sidebar to navigate between pages
    # selection = st.sidebar.radio("Navigation", list(pages.keys()))

    # Render the selected page
    # page = pages[selection]
    # page.render()
    image = Image.open("./resources/imgs/dn4logo.png")
    st.sidebar.image(image, width=200)
    st.title("Welcome to Sense Solutions!")

    # Create a two-column layout
    col1, col2 = st.columns(2)
    col1.image(image, caption="Image Caption")
    col2.markdown("#### We are delighted to present our state-of-the-art web application designed for tweet classification. With our innovative technology, you can effortlessly analyze and categorize tweets based on their sentiments, providing you with valuable insights into public opinions")
    # col2.write("We are delighted to present o")
    st.info("Our tweet classification web app utilizes advanced machine learning algorithms to accurately determine the sentiment of each tweet. Whether it's positive, negative, or neutral, our model can swiftly classify the sentiment, allowing you to gain a comprehensive understanding of public sentiment on various topics.")    
    image = Image.open("./resources/imgs/chatbot.png")
    st.image(image, caption="")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()

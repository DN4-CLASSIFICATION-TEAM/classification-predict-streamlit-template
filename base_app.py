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
from matplotlib.animation import ImageMagickWriter
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Function to load the selected classifier
def load_classifier(classifier_name):
    classifier_file = f"resources/{classifier_name}.pkl"
    classifier = joblib.load(open(classifier_file, "rb"))
    return classifier

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
 
 # Set page configuration
	st.set_page_config(page_title="DN4 Tweet Classifer", page_icon="🐦", layout="wide")

	st.markdown(
        """
        <style>
        body {
            background-color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set button background color to blue
	st.markdown(
		"""
		<style>
		.css-2trqyj:focus, .css-2trqyj:hover {
			background-color: blue;
		}
		</style>
		""",
		unsafe_allow_html=True
	)

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer prod by Sense Solutions")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Prediction", "Information", "About Us"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
			# will write the df to the page
			st.write(raw[['sentiment', 'message']])

		# Create a filter for sentiment categories
		sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Negative", "Neutral", "Pro News"])

        # Filter the data based on selected sentiment
		if sentiment_filter == "All":
			filtered_data = raw
		elif sentiment_filter == "Positive":
			filtered_data = raw[raw['sentiment'] == 1]
		elif sentiment_filter == "Negative":
			filtered_data = raw[raw['sentiment'] == -1]
		elif sentiment_filter == "Neutral":
			filtered_data = raw[raw['sentiment'] == 0]
		elif sentiment_filter == "Pro News":
			filtered_data = raw[raw['sentiment'] == 2]

		# filtered_data.drop["tweetid"]
        # Display the first 20 rows of the filtered data in a table
		st.table(filtered_data.head(20))

	# Building out the predication page
	# Building out the prediction page
	if selection == "Prediction":
		st.info("Prediction with ML Models")

		# Select classifier
		classifier_name = st.selectbox("Select Classifier", ["MNB", "Logistic_regression", "DT", "SVC"])

		# Load classifier
		classifier = load_classifier(classifier_name)

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text", "Type Here")

		if st.button("Classify"):
			# Transform user input
			vect_text = tweet_cv.transform([tweet_text]).toarray()

			# Make prediction using the selected classifier
			prediction = classifier.predict(vect_text)

			# When model has successfully run, print prediction
			st.success("Text Categorized as: {}".format(prediction))

			# Display an image
			image = ImageMagickWriter.open("./resources/imgs/EDSA_logo.png")
			st.image(image, caption="Image Caption", use_column_width=True)



	# Building out the predication page
	if selection == "About Us":
		st.text("Who are we?")

	# Building out the predication page
	if selection == "Home":
		st.info("This exciting product we've created at Sense Solutions can help your organization make better decisions using Twitter."
          + "We understand how important it is to know what people are thinking in today's online world."
          + "Twitter is a popular platform where people share their opinions and feelings. Our special model,"
          + "called the Twitter Sentiment Analysis Model, uses smart technology to study tweets and figure out if they are positive," 
          + "negative, or neutral.")
		# Benefits and Features
		st.header("Benefits and Features")
		st.markdown(
			"""
			- Valuable Insights: Our model gives you important information about what people think and feel on Twitter. This can help you understand trends and make smarter choices.
			- Real-time Updates: You'll get updates in real-time, which means you'll always be up to date with the latest opinions about various topics, including climate change.
			- Understanding Your Audience: Our model can show you what specific groups of people or areas of the world are saying. This helps you create messages and campaigns that connect with your target audience.
			- Stand Out from the Competition: By using our sentiment analysis model, you'll have an advantage over others because you can respond quickly to public sentiment, address concerns, and align your efforts with what people are saying.
			- Support for Decision Making: Our model acts as a reliable tool to help you make decisions based on data. You can use the insights to guide your marketing strategies, improve your products, and handle any crises that may arise.
			- Easy to Use: We provide a simple dashboard and tools that make it easy for you to understand the data. You don't need to be an expert to make sense of it and make informed choices.
			"""
		)
		st.text("Problem Statement")
		st.info("Many companies are built around lessening one’s environmental impact or carbon footprint."
				+ "They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals."
				+ "They would like to determine how people perceive climate change and whether or not they believe it is a real threat."
				+ "This would add to their market research efforts in gauging how their product/service may be received.")


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()

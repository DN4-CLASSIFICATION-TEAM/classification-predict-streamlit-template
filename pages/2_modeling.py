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
sys.path.append("./data_cleaner")
from data_cleaner import remove_noise


# Data dependencies
import pandas as pd

# TODO:
# Use one model to predict the sentiment of a tweet

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


# Company logo
image = Image.open("./resources/imgs/dn4logo.png")
st.sidebar.image(image, width=200)

# Creating sidebar with selection box -
# you can create multiple pages this way
options = ["Prediction", "Information", "Graphs"]
selection = st.sidebar.selectbox("Menu", options)


# Define the labels for different sentiments
sentiment_labels = {
    1: "Positive",
    -1: "Negative",
    0: "Neutral",
    2: "Pro News"
}

def clean_data(text):
    cleaned_text = remove_noise(text)
    return cleaned_text

raw['message'] = raw['message'].apply(clean_data)
raw = raw.drop("tweetid", axis=1)

# Building out the "Information" page
if selection == "Information":
    st.title("Insights")
    st.subheader(
        "Discover the power of data with Sense Solutions - Your partner for unlocking insights from social media")
    # Count the number of occurrences for each sentiment
    sentiment_counts = raw["sentiment"].value_counts()

    # Plot a doughnut chart
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=[sentiment_labels[sentiment]
    for sentiment in sentiment_counts.index], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title("Sentiment Distribution")

    # Display the doughnut chart
    st.pyplot(fig)

    if st.button("Generate Word Clouds"):
        with st.spinner("Generating Word Clouds..."):
            # Generate word cloud for each sentiment category
            sentiment_categories = raw["sentiment"].unique()
            num_columns = 2  # Number of columns in the subplot grid
            # Number of rows in the subplot grid
            num_rows = (len(sentiment_categories) + 1) // num_columns

            fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 8))

            for i, sentiment in enumerate(sentiment_categories):
                tweets = raw.loc[raw["sentiment"] == sentiment, "message"]
                wordcloud = WordCloud(
                    width=400, height=400, background_color="white").generate(" ".join(tweets))
                ax = axs[i // num_columns, i % num_columns]
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.set_title(sentiment_labels[sentiment])
                ax.axis("off")

            # If there are extra subplots, remove them
            if len(sentiment_categories) < num_rows * num_columns:
                for j in range(len(sentiment_categories), num_rows * num_columns):
                    fig.delaxes(axs[j // num_columns, j % num_columns])

            fig.suptitle("Word Clouds for Different Sentiments", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            # Display the combined word clouds
            st.pyplot(fig)

    # Create a filter for sentiment categories
    sentiment_filter = st.selectbox(
        "Filter by Sentiment", ["All"] + list(sentiment_labels.values()))

    # Filter the data based on selected sentiment
    if sentiment_filter == "All":
        filtered_data = raw
    else:
        sentiment_value = next(
            key for key, value in sentiment_labels.items() if value == sentiment_filter)
        filtered_data = raw[raw['sentiment'] == sentiment_value]

    # Display the first 20 rows of the filtered data in a table
    st.table(filtered_data.head(5))

# Building out the predication page
if selection == "Prediction":
    st.title("Tweet Classifer prod by Sense Solutions")
    st.subheader("Climate change tweet classification")
    st.info("Prediction with Machne Learning Models")

    # Select classifier
    classifier_name = st.selectbox(
        "Select Classifier", ["Logistic_regression", "MNB", "DT", "SVC"])

    # Load classifier
    classifier = load_classifier(classifier_name)

    # Creating a text box for user input
    tweet_text = st.text_area("Enter Text", "Type Here")

    if st.button("Classify"):
        with st.spinner("Classifying Tweet..."):
            # Transform user input
            vect_text = tweet_cv.transform([tweet_text]).toarray()

            # Make prediction using the selected classifier
            prediction = classifier.predict(vect_text)[0]

            # When model has successfully run, print prediction
            st.success("Text Categorized as a: {} sentiment".format(
                sentiment_labels[prediction]))

            # Generate word cloud for the entered text
            wordcloud = WordCloud().generate(tweet_text)

            # Display the word cloud
            with st.pyplot():
                plt.figure(figsize=(8, 8))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title("Word Cloud for Entered Text")
                st.pyplot()
                plt.show()

            # Display an image
            # image = Image.open("./resources/imgs/streamlit.png")
            # st.image(image, caption="Image Caption")

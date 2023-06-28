from data_cleaner import remove_noise
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import ImageMagickWriter
import numpy as np
import joblib
import os
import nltk
from PIL import Image
from wordcloud import WordCloud
import re
import pandas as pd
from nltk.corpus import stopwords
import sys
sys.path.append("./data_cleaner")
from comet_ml import Experiment

# Create a new Comet experiment
# experiment = Experiment(api_key="O2DQXkha3pCGKtPdWVSve0aKf", project_name="Tweet Classification", workspace="DN4")

# Log an example metric
# experiment.log_metric("accuracy", 0.85)

# Log example parameters
# experiment.log_parameter("learning_rate", 0.001)
# experiment.log_parameter("batch_size", 32)

# Train our model and log relevant information
# TODO:

# Log model weights
# experiment.log_model_weights("model", model)

# End the experiment
# experiment.end()

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Function to load a selected classifier
def load_classifier(classifier_name):
    classifier_file = f"resources/{classifier_name}.pkl"
    classifier = joblib.load(open(classifier_file, "rb"))
    return classifier


# Load default classifier
# classifier = load_classifier("logistic_regression")


# Company logo
image = Image.open("./resources/imgs/dn4logo.png")
st.sidebar.image(image, caption="", width=200)

# Creating sidebar with selection box
options = ["Prediction", "Information"]
selection = st.sidebar.selectbox("Menu", options)

# plot pie chart for sentiment split of retweets
def retweet_pie(df):
    df['sentiment'].value_counts().plot(kind='pie', autopct='%.2f')
    plt.title("Sentiment Split of Re-Tweets")
    return plt.show() # display the plot

# Generate word clouds
def word_cloud(df, num, column):
    handle_freq2 = nltk.FreqDist(np.hstack(df[df['sentiment'] == 2][column]))
    handle_freq1 = nltk.FreqDist(np.hstack(df[df['sentiment'] == 1][column]))
    handle_freq0 = nltk.FreqDist(np.hstack(df[df['sentiment'] == 0][column]))
    handle_freqneg1 = nltk.FreqDist(
        np.hstack(df[df['sentiment'] == -1][column]))

    top_handle2 = dict(handle_freq2.most_common(num))
    top_handle1 = dict(handle_freq1.most_common(num))
    top_handle0 = dict(handle_freq0.most_common(num))
    top_handleneg1 = dict(handle_freqneg1.most_common(num))

    wordcloud2 = WordCloud().generate_from_frequencies(top_handle2)
    wordcloud1 = WordCloud().generate_from_frequencies(top_handle1)
    wordcloud0 = WordCloud().generate_from_frequencies(top_handle0)
    wordcloudneg1 = WordCloud().generate_from_frequencies(top_handleneg1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    axs[0, 0].imshow(wordcloud2, interpolation="bilinear")
    axs[0, 0].set_title("Positive Sentiment")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(wordcloud1, interpolation="bilinear")
    axs[0, 1].set_title("Neutral Sentiment")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(wordcloud0, interpolation="bilinear")
    axs[1, 0].set_title("Negative Sentiment")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(wordcloudneg1, interpolation="bilinear")
    axs[1, 1].set_title("News")
    axs[1, 1].axis("off")

    plt.tight_layout()
    return plt.show() # display the plot



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

# Function to clean and classify tweets
def clean_and_classify_tweets(text):
    # Convert the input to a string
    text = str(text)

    # Clean the text data using your data cleaning function
    cleaned_data = clean_data(text)

    # Vectorize the cleaned data using your trained vectorizer
    vectorized_data = tweet_cv.transform([cleaned_data]).toarray()

    # Make predictions using the loaded classifier
    predictions = classifier.predict(vectorized_data)

    return predictions[0]



# clean the data, drop the tweetid column and create a new column with the cleaned tweets
raw['clean_tweet'] = raw['message'].apply(clean_and_classify_tweets)
raw = raw.drop("tweetid", axis=1)

# Declare df_train globally
df_train = None

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


def handle(text):
    x = re.findall(r'(@\w+)', text)
    return x


raw['Handle'] = raw['message'].apply(handle)

# Function to show data insights


def show_data_insights(df):
    st.subheader("Data Insights")

    # Display summary statistics
    st.write("Summary Statistics:")
    st.write(df.describe())

    # Display data visualization (e.g., histogram, bar chart, etc.)
    st.write("Data Visualization:")
    # Add your visualization code here using matplotlib or any other library


# Function to get most retweeted usernames
def get_most_retweeted_username(df, top_n=1):
    # Extract usernames from retweets with handles starting with '@'
    df['Username'] = df['message'].str.extract(r'RT\s+(@\w+):')

    # Count the occurrences of each username
    username_counts = df['Username'].value_counts()

    # Check if there are any retweets with usernames
    if not username_counts.empty:
        # Get the top usernames with the most retweets
        top_usernames = username_counts.head(top_n)

        # Create a pie chart to visualize the retweet counts
        plt.figure(figsize=(8, 8))
        plt.pie(top_usernames.values, labels=top_usernames.index, autopct='%.2f%%')
        plt.title(f'Top {top_n} Usernames with the Most Retweets')
        plt.axis('equal')

        # Display the chart
        st.subheader(f'Top {top_n} Usernames with the Most Retweets')
        st.pyplot(plt)
    else:
        st.write('No retweets with usernames found.')

# Find duplicate messages
df_retweet = raw[raw['message'].duplicated()].reset_index(drop=True)
# Building out the predication page
if selection == "Prediction":
    st.title("Tweet Classifer prod by Sense Solutions")
    st.subheader("Climate change tweet classification")
    st.subheader("Enter a tweet to be classified or upload your training data and test data to get accurate sentiment predictions")
    st.info("Prediction with Machne Learning Models")

    # Select classifier
    classifier_name = st.selectbox(
        "Select Classifier", ["Logistic_regression", "MNB", "DT", "SVC"])

    # Load classifier
    classifier = load_classifier(classifier_name)

     # Create a file uploader for test data
    uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

    # Check if a file is uploaded
    if uploaded_file is not None:
        with st.spinner("Uploading..."):
            # Read the uploaded CSV file
            df_test = pd.read_csv(uploaded_file)

            # Display the uploaded test data
            st.subheader("Uploaded Test Data")
            st.dataframe(df_test)

            # if st.button("Classify Your Data"):
            #     # Clean and classify the tweets
            #     predictions = clean_and_classify_tweets(df_test['message'])

            #     # Add the predictions as a new column in the DataFrame
            #     df_test['sentiment'] = predictions

            #     # Display the pie chart of sentiments
            #     st.subheader("Sentiment Split of Your Data")
            #     sentiment_counts = df_test['sentiment'].value_counts()
            #     st.plotly_chart(sentiment_counts.plot.pie(autopct="%.2f%%", figsize=(6, 6)))

            #     # Display the top words word clouds
            #     st.subheader("Top Words in Your Data")
            #     # word_cloud(df_test, 20, 'message')

            #     # Display the username with the most retweets
            #     st.subheader("Users with the Most Retweets")
            #     get_most_retweeted_username(df_test, top_n=20)

    col1, col2 = st.columns(2)
    
    if col1.button("Data insights"):
        if 'df_test' in locals():
            if st.button("Classify Your Data"):
                # Display the pie chart of sentiments
                st.subheader("Sentiment Split of Test Data")
                sentiment_counts = df_test['sentiment'].value_counts()
                st.plotly_chart(sentiment_counts.plot.pie(autopct="%.2f%%", figsize=(6, 6)))

                # Display the top words word clouds
                st.subheader("Top Words in Test Data")
                word_cloud(df_test, 20, 'message')

                # Display the username with the most retweets
                st.subheader("Users with the Most Retweets")
                get_most_retweeted_username(df_test, top_n=20)

    if col2.button("Get Retweets"):
        get_most_retweeted_username(raw, top_n=20)

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
            word_cloud(df_test, 20, 'message')


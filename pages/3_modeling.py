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
import seaborn as sns
sys.path.append("./data_cleaner")
# from comet_ml import Experiment
# nltk.download('stopwords')

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

# Function to load the vectorizer
@st.cache_data()  # Cache the vectorizer to avoid repeated loading
def load_vectorizer():
    vectorizer = joblib.load("resources/tfidfvect.pkl")
    return vectorizer

# Function to load the raw data
@st.cache_data  # Cache the raw data to avoid repeated loading
def load_raw_data():
    raw_data = pd.read_csv("resources/train.csv", nrows=1000) # Load 1000 rows of the raw data
    return raw_data

# Function to load a selected classifier
@st.cache_data()  # Cache the classifier to avoid repeated loading
def load_classifier(classifier_name):
    classifier_file = f"resources/{classifier_name}.pkl"
    classifier = joblib.load(open(classifier_file, "rb"))
    return classifier

# Load the vectorizer and raw data
tweet_cv = load_vectorizer()
raw = load_raw_data()


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
    df[column] = df[column].apply(clean_data)  # Clean the text column
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
# def clean_and_classify_tweets(df):
#     # Clean the text data using your data cleaning function
#     cleaned_data = df['message'].apply(clean_data)

#     # Vectorize the cleaned data using your trained vectorizer
#     vectorized_data = tweet_cv.transform(cleaned_data).toarray()

#     # Load the classifier
#     classifier = load_classifier("Logistic_regression")

#     # Make predictions using the loaded classifier
#     predictions = classifier.predict(vectorized_data)

#     return predictions


def handle(text):
    x = re.findall(r'(@\w+)', text)
    return x


# Function to display data insights
def show_data_insights(df):
    st.subheader("Data Insights")

    # Display summary statistics
    st.write("Summary Statistics:")
    st.write(df.describe())

    # Word clouds for sentiments
    st.write("Word Clouds for Sentiments:")
    sentiment_labels = {
        1: "Positive",
        -1: "Negative",
        0: "Neutral",
        2: "Pro News"
    }
    num_words = 10  # Number of words to display in the word cloud

    for sentiment, label in sentiment_labels.items():
        st.subheader(f"Word Cloud for {label} Sentiment:")
        filtered_data = df[df['sentiment'] == sentiment]
        text = " ".join(filtered_data['clean_tweet'])
        wordcloud = WordCloud(max_words=num_words).generate(text)

        plt.figure(figsize=(8, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(f"Top {num_words} words for {label} Sentiment")
        plt.axis("off")
        st.pyplot(plt)

    # Doughnut chart for sentiments
    st.write("Doughnut Chart for Sentiments:")
    sentiment_counts = df['sentiment'].value_counts()
    labels = [sentiment_labels[sentiment] for sentiment in sentiment_counts.index]
    values = sentiment_counts.values

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%.2f%%', startangle=90, colors=sns.color_palette('Set3'))
    ax.axis('equal')
    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)

# Drop the "tweetid" column
raw = raw.drop("tweetid", axis=1)


# Building out the "Information" page
if selection == "Information":
    st.title("Insights")
    st.subheader("Discover the power of data with Sense Solutions - Your partner for unlocking insights from social media")

    st.markdown("## Data Insights")
    st.write("Our analysis of the dataset revealed valuable insights about sentiment distribution in social media. Here are some key findings:")

    image = Image.open("./resources/imgs/len.png")
    st.image(image, caption="", width=800)
    
    # Display the sentiment distribution doughnut chart
    st.markdown("### Sentiment Distribution")
    # st.pyplot(fig)
    
    image = Image.open("./resources/imgs/dist.png")
    st.image(image, caption="", width=800)

    # Display word clouds for each sentiment
    st.markdown("### Word Clouds")
    st.write("Word clouds provide a visual representation of the most frequently occurring words associated with each sentiment.")

    image = Image.open("./resources/imgs/cloud.png")
    st.image(image, caption="", width=800)

    st.markdown("## About Our Solutions")
    st.write("At Sense Solutions, we specialize in unlocking insights from social media data. With our advanced data analytics techniques and machine learning models, we help businesses harness the power of social media to drive informed decision-making.")

    st.write("Our solutions include:")
    st.markdown("- Sentiment Analysis: Gain a deep understanding of public sentiment towards your brand, products, or services.")
    st.markdown("- Trend Analysis: Identify emerging trends and topics relevant to your industry.")
    st.markdown("- Influencer Identification: Find influential voices in your target audience to amplify your brand reach.")
    st.markdown("- Customer Segmentation: Divide your customer base into meaningful segments for targeted marketing campaigns.")
    st.markdown("- Social Media Monitoring: Monitor and track conversations about your brand in real-time.")

    st.write("Partner with us to unlock the power of social media data and make data-driven decisions for your business.")


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
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(top_usernames.values, labels=top_usernames.index, autopct='%.2f%%')
        ax.set_title(f'Top {top_n} Usernames with the Most Retweets')
        ax.axis('equal')

        # Display the chart
        st.pyplot(fig)
    else:
        st.write('No retweets with usernames found.')


# Building out the predication page
if selection == "Prediction":
    st.title("Tweet Classifer powered by Sense Solutions")
    st.subheader("Climate change tweet classification")
    st.subheader("Enter a tweet to be classified or upload test data for accurate sentiment predictions")

    st.info("Prediction with Machne Learning Models")
    
     # Creating a text box for user input
    tweet_text = st.text_area("Enter Text", "Type Here")
    
    st.info("Select one model for prediction")
    # Select classifier
    classifier_name = st.selectbox(
        "Select Classifier", ["Logistic_regression", "decision_tree", "knn", "random_forest"])

    # Load classifier
    classifier = load_classifier(classifier_name)

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

    

     # Create a file uploader for test data
    uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

    # Check if a file is uploaded
    if uploaded_file is not None:
        with st.spinner("Uploading..."):
            col1, col2 = st.columns(2)

            # Read the uploaded CSV file
            df_test = pd.read_csv(uploaded_file)

            # Display the uploaded test data
            st.subheader("Uploaded Test Data successfully")

            if col1.button("Classify Your Data"):
                # Clean and classify the tweets
                df_test['clean_tweet'] = df_test['message'].apply(clean_data)
                
                # Apply vectorization to the 'clean_tweet' column
                vectorized_data = tweet_cv.transform(df_test['clean_tweet']).toarray()
                
                # Make predictions using the selected classifier
                predictions = classifier.predict(vectorized_data)

                # Add the predictions as a new column in the DataFrame
                df_test['sentiment'] = predictions

                st.dataframe(df_test.head(5))

                st.subheader("Sentiment Split of Your Data")
                sentiment_counts = df_test['sentiment'].map(sentiment_labels).value_counts()
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%.2f%%")
                st.pyplot(fig)

                # Display the top words word clouds
                st.subheader("Word Clouds of Your Data")
                word_cloud(df_test, 10, 'clean_tweet')


            if col2.button("Get Retweets"):
                st.subheader("Users with the Most Retweets")
                get_most_retweeted_username(df_test, top_n=20)


# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

logo = Image.open('resources/imgs/logo.jpeg')
#olwethu = Image.open('resources/imgs/streamlit-logo.png')
#pertunia = Image.open('resources/imgs/streamlit.png')
#orifuna = Image.open('resources/imgs/EDSA_logo.png')
#neg_freq = Image.open('resources/imgs/streamlit-logo.png')
#neu_freq = Image.open('resources/imgs/streamlit.png')

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

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
    # Add an image to the sidebar
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.sidebar.image('resources/imgs/logo.jpeg', use_column_width=True)
	st.title("Tweets Classification")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Welcome", "About the App", "Information", "Prediction", "About Us"]
	selection = st.sidebar.radio("Choose Option", options)

	#Building welcome page
	if selection == "Welcome":
		
		st.markdown("<h2 style='text-align: center;'>Let's Classify</h2>", unsafe_allow_html=True)
		wel = Image.open("resources/imgs/Welcome.png")
		st.image(wel,use_column_width=True, clamp=False, width = 200, output_format="PNG")
		st.write("Stand Out from the Competition: By using our sentiment analysis model, you'll have an advantage over others because you can respond quickly to public sentiment, address concerns, and align your efforts with what people are saying.")
		st.write("Support for Decision Making: Our model acts as a reliable tool to help you make decisions based on data. You can use the insights to guide your marketing strategies, improve your products, and handle any crises that may arise.")

    	


	# Building out the "Information" page
	if selection == "Information":
		sentiment_labels={-1: 'Anti', 0: 'Neutral', 1: 'Pro', 2: 'News'}
		raw['sentiment_label']=raw['sentiment'].replace(sentiment_labels)

		count_data=raw['sentiment_label'].value_counts()
		percent_data=(count_data / len(raw)) * 100
		plt.figure(figsize=(8, 6))
		plt.pie(count_data, labels=count_data.index, autopct='%1.1f%%', startangle=90)
		plt.axis('equal')
	

		plt.title('Sentiment Distribution')
		st.pyplot(plt)

		st.subheader("Raw Twitter data and label")
		# Create a filter for sentiment categories
		sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Negative", "Neutral", "Pro News"])
		cv = CountVectorizer(stop_words='english')

        # Filter the data based on selected sentiment
		if sentiment_filter == "All":
			filtered_data = raw
			words = cv.fit_transform(raw.message)
			sum_words = words.sum(axis=0)
			words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
			words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
			wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words_freq))
		elif sentiment_filter == "Positive":
			filtered_data = raw[raw['sentiment'] == 1]
			words = cv.fit_transform(raw.message)
			sum_words = words.sum(axis=0)
			words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
			words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
			wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words_freq))
		elif sentiment_filter == "Negative":
			filtered_data = raw[raw['sentiment'] == -1]
			words = cv.fit_transform(raw.message)
			sum_words = words.sum(axis=0)
			words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
			words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
			wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words_freq))
		elif sentiment_filter == "Neutral":
			filtered_data = raw[raw['sentiment'] == 0]
			words = cv.fit_transform(raw.message)
			sum_words = words.sum(axis=0)
			words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
			words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
			wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words_freq))
		elif sentiment_filter == "Pro News":
			filtered_data = raw[raw['sentiment'] == 2]
			words = cv.fit_transform(raw.message)
			sum_words = words.sum(axis=0)
			words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
			words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
			wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words_freq))

		st.image(wordcloud.to_array())
		st.table(filtered_data.head(20))
		
			
	if selection == "About the App":

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
		
		
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text", "Type Here")

		if st.button("Classify"):
			classifier = joblib.load('resources/LR.pkl')
			# Transform user input
			vect_text = tweet_cv.transform([tweet_text]).toarray()

			# Make prediction using the selected classifier
			prediction = classifier.predict(vect_text)
			st.write("The predicted result is:", prediction)

			# When model has successfully run, print prediction
			#st.success("Text Categorized as: {}".format(prediction))

			# Display an image
			#image = Image.open("./resources/imgs/EDSA_logo.png")
			#st.image(image, caption="Image Caption", use_column_width=True)


	# Streamlit app code
	if selection == "About Us":
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

		imi = Image.open("resources/imgs/Imi.jpeg")
		leh = Image.open("resources/imgs/Lehlohonolo.jpeg")
		ori = Image.open("resources/imgs/Orifuna.jpeg")
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

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

import re
import contractions
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
nltk.download('stopwords')

def remove_noise(text):
    # Replace URLs with "url"
    text = re.sub(r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+', 'http', text)

    # Replace Twitter handles with "user"
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags using regular expression
    # text = re.sub(r'#\w+', '', text)

    # Replace special characters with a space
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text
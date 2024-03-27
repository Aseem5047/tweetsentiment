from flask import Flask, request, jsonify
import pickle
import re

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import string
import pandas as pd
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

# Load tweets from CSV
# tweets_df = pd.read_csv('extractedTweets.csv')
tweets_df = pd.read_csv('twitterValidation.csv')

# Load vectorizer
vectoriser = None
with open('vectoriser.pickle', 'rb') as file:
    vectoriser = pickle.load(file)

# Load model
LRmodel = None
with open('SentimentLR.pickle', 'rb') as file:
    LRmodel = pickle.load(file)

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Function to preprocess tweets
def tweets_to_words(tweets):
    ''' Convert list of tweet texts into a list of sequences of words '''
    processed_tweets = []
    for tweet in tweets:
        # Convert to lowercase
        tweet = tweet.lower()

        # Remove URLs
        tweet = re.sub(r'http\S+|www\S+|https\S+', 'URL', tweet)

        # Remove mentions
        tweet = re.sub(r'@[A-Za-z0-9_]+', 'USERNAME', tweet)
        
        # Replace all emojis.
        # e.g., emojis = {'😀': 'happy', '😂': 'laughing', ...}
        emojis = {}
        for emoji, meaning in emojis.items():
            tweet = tweet.replace(emoji, "EMOJI" + meaning)   

        # Remove hashtags
        tweet = re.sub(r'#\w+', '', tweet)

        # Tokenize using TweetTokenizer
        tokenizer = TweetTokenizer()
        words = tokenizer.tokenize(tweet)

        # Remove punctuation
        words = [word.strip(string.punctuation) for word in words if word.strip(string.punctuation)]

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if w not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]

        processed_tweets.append(words)

    return processed_tweets

# Predict function
def predict(vectoriser, model, texts):
    processedText = tweets_to_words(texts)
    processedText = [' '.join(tokens) for tokens in processedText]
    textdata = vectoriser.transform(processedText)
    sentiment = model.predict(textdata)
    
    data = []
    for text, pred in zip(texts, sentiment):
        data.append((text, pred))
  
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df['sentiment'] = df['sentiment'].replace([0, 1, 2], ["Negative", "Neutral", "Positive"])
 
    return df

# Predict route
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_route():
    try:
        data = request.get_json()
        texts = data.get('tweets', [])

        # vectoriser, LRmodel = load_models()

        result = predict(vectoriser, LRmodel, texts)

        return jsonify({'results': result.to_dict(orient='records')})

    except Exception as e:
        # If an exception occurs, return an error response
        error_message = str(e)
        return jsonify({'error': error_message}), 500

# Endpoint to generate random tweets
@app.route('/random_tweets', methods=['POST'])
@cross_origin()
def random_tweets():
    try:
        data = request.get_json()
        count = int(data.get('count', 5))  # Default count is 5 if not provided by the user

        # Check if count is greater than the total number of tweets
        if count > len(tweets_df):
            return jsonify({'error': 'Count exceeds the total number of tweets.'}), 400

        # Shuffle the DataFrame to ensure randomness
        shuffled_tweets_df = tweets_df.sample(frac=1)

        # Randomly select tweets
        random_tweets = shuffled_tweets_df.head(count)['Tweets'].tolist()

        return jsonify({'tweets': random_tweets})
      
    except Exception as e:
        # If an exception occurs, return an error response
        error_message = str(e)
        return jsonify({'error': error_message}), 500

    
# Root route
@app.route('/')
def root():
    return 'Welcome to the sentiment analysis API!'

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    

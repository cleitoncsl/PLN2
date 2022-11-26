# Import Librariesfrom textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# texto
texto = input("Please enter keyword or hashtag to search: ")
noOfTweet = int(input("Please enter how many tweets to analyze: "))

def percentage(part,whole):
    return 100 * float(part)/float(whole)

# Authentication

diretorio = r"E:\Users\cleit\Documents\CURSO PYTHON\PLN"
nome = "cloud.png"
arquivo = diretorio + " \ " + nome
acecessTokenBearer = "AAAAAAAAAAAAAAAAAAAAAFMOjwEAAAAAq11%2BDpuyrlaYs3K33G59O7wLYPI%3DT9utmwv7th0rZF8jCZT98yQqypAzTbO538Nnk3doe8bWnTnP2I"

client = tweepy.Client(bearer_token=acecessTokenBearer)
tweets = "{} is:retweet lang:pt".format(texto)
paginator = tweepy.Paginator(
    client.search_recent_tweets,
    query=tweets,
    max_results=noOfTweet,
    limit=10
)

contador = 0
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

for tweet in paginator.flatten():
    tweet_list.append(tweet)
    analysis = TextBlob(tweet.text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    neg = score["neg"]
    neu = score["neu"]
    pos = score["pos"]
    comp = score["compound"]
    polarity += analysis.sentiment.polarity
    contador += 1
    print("----ANALISE DO TWEET N=" "{}".format(contador))
    print(tweet.text)
    print("----SENTIMENTO----")
    print("->NEGATIVO=" "{}".format(neg))
    print("->NEUTRO=" "{}".format(neu))
    print("->POSITIVO=" "{}".format(pos))
    print("->COMPOSTO=" "{}".format(comp))

    if neg > pos:
        negative_list.append(tweet.text)
        negative += 1
        print("RESULTADO: NEGATIVO")
    elif pos > neg:
        positive_list.append(tweet.text)
        positive += 1
        print("RESULTADO: POSITIVO")
    elif pos == neg:
        neutral_list.append(tweet.text)
        neutral += 1
        print("RESULTADO: NEUTRO")

positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, ".1f")
negative = format(negative, ".1f")
neutral = format(neutral, ".1f")

#Number of Tweets (Total, Positive, Negative, Neutral)tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
print("total number: ", len(tweet_list))
print("positive number: ", len(positive_list))
print("negative number: ", len(negative_list))
print("neutral number: ", len(neutral_list))

#Creating PieCart
labels = ['Positive ['+str(positive)+'%]', 'Neutral ['+str(neutral)+'%]','Negative ['+str(negative)+'%]']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue', 'red']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis Result for keyword= "+texto+"")
plt.axis('equal')
plt.show()
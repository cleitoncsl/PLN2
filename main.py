# Import Libraries
from pandas.io.formats import string
import tweepy
import unicodedata
import matplotlib.pyplot as plt
import pandas as pd
import string
import numpy as np
import nltk
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
texto = input("Digite o Termo ou rashtag desejado : ")
noOfTweet = int(input("Digite a quantidade de Tweets a ser analisado: "))

def percentage(part,whole):
    return 100 * float(part)/float(whole)

# Authentication

diretorio = r"E:\Users\cleit\Documents\CURSO PYTHON\PLN2"
nome = "cloud.png"
arquivo = diretorio + " \ " + nome
acecessTokenBearer = "AAAAAAAAAAAAAAAAAAAAAFMOjwEAAAAAq11%2BDpuyrlaYs3K33G59O7wLYPI%3DT9utmwv7th0rZF8jCZT98yQqypAzTbO538Nnk3doe8bWnTnP2I"

client = tweepy.Client(bearer_token=acecessTokenBearer)
# query = "{} is:retweet lang:pt".format(texto)
query = "{} lang:pt".format(texto)
paginator = tweepy.Paginator(
    client.search_recent_tweets,
    query=query,
    max_results=10,
    limit=10
)

tweet_list = []
for tweet in paginator.flatten():
    tweet_list.append(tweet)
    # print(tweet)

tweet_list_df = pd.DataFrame(tweet_list)
tweet_list_df = pd.DataFrame(tweet_list_df['text'])
print(tweet_list_df.head(5))


def preprocess_tweet(sen):
    """Limpeza de caracteres"""

    # CaixaBaixa
    sentence = sen.lower()

    # Remove RT
    sentence = re.sub('rt @\w+: ', " ", sentence)

    # Remove special characters
    sentence = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # Removing stopwords
    stopword = nltk.corpus.stopwords.words('portuguese')

    return sentence

cleaned_tweets = []
encoding = 'utf-8'

# criar modulo tradutor




# criar modulo tradutor

for tweet in tweet_list_df['text']:
    pre_process_byte = unicodedata.normalize('NFD', tweet).encode('ascii', 'ignore')
    pre_process = pre_process_byte.decode(encoding)
    cleaned_tweet = preprocess_tweet(pre_process)
    cleaned_tweets.append(cleaned_tweet)

tweet_list_df['cleaned'] = pd.DataFrame(cleaned_tweets)
tweet_list_df.head(5)

contador = 0
tweet_list_df[['polarity', 'subjectivity']] = tweet_list_df['cleaned'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tweet_list_df['cleaned'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    contador += 1
    print("------------------------------")
    print("----ANALISE DO TWEET N=" "{}----".format(contador))
    print("------------------------------")
    print("TEXTO-> " "{}".format(row))
    print("------SENTIMENTO------")
    print("->NEGATIVO= " "{}".format(neg))
    print("->NEUTRO= " "{}".format(neu))
    print("->POSITIVO= " "{}".format(pos))
    print("->COMPOSTO= " "{}".format(comp))
    print("------SENTIMENTO------")
    print(" ")

    if neg > pos:
        tweet_list_df.loc[index, 'sentiment'] = "negative"
        print("RESULTADO: NEGATIVO")
        print(" ")
    elif pos > neg:
        tweet_list_df.loc[index, 'sentiment'] = "positive"
        print("RESULTADO: NEGATIVO")
        print(" ")

    else:
        tweet_list_df.loc[index, 'sentiment'] = "neutral"
        print("RESULTADO: NEUTRO")
        print(" ")

    tweet_list_df.loc[index, 'neg'] = neg
    tweet_list_df.loc[index, 'neu'] = neu
    tweet_list_df.loc[index, 'pos'] = pos
    tweet_list_df.loc[index, 'compound'] = comp
    tweet_list_df.head(5)

# Creating new data frames for all sentiments (positive, negative and neutral)
tweet_list_df_negative = tweet_list_df[tweet_list_df["sentiment"]=="negative"]
tweet_list_df_positive = tweet_list_df[tweet_list_df["sentiment"]=="positive"]
tweet_list_df_neutral = tweet_list_df[tweet_list_df["sentiment"]=="neutral"]

# Function for count_values_in single columns
def count_values_in_column(data, feature):
    total = data.loc[:, feature].value_counts(dropna=False)
    percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])

# Count values for sentiment
count_values_in_column(tweet_list_df,"sentiment")
print(count_values_in_column(tweet_list_df,"sentiment"))
pichart = count_values_in_column(tweet_list_df, "sentiment")
name = pichart.index
size = pichart["Percentage"]
plt.title("Sentiment Analysis Result for keyword= "+texto+"")
plt.axis('equal')
plt.style.use('default')

# Create a circle for the center of the plot
my_circle = plt.Circle((0, 0), 0.7, color="white")
plt.pie(size, labels=name, colors=['green','blue','red'])
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

#Calculating tweet's lenght and word count
tweet_list_df['text_len'] = tweet_list_df['cleaned'].astype(str).apply(len)
tweet_list_df['text_word_count'] = tweet_list_df['cleaned'].apply(lambda x: len(str(x).split()))

round(pd.DataFrame(tweet_list_df.groupby("sentiment").text_len.mean()),2)
round(pd.DataFrame(tweet_list_df.groupby("sentiment").text_word_count.mean()),2)

#Removing Punctuation
def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

tweet_list_df['punct'] = tweet_list_df['text'].apply(lambda x: remove_punct(x))

#Appliyng tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text

tweet_list_df['tokenized'] = tweet_list_df['punct'].apply(lambda x: tokenization(x.lower()))

# Removing stopwords
stopword = nltk.corpus.stopwords.words('portuguese')


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


tweet_list_df['nonstop'] = tweet_list_df['tokenized'].apply(lambda x: remove_stopwords(x))

#Appliyng Stemmer
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

tweet_list_df['stemmed'] = tweet_list_df['nonstop'].apply(lambda x: stemming(x))

#Cleaning Text
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text


tweet_list_df.head(5)

#Appliyng Countvectorizer
countVectorizer = CountVectorizer(analyzer=clean_text)
countVector = countVectorizer.fit_transform(tweet_list_df['text'])
print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())

count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()

# Most Used Words
count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0,ascending=False).head(20)
countdf[1:11]

#Function to ngram
def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

#n2_bigram
n2_bigrams = get_top_n_gram(tweet_list_df['text'],(2,2),20)
print(n2_bigrams)

#n3_trigram
n3_trigrams = get_top_n_gram(tweet_list_df['text'],(3,3),20)
print(n3_trigrams)

def create_wordcloud(text):
    mask = np.array(Image.open(r"E:\Users\cleit\Documents\CURSO PYTHON\PLN2\cloud.png"))
    stopwords = set(stopword)
    wc = WordCloud(background_color="white",
                   mask=mask,
                   max_words=100,
                   stopwords=stopwords,
                   repeat=True)
    wc.generate(str(text))
    wc.to_file("c1_wordcloud.png")
    print("Word Cloud Saved Successfully")
    nome_final = "c1_wordcloud.png"
    arquivo_final = "{}\{}".format(diretorio, nome_final)
    print(arquivo_final)
    imagem = Image.open(arquivo_final, mode='r', formats=None)
    imagem.show()

create_wordcloud(tweet_list_df_negative["cleaned"].values)
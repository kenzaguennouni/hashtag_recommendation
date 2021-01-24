import pandas as pd
import numpy as np
import re
from textblob import Word, TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import gensim.downloader as api
import ast
import pickle
from langdetect import detect

file_name = "stopwords.pkl"


open_file = open(file_name, "rb")
list_stopwords = pickle.load(open_file)
open_file.close()
stop_words = list_stopwords

word2vec = gensim.models.keyedvectors.KeyedVectors.load("glove-twitter-50")
tweets = pd.read_csv("data_tweets.csv")
def preprocess_tweets(tweet, custom_stopwords=["RT"]):
    preprocessed_tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
    preprocessed_tweet = re.sub('[^\w\s#@]', '', preprocessed_tweet)
    preprocessed_tweet = re.sub('#[\w]+', '', preprocessed_tweet)
    preprocessed_tweet = " ".join(Word(word).lemmatize() for word in preprocessed_tweet.split() if  word not in custom_stopwords and word not in stop_words)
    return preprocessed_tweet.lower()
def tweet_to_vec(tweet, word2vec):
    vec = []
    for word in tweet.split():
        try:
            vec.append(word2vec[word])
        except:
            vec.append(np.zeros((50,)))
    return np.sum(vec, axis=0)
tweets["processed_tweet"] = tweets["tweet"].apply(lambda tweet : preprocess_tweets(tweet))
tweets["hashtags"] = tweets["hashtags"].apply(lambda hashtag : ast.literal_eval(hashtag))
tweets.drop_duplicates(subset=["tweet"],inplace=True)
tweets.reset_index(inplace=True)
def build_embedding_dict(tweets, word2vec):
    embedding_dict = []
    bad_tweet_indexs = []
    for i in range(len(tweets)):
        vec = tweet_to_vec(tweets[i], word2vec)
        if vec.shape != ():
            embedding_dict.append(vec)
        else:
            bad_tweet_indexs.append(i)
    return np.hstack(embedding_dict).reshape(-1, 50) , bad_tweet_indexs
embedding_dict, bad_tweet_indexs = build_embedding_dict(tweets["processed_tweet"].values.tolist(), word2vec)
tweets.drop(bad_tweet_indexs, axis=0, inplace=True)
tweets.reset_index(inplace=True)
tweets.drop(["index"], axis=1, inplace=True)


def similar_tweets(tweet, embedding_dict, word2vec):
    cosine_sim = cosine_similarity(embedding_dict, tweet_to_vec(tweet, word2vec).reshape(1, -1))
    flat_cosine_sim = cosine_sim.flatten()
    index = np.argpartition(flat_cosine_sim, -5)[-5:]
    sim_tweets = []
    sim_hashtags = []

    for i in index:
        sim_tweets.append(tweets["tweet"][i])

        for hashtag in tweets["hashtags"][i]:
            lang = detect(hashtag)
            if (lang == 'en'):
                sim_hashtags.append(hashtag)
    return sim_hashtags[::-1]

#print(similar_tweets('machine learning and dataScience is the future', embedding_dict, word2vec))
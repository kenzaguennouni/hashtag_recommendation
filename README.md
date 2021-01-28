# hashtag_recommendation
## Table of contents
* [General info](#general-info)

* [Technologies](#Technologies)

* [Setup](#setup)

* [Result](#Result)

## General info
The idea behind the project is to recommend hashtags for tweet. 
Then we will deploy a FLASK application to use it. 
This guideline is about how to run the Flask application locally.  
The data file is the result of extracting data using tweepy and twint.

  * Twint is an advanced Twitter scraping tool written in Python that allows you to retrieve Tweets from Twitter profiles without using the API
  * Tweepy is python library for accessing Twitter API for retrieving tweets.
  
This application based on Word2vec and uses a glove (Global Vectors for Word Representation) tweet-50 is a gensim pretrained model.
After i download my model i use a function build_embedding_dict  to have a vector representation for all my tweet and i call function "similar tweets" whitch take the tweet we want to predict hashtag, my embedding_dict and my word2vec pre-trained.

## Technologies and librairies

Project is created with:
* Python version: 3 or more
* PyCharm : community edition
* tweepy
* twint
* texblob
* flask
* sklearn
* gensim

## Setup 

To run this project locally follow the steps below : 

1- Clone the repo.

2- Open the project in PyCharm.

3- Extract zip file containing data : data_tweets.zip

4- Install the required libraries from the PyCharm terminal :
``` 
$ pip install -r requirements.txt 
```

5- run download_glove.py

6- Use application without flask run hashtag_recommendation.py

7- Use application with flask Run app.py

## Result 

![alt text](https://user-images.githubusercontent.com/77112759/105642880-cb0d2200-5e8c-11eb-92ec-a609707da3f6.png)


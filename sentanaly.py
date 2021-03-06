# -*- coding: utf-8 -*-
"""Untitled

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KgDRUcciYgUwdkjWe2iXlFmITjdUxOp_
"""

import sys
import os
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

import numpy as np
import pandas as pd
``
from tqdm import tqdm

from os import path

#authenticate with watson

from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

import config

ibm_watson_authenticator = IAMAuthenticator(config.IBM_WATSON_API_KEY)
ibm_watson_tone_analyzer = ToneAnalyzerV3(version='2017-09-21', authenticator=ibm_watson_authenticator)
ibm_watson_tone_analyzer.set_service_url(config.IBM_WATSON_URL)

!pip install --upgrade git+https://github.com/flairNLP/flair.git

clear_output()

from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')

clear_output()

from flair.models import TextClassifier

classifier = TextClassifier.load('en-sentiment')

clear_output()

#AUTH need KEYS HERE
auth = tweepy.AppAuthHandler(TWITTER_KEY, TWITTER_SECRET_KEY)

api = tweepy.API(auth, wait_on_rate_limit=True,
				   wait_on_rate_limit_notify=True)

if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

#@title Twitter Search API Inputs
#@markdown ### Enter Search Query:
searchQuery = '' #@param {type:"string"}
#@markdown ### Enter Max Tweets To Scrape:

maxTweets = 1000 #@param {type:"slider", min:0, max:45000, step:100}
Filter_Retweets = True #@param {type:"boolean"}

tweetsPerQry = 100  # this is the max the API permits
tweet_lst = []

if Filter_Retweets:
  searchQuery = searchQuery + ' -filter:retweets'  # to exclude retweets

sinceId = None


max_id = -10000000000

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
while tweetCount < maxTweets:
    try:
        if (max_id <= 0):
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry, lang="en")
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        lang="en", since_id=sinceId)
        else:
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        lang="en", max_id=str(max_id - 1))
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        lang="en", max_id=str(max_id - 1),
                                        since_id=sinceId)
        if not new_tweets:
            print("No more tweets found")
            break
        for tweet in new_tweets:
          if hasattr(tweet, 'reply_count'):
            reply_count = tweet.reply_count
          else:
            reply_count = 0
          if hasattr(tweet, 'retweeted'):
            retweeted = tweet.retweeted
          else:
            retweeted = "NA"
            
          # fixup search query to get topic
          topic = searchQuery[:searchQuery.find('-')].capitalize().strip()
          
          # fixup date
          tweetDate = tweet.created_at.date()
          
          tweet_lst.append([tweetDate, topic, 
                      tweet.id, tweet.user.screen_name, tweet.user.name, tweet.text, tweet.favorite_count, 
                      reply_count, tweet.retweet_count, retweeted])

        tweetCount += len(new_tweets)
        print("Downloaded {0} tweets".format(tweetCount))
        max_id = new_tweets[-1].id
    except tweepy.TweepError as e:
        # error
        print("some error : " + str(e))
        break

clear_output()
print("Downloaded {0} tweets".format(tweetCount))

pd.set_option('display.max_colwidth', -1)

# load it into a pandas dataframe
tweet_df = pd.DataFrame(tweet_lst, columns=['tweet_dt', 'topic', 'id', 'username', 'name', 'tweet', 'like_count', 'reply_count', 'retweet_count', 'retweeted'])
tweet_df.to_csv('tweets.csv')
tweet_df.head()

EMOTIONAL_TONES = ["Anger", "Disgust", "Fear", "Joy", "Sadness"]

def get_emotion(text : str) -> pd.Series:

    json_output = ibm_watson_tone_analyzer.tone({'text': text}, content_type='application/json').get_result()

    main_emotion = ""
    high_score = 0.0

    for tone in json_output["document_tone"]["tones"]: # iterate through the tones
        if tone["tone_name"] in EMOTIONAL_TONES and tone['score'] > high_score:
            main_emotion = tone['tone_name']
            high_score = tone["score"]

    return pd.Series((main_emotion, np.nan if main_emotion == "" else high_score))
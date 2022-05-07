#!/usr/bin/env python
# coding: utf-8
# %%


#Use TF-IDF embeddings to train topic model
from bertopic import BERTopic
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

#Load in data
df = pd.read_csv('data/tweets.csv')
#Drop tweets not in english
df = df.loc[df['language'] == 'en']
df['tweet'] = df['tweet'].str.replace(r'http\S+', '')
df = df.loc[df['tweet'] != '']
docs = df['tweet'].reset_index(drop=True)

#Create vectorizer
vectorizer = TfidfVectorizer(min_df=5)
embeddings = vectorizer.fit_transform(docs)

#Train our topic model using TF-IDF vectors
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs, embeddings)

#Save model
topic_model.save('project_BERTopic')


#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Use TF-IDF embeddings to train topic model
from bertopic import BERTopic
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#Load in data
df = pd.read_csv('tweets.csv')
#Drop tweets not in english
df = df.loc[df['language'] == 'en']
docs = df['tweet']

#Create vectorizer
vectorizer = TfidfVectorizer(min_df=5)
embeddings = vectorizer.fit_transform(docs)

#Train our topic model using TF-IDF vectors
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs, embeddings)

#Save model
topic_model.save('project_BERTopic')


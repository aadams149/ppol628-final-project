#!/usr/bin/env python
# coding: utf-8
# %%


import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import yaml

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

docs = params["preprocessing"]["max_min_docs"]
ngrams = params['preprocessing']['n_grams']
    
#Read in data
tweets = pd.read_csv('data/tweets.csv')
tweets = tweets.loc[tweets['language'] == 'en']

states = pd.read_csv('data/elected_officials.csv')

states = states.melt(id_vars = ['State',
                                'StateAbbr',
                                'Name',
                                'Party',
                                'Inauguration',
                                'Title',
                                'office'],
                    value_vars = ['officialTwitter',
                                  'campaignTwitter',
                                  'othertwitter'],
                    var_name = 'account_type',
                    value_name = 'twitter')

states['twitter'] = states['twitter'].str.lower()
states = states.loc[states['Party'] != 'Independent']

tweets = tweets.merge(states, left_on = 'username', right_on = 'twitter')

#Create numeric labels based on state names

#Merge labels into data frame
labels = pd.DataFrame(tweets['Party'].unique()).reset_index()
#Add one because zero indexed
labels['index'] = labels['index']+1
labels.columns = ['label', 'Party']
tweets = tweets.merge(labels, on = 'Party')

#Select labels as targets
y = tweets['label']

#Select text columns as features
X = tweets["tweet"]

#Training test split 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

#Preprocess text 
vectorizer = TfidfVectorizer(
    min_df=docs['smallest'],
    max_df=docs['largest'],
    stop_words="english",
    ngram_range = (ngrams['min'],ngrams['max'])
)

#Create pipeline with preprocessing and linear SVC
pipe = Pipeline([
    ('preprocess', vectorizer),
    ('LinearSVC', LinearSVC())
])

#Fit pipe to training data
fitted_pipe = pipe.fit(X_train, y_train)

#Export pickeled pipe
joblib.dump(fitted_pipe, 'outputs/bc_party_pipe.pkl')

#Generate predictions
y_pred = pipe.predict(X_test)

#Output metrics to JSON
metrics = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
metrics["weighted avg"].to_json("metrics/bc_party_metrics.json")


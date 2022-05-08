---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Alexander Adams

# PPOL628 Text as Data

# Final Project Notebook


I scraped tweets from several accounts and concatenated them all into a single .csv file, called `tweets.csv`.

```python
#!dvc pull
```

```python
from bertopic import BERTopic
import numpy as np
import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import yaml
pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.max_seq_items = None
```

```python
tweets = pd.read_csv('data/tweets.csv')
```

```python
#Drop tweets not in english
tweets = tweets.loc[tweets['language'] == 'en']
tweets['tweet'] = tweets['tweet'].str.replace(r'http\S+', '')
tweets = tweets.loc[tweets['tweet'] != '']
tweets = tweets.reset_index(drop=True)
```

```python
tweets.shape
```

```python
tweets.dtypes
```

```python
tweets.head(5)
```

# Topic Modeling: What Do State-Level Elected Officials Tweet About?

```python
topic_model = BERTopic.load('project_BERTopic')
```

```python
topics_list = topic_model.get_topics()
len(topic_model.get_topics())
```

```python
topic_model.visualize_topics(topics_list)
```

```python
probs = topic_model.hdbscan_model.probabilities_
topics = topic_model._map_predictions(topic_model.hdbscan_model.labels_)
```

```python
new_topics, new_probs = topic_model.reduce_topics(tweets['tweet'], topics, probs, nr_topics = 10)
```

```python
topic_model.visualize_topics()
```

```python
topic_model.get_topic_info()
```

```python
dynamic_topics = topic_model.topics_over_time(tweets['tweet'],
                                              new_topics, 
                                              tweets['date'])
```

```python
topic_model.visualize_topics_over_time(dynamic_topics,
                                       topics=[0,1,2,3,4,5,6,7,8,9],
                                       width = 950)
```

I expected some of the topics to be cyclical or intermittent, but I am surprised at how clear the spikes are. Topic 0, with the top words "your vote ballot", spikes almost every november and is nonexistent the rest of the year. Topic 2, which is about veterans, exhibits similar patterns. Topic 7, which is about Ukraine, only appears starting in February 2022, and topic 1, which is about COVID-19, sees its biggest spikes during the winter of 2020-21 and the Omicron wave beginning in late 2021. In general, all of these topics spike in the winter, and occur barely if at all during the rest of the year.


___________

# Multiclass Classification: What can I predict using tweets?


For the bulk of this project, I chose to run several multiclass classification tasks, in order to identify what, if anything, could be predicted by these tweets. First, I tried to see if I could identify the state an official represents:

Task: Multiclass Classification (State)

Number of Classes: 50 (U.S. States)

Script: `multiclass_state.py`

DVC YAML Stage: `multiclass_state`

```python
import joblib
import numpy as np
from sklearn.metrics import (confusion_matrix, multilabel_confusion_matrix, 
precision_recall_fscore_support, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
```

```python
#Load the trained multiclass pipeline
pipe = joblib.load('outputs/mc_state_pipe.pkl')
```

```python
#Perform necessary data processing
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

tweets = tweets.merge(states, left_on = 'username', right_on = 'twitter')

#Create numeric labels based on state names

#Merge labels into MTG data frame
labels = pd.DataFrame(tweets['State'].unique()).reset_index()
#Add one because zero indexed
labels['index'] = labels['index']+1
labels.columns = ['state_label', 'State']
tweets = tweets.merge(labels, on = 'State')
```

```python
#Select labels as targets
y = tweets['state_label']

#Select text columns as features
X = tweets["tweet"]
```

```python
pipe.fit(X,y)
```

```python
y_pred = pipe.predict(X)
```

Rather than print out a 50x50 confusion matrix, I'm going to simplify the matrix to just a few columns:

    -state: the abbreviation for the state
    -correct: the number of correctly classified tweets for that state
    -incorrect: the number of incorrectly classified tweets for that state
    -errors: the labels which were applied incorrectly for each state

```python
cm = confusion_matrix(y,y_pred)
```

```python
state_cm = pd.DataFrame.from_dict({'state': pd.unique(tweets['StateAbbr']),
                                   'correct': np.diag(cm),
                                   'incorrect': cm.sum(1)-np.diag(cm),
                                   'total_tweets': cm.sum(1),
                                   'precision': np.diag(cm)/cm.sum(0),
                                   'recall': np.diag(cm)/cm.sum(1)})
```

```python
cm = pd.DataFrame(cm)
cm.columns = pd.unique(tweets['StateAbbr'])
cm.index = pd.unique(tweets['StateAbbr'])
```

```python
cols = cm.columns.values
mask = cm.gt(0.0).values
np.fill_diagonal(mask, False)
out = [cols[x].tolist() for x in mask]
```

```python
state_cm['errors'] = out
```

```python
state_cm
```

I don't see any real trends here in terms of geography. That suggests to me that, while the Linear Support Vector Classifier was effective most of the time (as evidenced by the uniformly high precision and recall scores), incorrect guesses were not informed by geography (i.e. for a tweet by an Ohio official, the classifier was not more likely to select another Midwestern state than a non-midwestern state). The one interesting pattern that is clear, however, is that Colorado and California appear in many of these error lists. California makes sense, since it is the largest state (and it is possible that officials in larger states tweet more than officials in smaller states because more happens in larger states). But Colorado is a mid-sized state; I am not sure why the classifier would be more likely to predict Colorado as the label than other states. 


________

Next, I tried to see if I could identify the office an official holds:

Task: Multiclass Classification (Political Office)

Number of Classes: 5 (Governor, Lieutenant Governor, Attorney General, Secretary of State, Treasurer)

Script: `multiclass_office.py`

DVC YAML Stage: `multiclass_office`

```python
#Load the trained multiclass pipeline
pipe = joblib.load('outputs/mc_office_pipe.pkl')
```

```python
labels = pd.DataFrame(tweets['office'].unique()).reset_index()
#Add one because zero indexed
labels['index'] = labels['index']+1
labels.columns = ['office_label', 'office']
tweets = tweets.merge(labels, on = 'office')
```

```python
#Select labels as targets
y = tweets['office_label']

#Select text columns as features
X = tweets["tweet"]
```

```python
pipe.fit(X,y)
```

```python
y_pred = pipe.predict(X)
```

```python
cm = pd.DataFrame(confusion_matrix(y,y_pred))
cm.columns = pd.unique(tweets['office'])
cm.index = pd.unique(tweets['office'])
cm
```

```python
cm = confusion_matrix(y,y_pred)
office_cm = pd.DataFrame.from_dict({'office': pd.unique(tweets['office']),
                                   'correct': np.diag(cm),
                                   'incorrect': cm.sum(1)-np.diag(cm),
                                   'total_tweets': cm.sum(1),
                                   'precision': np.diag(cm)/cm.sum(0),
                                   'recall': np.diag(cm)/cm.sum(1)})
```

```python
office_cm
```

________

Next, I tried to see if I could identify the political party of an official:

Task: Binary Classification (Political Party)

Number of Classes: 2 (Democrat, Republican)

Script: `twoclass_party.py`

DVC YAML Stage: `twoclass_party`

```python
#Load the trained multiclass pipeline
pipe = joblib.load('outputs/bc_party_pipe.pkl')
```

```python
labels = pd.DataFrame(tweets['Party'].unique()).reset_index()
#Add one because zero indexed
labels['index'] = labels['index']+1
labels.columns = ['party_label', 'Party']
tweets = tweets.merge(labels, on = 'Party')
partyclass = tweets.loc[tweets['Party'] != 'Independent']
```

```python
#Select labels as targets
y = partyclass['party_label']

#Select text columns as features
X = partyclass["tweet"]
```

```python
pipe.fit(X,y)
```

```python
y_pred = pipe.predict(X)
```

```python
cm = pd.DataFrame(confusion_matrix(y,y_pred))
cm.columns = pd.unique(partyclass['Party'])
cm.index = pd.unique(partyclass['Party'])
cm
```

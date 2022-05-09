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

<!-- #region slideshow={"slide_type": "slide"} -->
# Tweet Like a Politician

## Using Tweets to Predict Identity Characteristics of State-Level Political Figures in the United States

### Alexander Adams

### PPOL628 Text as Data

### Final Project
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Background

* U.S. politics is increasingly nationalized
    * Tweets by members of congress likely only convey partisanship
* Politics at the state level may be less polarized
    * More able to identify traits because partisanship is lower
* Tweets may also be less focused on national culture war and more focused on real issues
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Dataset

* 48,249 tweets, scraped from official, campaign, and personal accounts
* Political office: Governor, Lieutenant Governor, Secretary of State, Attorney General, Treasurer
* Tweet data includes date tweet was posted
* Metadata: politician's name, state, office, and political party
* Majority of tweets are from 2018 or later
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Question 1

#### What topics do state-level politicians tweet about?
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Task: Topic Modeling

Method: BERTopic

Script: `topic_model.py`

DVC YAML Stage: `topic_model`
<!-- #endregion -->

```python slideshow={"slide_type": "skip"}
#!dvc pull
```

```python slideshow={"slide_type": "skip"}
from bertopic import BERTopic
import matplotlib.pyplot as plt
%matplotlib inline 
import numpy as np
import pandas as pd
import re
import seaborn as sn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import yaml
pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.max_seq_items = None
```

```python slideshow={"slide_type": "skip"}
tweets = pd.read_csv('data/tweets.csv')
```

```python slideshow={"slide_type": "skip"}
#Drop tweets not in english
tweets = tweets.loc[tweets['language'] == 'en']
tweets['tweet'] = tweets['tweet'].str.replace(r'http\S+', '')
tweets = tweets.loc[tweets['tweet'] != '']
tweets = tweets.reset_index(drop=True)
```

```python slideshow={"slide_type": "skip"}
tweets.shape
```

```python slideshow={"slide_type": "skip"}
tweets.dtypes
```

```python slideshow={"slide_type": "skip"}
tweets.head(5)
```

<!-- #region slideshow={"slide_type": "skip"} -->
#### Topic Modeling: What Do State-Level Elected Officials Tweet About?
<!-- #endregion -->

```python slideshow={"slide_type": "skip"}
topic_model = BERTopic.load('project_BERTopic')
```

```python slideshow={"slide_type": "skip"}
topics_list = topic_model.get_topics()
len(topic_model.get_topics())
```

```python slideshow={"slide_type": "slide"} hidePrompt=true hideCode=true
#Static image in case real plot doesn't load
from IPython.display import Image
Image(filename='plots/topicmodel_full.png') 
```

```python slideshow={"slide_type": "skip"}
topic_model.visualize_topics(topics_list)
```

```python slideshow={"slide_type": "skip"}
probs = topic_model.hdbscan_model.probabilities_
topics = topic_model._map_predictions(topic_model.hdbscan_model.labels_)
```

```python slideshow={"slide_type": "skip"}
new_topics, new_probs = topic_model.reduce_topics(tweets['tweet'], topics, probs, nr_topics = 10)
```

```python slideshow={"slide_type": "slide"} hidePrompt=true hideCode=true
Image(filename='plots/topicmodel_10.png')
```

```python slideshow={"slide_type": "skip"}
topic_model.visualize_topics()
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Topics
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"} hidePrompt=true hideCode=false
topic_model.get_topic_info()[1:10]
```

```python slideshow={"slide_type": "skip"}
dynamic_topics = topic_model.topics_over_time(tweets['tweet'],
                                              new_topics, 
                                              tweets['date'])
```

```python slideshow={"slide_type": "slide"} hideCode=true hidePrompt=true
#Static image in case full plot does not load
Image(filename='plots/topics_over_time.png')
```

```python slideshow={"slide_type": "skip"} hideCode=false hidePrompt=true
topic_model.visualize_topics_over_time(dynamic_topics,
                                       topics=[0,1,2,3,4,5,6,7,8,9],
                                       width = 950)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Observations:

* Most topics spike in winter
    * Ex. Topic 0 (vote/ballot = elections), Topic 8 (veterans), Topic 1 (Christmas)
* Topics related to COVID-19 also spiked around the same time as major waves (esp. Omicron)
* Some iterations of this graph generated during testing included Ukraine topic
    * Basically nonexistent until Feb 2022, then huge spike
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Question 2: Can Tweets be used to predict the state an official leads?
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Task: Multiclass Classification (State)

Method: Linear Support Vector Classifier

Number of Classes: 50 (U.S. States)

Script: `multiclass_state.py`

DVC YAML Stage: `multiclass_state`
<!-- #endregion -->

```python slideshow={"slide_type": "skip"}
import joblib
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, classification_report)
```

```python slideshow={"slide_type": "skip"}
#Load the trained multiclass pipeline
pipe = joblib.load('outputs/mc_state_pipe.pkl')
```

```python slideshow={"slide_type": "skip"}
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

```python slideshow={"slide_type": "skip"}
#Select labels as targets
y = tweets['state_label']

#Select text columns as features
X = tweets["tweet"]
```

```python slideshow={"slide_type": "skip"}
pipe.fit(X,y)
```

```python slideshow={"slide_type": "skip"}
y_pred = pipe.predict(X)
```

<!-- #region slideshow={"slide_type": "slide"} -->
Rather than print out a 50x50 confusion matrix, I'm going to simplify the matrix to just a few columns:

    -state: the abbreviation for the state
    -correct: the number of correctly classified tweets for that state
    -incorrect: the number of incorrectly classified tweets for that state
    -errors: the labels which were applied incorrectly for each state
    -precision: true positives/(true positives + false positives)
    -recall: true positives/(true positives + false negatives)
    -errors: the state labels which were generated as false negatives

<!-- #endregion -->
```python slideshow={"slide_type": "skip"}
cm = confusion_matrix(y,y_pred)
```

```python slideshow={"slide_type": "skip"}
state_cm = pd.DataFrame.from_dict({'state': pd.unique(tweets['StateAbbr']),
                                   'correct': np.diag(cm),
                                   'incorrect': cm.sum(1)-np.diag(cm),
                                   'total_tweets': cm.sum(1),
                                   'precision': np.diag(cm)/cm.sum(0),
                                   'recall': np.diag(cm)/cm.sum(1)})
```

```python slideshow={"slide_type": "skip"}
cm = pd.DataFrame(cm)
cm.columns = pd.unique(tweets['StateAbbr'])
cm.index = pd.unique(tweets['StateAbbr'])
```

```python slideshow={"slide_type": "skip"}
cols = cm.columns.values
mask = cm.gt(0.0).values
np.fill_diagonal(mask, False)
out = [cols[x].tolist() for x in mask]
```

```python slideshow={"slide_type": "skip"}
state_cm['errors'] = out
```

```python slideshow={"slide_type": "slide"} hidePrompt=true hideCode=true
state_cm
```

<!-- #region slideshow={"slide_type": "slide"} -->
Observations:

* No apparent regional trends in errors
    * i.e. Southern states (like AL) were no more likely to be misclassified as other southern states than as states in other parts of the country
    * Possible that creating region labels would not improve performance
* Consistently strong performance across states
    * All precision and recall scores > 0.9, most are 0.98 or greater
    * Lowest scores are recall for California and Colorado
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Question 3: Can I predict the office a politician holds?
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Task: Multiclass Classification (Political Office)

Method: Linear Support Vector Classifier

Number of Classes: 5 (Governor, Lieutenant Governor, Attorney General, Secretary of State, Treasurer)

Script: `multiclass_office.py`

DVC YAML Stage: `multiclass_office`
<!-- #endregion -->

```python slideshow={"slide_type": "skip"}
#Load the trained multiclass pipeline
pipe = joblib.load('outputs/mc_office_pipe.pkl')
```

```python slideshow={"slide_type": "skip"}
labels = pd.DataFrame(tweets['office'].unique()).reset_index()
#Add one because zero indexed
labels['index'] = labels['index']+1
labels.columns = ['office_label', 'office']
tweets = tweets.merge(labels, on = 'office')
```

```python slideshow={"slide_type": "skip"}
#Select labels as targets
y = tweets['office_label']

#Select text columns as features
X = tweets["tweet"]
```

```python slideshow={"slide_type": "skip"}
pipe.fit(X,y)
```

```python slideshow={"slide_type": "skip"}
y_pred = pipe.predict(X)
```

```python slideshow={"slide_type": "subslide"} hideCode=true hidePrompt=true
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels = pd.unique(tweets['office']))
#plt.savefig('plots/office_cm.png')
```

```python slideshow={"slide_type": "skip"}
cm = confusion_matrix(y,y_pred)
office_cm = pd.DataFrame.from_dict({'office': pd.unique(tweets['office']),
                                   'correct': np.diag(cm),
                                   'incorrect': cm.sum(1)-np.diag(cm),
                                   'total_tweets': cm.sum(1),
                                   'precision': np.diag(cm)/cm.sum(1),
                                   'recall': np.diag(cm)/cm.sum(0)})
```

```python slideshow={"slide_type": "fragment"} hidePrompt=true hideCode=true
office_cm
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Observations:

* Fewer classes, but overall a less effective classifier
    * Esp. Lt. Governors (precision = 0.888)
    * Maybe Lt. Governors have less distinctive tweets than other state-level officials?
* Mean recall is slightly higher than mean precision
    * Classifier is better at avoiding false negatives than false positives
* Classes are imbalanced; count(governor) = 1.5x/2x count(other offices)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Question 4: Can I predict the political party of a state-level political official?
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Task: Binary Classification (Political Party)

Method: Linear Support Vector Classifier

Number of Classes: 2 (Democrat, Republican)

Script: `twoclass_party.py`

DVC YAML Stage: `twoclass_party`

Note: 2 officials are Independents, and were excluded from this model. In Minnesota, the Democratic party is called the Democratic Farmer-Labor party (DFL); politicians in that party were recoded as Democrats.
<!-- #endregion -->

```python slideshow={"slide_type": "skip"}
#Load the trained multiclass pipeline
pipe = joblib.load('outputs/bc_party_pipe.pkl')
```

```python slideshow={"slide_type": "skip"}
labels = pd.DataFrame(tweets['Party'].unique()).reset_index()
#Add one because zero indexed
labels['index'] = labels['index']+1
labels.columns = ['party_label', 'Party']
tweets = tweets.merge(labels, on = 'Party')
partyclass = tweets.loc[tweets['Party'] != 'Independent']
```

```python slideshow={"slide_type": "skip"}
#Select labels as targets
y = partyclass['party_label']

#Select text columns as features
X = partyclass["tweet"]
```

```python slideshow={"slide_type": "skip"}
pipe.fit(X,y)
```

```python slideshow={"slide_type": "skip"}
y_pred = pipe.predict(X)
```

```python slideshow={"slide_type": "subslide"} hidePrompt=true hideCode=true
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels = pd.unique(partyclass['Party']))
#plt.savefig('plots/party_cm.png')
```

```python slideshow={"slide_type": "fragment"} hidePrompt=true
print(classification_report(y, y_pred, target_names=pd.unique(partyclass['Party'])))
```

<!-- #region slideshow={"slide_type": "slide"} -->
Observations:

* Classifier can predict if a tweet was tweeted by a Republican or a Democrat with 97% accuracy
* Strong evidence that the two parties do tweet differently
    * Suggests initial hypothesis (state-level politics is not as polarized/nationalized as federal politics) is not true
        * At least not on Twitter
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Question 5: Can I predict *how* partisan an elected official is, based on their tweets?
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Task: Ideal Point Generation

Method: Wordfish (via R packages `quanteda` and `quanteda.textmodels`)

Output: Value indicating ideological position on left-right scale (further right = more conservative)

Script: `ideal_points.R`

Note: I was only able to find ideal points for governors and state treasurers. For Lt. Governors, Secretaries of State, and Attorneys General, the algorithm did not converge. 
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
#### Ideal points of governors
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"} hidePrompt=true hideCode=true
Image(filename='plots/gov_ideal.png') 
```

<!-- #region slideshow={"slide_type": "slide"} -->
#### Ideal points of state treasurers:
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"} hideCode=true hidePrompt=true
Image(filename='plots/trs_ideal.png')
```

<!-- #region slideshow={"slide_type": "slide"} -->
Observations:

* Governors are more polarized than treasurers
    * Even then, governors are not completely polarized (Ex. Charlie Baker (MA), Jim Justice (WV))
* Polarization could be linked to visibility
    * Officials from TX, FL tend to be at extremes
        * Do tweets make them more polarizing, or are tweets byproduct of polarization?
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
Conclusions and avenues for further exploration:

* Incorporate additional data (margin of victory in most recent election, partisanship of state)
* Consider length of incumbency
    * Wanted to test pre-/post-inauguration, but ran out of time
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
# Thank you for listening! Any questions?
<!-- #endregion -->

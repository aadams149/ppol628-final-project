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
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import yaml
pd.options.display.max_columns = None
```

```python
tweets = pd.read_csv('tweets.csv')
```

```python
tweets = tweets.loc[tweets['language'] == 'en']
```

```python
tweets.shape
```

```python
tweets.dtypes
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

I expected some of the topics to be cyclical or intermittent, but I am surprised at how clear the spikes are. Topic 5, with the top words "your vote ballot polls", spikes almost every november and is nonexistent the rest of the year, as does topic 4, which is about veterans. Topic 9, which is about Ukraine, only appears starting in February 2022, and topic 2, which is about COVID-19, sees its biggest spikes during the winter of 2020-21 and the Omicron wave beginning in late 2021. In general, all of these topics spike in the winter, and occur barely if at all during the rest of the year.


___________

# Multiclass Classification: What can I predict using tweets?

```python

```

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


I scraped tweets from several accounts. Each set of tweets is in its own .csv file, so the below code loads in each .csv and appends it to a list of dataframes, which is then concatenated into one pandas dataframe.

```python
#!dvc pull
```

```python
import glob
import os
import pandas as pd
pd.options.display.max_columns = None
```

```python
data_full.to_csv('tweets.csv', index = False)
```

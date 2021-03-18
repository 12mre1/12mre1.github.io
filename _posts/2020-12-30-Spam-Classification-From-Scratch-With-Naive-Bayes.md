---
layout: post
title: "Spam Classification From Scratch with Naive Bayes"
date: 2020-12-30
---

## Spam Classification From Scratch With Naive Bayes

_Prerequisite Math: Calculus (Derivatives), Intermediate Statistics (Prior, Likelihood, Posterior)_

_Prerequisite Coding: Python (Pandas)_

One of the all time popular uses for classification algorithms is labelling email messages as Spam(bad) or Ham(good). Doing so allows us to automatically filter out messages we know are not actually important. But determining what exactly separates the ham from the spam is not something we can easily define. Although humans are great at determining whether a given email is spam, the goal with Machine Learning is to idenitify the __decision rule__ that separates the classes, so that any message which is spam can be filtered without humans having to read it.

In this post, i'm going to implement a very simple model called __Naive Bayes__, which classifies emails based only on the words in their message. I'll be
using the python language not only to run the model itself, but also to preprocess the dataset. To train the model, I'll be using a dataset of emails 
created for [this Kaggle competition](https://www.kaggle.com/uciml/sms-spam-collection-dataset?select=spam.csv). You can freely download the data in csv format
(in fact I encourage you to do so and follow along with this post). To run the necessary packages, i'm using a google colab notebook, but you can easily 
use this code on your local machine, or in an IDE (e.g. Anaconda). To begin, I load the data into colab. The following code is unique to colab, and just lets
me upload the csv file from my drive:
```python
# Load the dataset
from google.colab import files
data_to_load = files.upload()
> Saving spam.csv to spam.csv
```
After loading the necessary libraries, the first thing I'm going to do is convert the data from Colab's memory into a `pandas` dataframe. This is easy to do using the `IO` package. At the same time, I can see that when I do this, the dataframe is not quite what I want. We are left with a few extra columns that I would like to remove 
(there should only be two). The columns should have proper names, and I would also like the labels to be numeric, instead of strings. It is also good practice to
examine the proportions of classes in your dataset. I accomplish all these steps in a single function, `clean_email_df` defined below:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re


# Convert to pandas DF
df = pd.read_csv(io.BytesIO(data_to_load['spam.csv']), encoding = 'latin1')
print('Before cleaning:\n',df.head(5))

# A bit of cleaning
def clean_email_df(DF):
  '''Remove extra columns, give proper col names,
  and re-adjust labels for binary classification. Also
  show the balance between classes'''
  # Remove unwanted columns
  data = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

  # rename other columns while we're at it
  data.columns = ['label', 'message']
  print('Now we keep just what we need:\n', data.head())

  # Change spam = 1, ham = 0
  data.label = (data.label == 'spam').astype(float)

  # Compute proportion of spam observations
  spam_proportion = round(data.label.sum()/len(data), 2)
  print('\n\nSpam labels make up approximately {} percent of the dataset\n\n'.format(spam_proportion))

  # return the clean DF
  return data

data = clean_email_df(df)
print('After cleaning:\n', data.head(5))

>Before cleaning:
>      v1  ... Unnamed: 4
>0   ham  ...        NaN
>1   ham  ...        NaN
>2  spam  ...        NaN
>3   ham  ...        NaN
>4   ham  ...        NaN
>
>[5 rows x 5 columns]
>Now we keep just what we need:
>   label                                            message
>0   ham  Go until jurong point, crazy.. Available only ...
>1   ham                      Ok lar... Joking wif u oni...
>2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
>3   ham  U dun say so early hor... U c already then say...
>4   ham  Nah I dont think he goes to usf, he lives aro...
>
>
>Spam labels make up approximately 0.13 percent of the dataset
>
>
>After cleaning:
>    label                                            message
>0    0.0  Go until jurong point, crazy.. Available only ...
>1    0.0                      Ok lar... Joking wif u oni...
>2    1.0  Free entry in 2 a wkly comp to win FA Cup fina...
>3    0.0  U dun say so early hor... U c already then say...
>4    0.0  Nah I dont think he goes to usf, he lives aro...
```
We can see that we are left only with the binary label (1 = spam, 0 = ham), and the text of the message. Note that it is also good practice to give the minority
class the positive label in binary classification. We can see that the ham/spam breakdown of our dataset is 87/13. This is important to remember as we
fit models, because a classifier that predicts ham for every email would get a training accuracy of 87% (thus what we define as 'good' measures of performance
must take this baseline into account).

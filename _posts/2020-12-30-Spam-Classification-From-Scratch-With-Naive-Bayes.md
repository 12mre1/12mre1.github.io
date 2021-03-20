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
use this code on your local machine, or in an IDE (e.g. Anaconda). Here is a rough outline of the concepts I cover:

1. Tokenization
2. Converting tokens to lowercase
3. Removing punctuation and special characters
4. Removing stopwords
5. Stemming and Lemmatization
6. Removing blank tokens
7. Splitting data into train/test
8. Fitting the Naive Bayes Model
9. Evaluating prediction on the test set

To begin, I load the data into colab. The following code is unique to colab, and just lets
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
```
<span style="background-color: #d7f5f5">
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

</span>

We can see that we are left only with the binary label (1 = spam, 0 = ham), and the text of the message. Note that it is also good practice to give the minority
class the positive label in binary classification. We can see that the ham/spam breakdown of our dataset is 87/13. This is important to remember as we
fit models, because a classifier that predicts ham for every email would get a training accuracy of 87% (thus what we define as 'good' measures of performance
must take this baseline into account).

To get a sense of what these email messages look like, let's define a function that prints a random message (along with its label). Because I do not
set a random seed within this function, running it multiple times will give different results:

```python
# Print a random message

# This function will print a random email message
def print_random_message(df):
  i = np.random.randint(len(df))
  message = df.iloc[i,1]
  label = df.iloc[i,0]
  print('Here is one of the messages:\n\n', message)
  print('\nIt is labelled as:\n\n',label)

# call the function
print_random_message(data)

>Here is one of the messages:
>
> They said if its gonna snow, it will start around 8 or 9 pm tonite! They are predicting an inch of accumulation.
>
>It is labelled as:
>
> 0.0
```
This particular message is ham. We can see that the message comes as a __string__, which is just a long sequence of characters.
Note that is the natural language processing setting, we treat whitespace as its own character. So we can also think about these
strings as sequences of words, separated by whitespace. However, the string on its own is not a very useful predictor of ham and
spam, because almost every string is different. If we're just measuring similarity between strings by treating them as sequences 
of characters, most of any string is noise (ie not a useful signal for our classification). There are models that use language
at the character level to predict, for example, the next word in a sentence, or next sentence in a paragraph (such models are 
called, not surprisingly, __language models__). But in this case, our individual unit will be words, or word-like objects. We 
call these objects __tokens__, and most are just words (ie `are`, `just`, and `most` are tokens), but occasionally we treat other 
special character sequences as their own tokens, depending on the text (ie `Government of Canada` might be a single token).

Converting running text from string format into lists of tokens is called __tokenization__. Splitting running text this way 
may see like a complicated task, but there are several python packages that do this automatically. The one I'll use is called
`nltk`. The following code converts our running text into lists of tokens:
```python
#### Tokenize ####

import nltk
# Download the tokenizer
nltk.download('punkt')

# Create a new column in our DF that contains token lists instead of raw text
data['tokens'] = data['message'].apply(nltk.word_tokenize)

print(data['tokens'].head(5))

>[nltk_data] Downloading package punkt to /root/nltk_data...
>[nltk_data]   Unzipping tokenizers/punkt.zip.
>0    [Go, until, jurong, point, ,, crazy.., Availab...
>1             [Ok, lar, ..., Joking, wif, u, oni, ...]
>2    [Free, entry, in, 2, a, wkly, comp, to, win, F...
>3    [U, dun, say, so, early, hor, ..., U, c, alrea...
>4    [Nah, I, do, nt, think, he, goes, to, usf, ,,...
>Name: tokens, dtype: object
```
We can see that our dataframe now has an additional column containing the list of tokens from the running text. Another common preprocessing
step is to convert our tokens into lowercase only. This prevents our model from treating two tokens as separate just because one of them
appears at the beginning of a sentence, or in a title, for example. Note that in some cases, capitalization may be an indicator of the 
position, and you may not want to convert all tokens to lowercase. But for our modelling stategy, the position of the token does not matter
(only its presence). I'll talk more about this later when I introduce the Naive Bayes model, but for now, notice that the following code
converts our tokens to lowercase:
```python
##### Convert tokens into lowercase ####

def lower_tokens(df, colname):
  '''Convert a df column of tokens to lowercase.
  Inputs: df - the dataframe containing the tokens
          col_name - the name of the column containing
          the tokens
  Output: A new df with lowercase tokens appended as 
  separate column.'''

  # Create a list of lists with what we want
  lowercase_tokens = []

  # For every row in DF
  for row in df[colname]:
    # Add the lowercase version of token to row list
    lowercase_tokens.append([t.lower() for t in row])

  # add the new info to our df
  df['lowercase_tokens'] = lowercase_tokens
  # print some lowercase token lists
  print(data['lowercase_tokens'].head(5))
  # return new df
  return(df)

# Execute function
data = lower_tokens(data, 'tokens')

print(list(data))

>0    [go, until, jurong, point, ,, crazy.., availab...
>1             [ok, lar, ..., joking, wif, u, oni, ...]
>2    [free, entry, in, 2, a, wkly, comp, to, win, f...
>3    [u, dun, say, so, early, hor, ..., u, c, alrea...
>4    [nah, i, do, nt, think, he, goes, to, usf, ,,...
>Name: lowercase_tokens, dtype: object
>['label', 'message', 'tokens', 'lowercase_tokens']
``` 

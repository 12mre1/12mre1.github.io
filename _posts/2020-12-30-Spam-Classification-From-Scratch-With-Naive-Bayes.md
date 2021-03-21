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

#--------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
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

#--------------------------------------------------------------
>0    [go, until, jurong, point, ,, crazy.., availab...
>1             [ok, lar, ..., joking, wif, u, oni, ...]
>2    [free, entry, in, 2, a, wkly, comp, to, win, f...
>3    [u, dun, say, so, early, hor, ..., u, c, alrea...
>4    [nah, i, do, nt, think, he, goes, to, usf, ,,...
>Name: lowercase_tokens, dtype: object
>['label', 'message', 'tokens', 'lowercase_tokens']
``` 
Now all our tokens have been lowered. Another common preprocessing step is to remove punctuation marks from the tokens. When we
split the running text by whitespace, any punctuation will have been appended to the word it is next to. Unless we deal with this,
a word used in the middle of a sentence (e.g. `green`) will be treated differently than the identical word used at the end (`green!`).
This is not so difficult to do, if we use a simple regular expression. __Regular expressions__ are sequences of characters designed to
extract patterns from larger strings. Although I will not go into much detail here, you can learn all about them at the tutorial 
website [regexone](https://regexone.com/). I recommend completing the entire set of exercises. You'll be a regex expert in no time.

The following code uses the regex symbol `\w` to denote all alphanumeric characters (letters, numbers and underscores). Using the condition
that I keep only those types of characters, I drop all forms of punctuation. Note that in some cases, there may be some special characters
you want to keep (e.g. `#` or `@` in twitter text). In cases like that, you may want to extract special text first, then remove punctuation
in this way. The `re` python library has a great function for substituting based on regex: 
```python
##### Let's remove punctuation #####

def remove_punct(df, colname):
  '''Remove punctuation from a col of tokens.
  Inputs: df - the dataframe containing the tokens
          colname - the name of the column containing
          the tokens
  Output: A new df with punctuationless tokens appended as 
  separate column.'''
  # Instantiate list of row lists
  tokens_no_punct = []
  # Create a list of lists with what we want
  for row in df[colname]:
    tokens_no_punct.append([re.sub('[^\w\s]','', t) for t in row])

  # add the new info to our df
  df['tokens_no_punct'] = tokens_no_punct
  # print some new token lists 
  print(df['tokens_no_punct'].head(5))

  return df

# Execute function
data = remove_punct(data, 'lowercase_tokens')

#--------------------------------------------------------------------
>0    [go, until, jurong, point, , crazy, available,...
>1                   [ok, lar, , joking, wif, u, oni, ]
>2    [free, entry, in, 2, a, wkly, comp, to, win, f...
>3    [u, dun, say, so, early, hor, , u, c, already,...
>4    [nah, i, do, nt, think, he, goes, to, usf, , h...
>Name: tokens_no_punct, dtype: object
```
Excellent! Now we our tokens are almost completely clean. Notice that several words are repeated quite often (e.g. a , u). This
is not necessarily a bad thing - sometimes key words will be crucial in classifying the email. However, many words we see often
in spam and ham messages are only repeated often because they are used the most frequently in the english language. These are 
words you might be able to guess even without looking at the tokens (e.g. _the_, _it_, _be_, _a_). We call these types of tokens
__stopwords__, and because they tend to appear often in all observations regardless of class, they have almost no predictive 
power. It is common to remove them from our observations, and that is what we will do here. Rather than trying to generate our
own list of stopwords, we can use a freely available one that comes with `nltk`. This given list has almost 200 stopwords, and 
in this case I use a subset of this:
```python
######### Remove Stopwords #########

##### Time to remove Stopwords #####

from nltk.corpus import stopwords
nltk.download('stopwords')
# print the top 75 most popular english words
sw = stopwords.words('english')[:75]

# I converted to np array for better printing
print(np.array(sw).reshape((15,5)))

def remove_sws(df, colname, sw_list):
  tokens_no_sw = []
  for row in df[colname]:
    tokens_no_sw.append([w for w in row if w not in sw_list])
  # Add column to df
  data['tokens_no_sw'] = tokens_no_sw

  # Print some examples
  print(df['tokens_no_sw'].tail(5))

  return df

# Execute function

data = remove_sws(data, 'tokens_no_punct', sw)

#---------------------------------------------------------------
>[nltk_data] Downloading package stopwords to /root/nltk_data...
>[nltk_data]   Unzipping corpora/stopwords.zip.
>[['i' 'me' 'my' 'myself' 'we']
> ['our' 'ours' 'ourselves' 'you' "you're"]
> ["you've" "you'll" "you'd" 'your' 'yours']
> ['yourself' 'yourselves' 'he' 'him' 'his']
> ['himself' 'she' "she's" 'her' 'hers']
> ['herself' 'it' "it's" 'its' 'itself']
> ['they' 'them' 'their' 'theirs' 'themselves']
> ['what' 'which' 'who' 'whom' 'this']
> ['that' "that'll" 'these' 'those' 'am']
> ['is' 'are' 'was' 'were' 'be']
> ['been' 'being' 'have' 'has' 'had']
> ['having' 'do' 'does' 'did' 'doing']
> ['a' 'an' 'the' 'and' 'but']
> ['if' 'or' 'because' 'as' 'until']
> ['while' 'of' 'at' 'by' 'for']]
>5567    [2nd, time, tried, 2, contact, u, u, won, å750...
>5568      [will, ì_, b, going, to, esplanade, fr, home, ]
>5569    [pity, , , in, mood, , so, , any, other, sugge...
>5570    [guy, some, bitching, acted, like, d, interest...
>5571                             [rofl, , true, to, name]
>Name: tokens_no_sw, dtype: object
```
You can see the list of stopwords above, along with some of our tokenized messages once those stopwords have been removed.

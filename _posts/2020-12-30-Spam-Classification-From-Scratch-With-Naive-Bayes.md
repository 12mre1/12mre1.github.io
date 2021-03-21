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
You can see the list of stopwords above, along with some of our tokenized messages once those stopwords have been removed. But we're nto quite done yet.
I might not be obvious here, but in any given corpus of text, there will be many words that actually have identical meanings, but are only different 
because of tense (run, ran running) or because of plurality (dog, dogs). Ideally, we'd like to merge these various forms by converting them into 
identical tokens. However english is a very irregular language, and this is not a very easy task. Two common approaches for doing this are called
stemming and lemmatization. 

__Stemming__ is the act of converting a word to its root by removing the word's suffix. For example, _running_ would be converted to _run_, _apples_
to _apple_, and _digestion_ to _digest_. However, given the oddities and irregularities of the english language, there is no universal set of rules
for pefectly accomplishing this. Thus, most stemmers are large, hand-compiled databases of words/tokens that map irregular forms into their roots.
However, many relationships can be missed (ie _knives_ to _knife_), and sometimes when rules are used, they don't quite give a perfect result (ie 
_babies_ becomes _babi_). Perhaps the most common stemmer is called the __Porter stemmer__, named for its creator. Although not perfect, using a 
stemmer can give significant improvement further down the line.  

__Lemmatizing__ is a similar process that takes this a step further. Instead of converting tokens to their roots, it converts them to their __lemmas__,
or dictionary words. This is particularly useful for converting different verb tenses into single tokens. For example (run, ran, running, runs) would
all convert to run. This is typically stronger than stemming alone, however most lemmatizers are also hand-compiled databases of words. Thus, not only
are they difficult to create, but they are not perfect. However, just like stemming, using a lemmatizer will often give significant improvement. Just
like stemmers, we have several options for our lemmatizer, but one of the most common ones is called the _WordNet_ lemmatizer, and that is what I will
use here. The following code stems and lemmatizes our token lists:
```python
######### Stemming and Lemmatization ###########

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# download lemmatizer
nltk.download('wordnet')

# Instantiate stemmer and lemmatizer objects
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

### Stemming our dataset (Snowball Stemmer) ###

def stem_tokens(df, colname):
  '''Convert tokens into their stems (roots)
  Inputs: df - The DataFrame of tokens.
          colname - The column name with the tokens.
  Output: Dataframe with new column of stemmed tokens.'''

  # Instantiate list of rows
  stemmed_tokens = []
  # Loop for each row in df to stem token by token
  for row in df[colname]:
    stemmed_tokens.append([stemmer.stem(t) for t in row])

  # Add stemmed tokens to df
  df['stemmed_tokens'] = stemmed_tokens

  # Print results
  print('\nAfter stemming:\n\n', df['stemmed_tokens'].head(3))

  return df

##### Lemmatize the Dataset #####

def lemmatize_tokens(df, colname):
  '''Lemmatize tokens (convert to dictionary word)
  Inputs: df - The DataFrame of tokens.
          colname - The column name with the tokens.
  Output: Dataframe with new column of lemmatized tokens.'''

  # Instantiate list of rows
  lem_tokens = []
  for row in df[colname]:
    lem_tokens.append([lemmatizer.lemmatize(t) for t in row])

  # Add lemmatized tokens to df
  df['lem_tokens'] = lem_tokens
  # Print results
  print('\nAfter Lemmatizing:\n\n',df['lem_tokens'].head(3))

  return df

# Execute lemmatizing and stemming

data = stem_tokens(data, 'tokens_no_sw')
data = lemmatize_tokens(data, 'stemmed_tokens')

print(data.head(3))

#-----------------------------------------------------------------

>[nltk_data] Downloading package wordnet to /root/nltk_data...
>[nltk_data]   Unzipping corpora/wordnet.zip.
>
>After stemming:
>
>0    [go, jurong, point, , crazi, avail, onli, in, ...
>1                     [ok, lar, , joke, wif, u, oni, ]
>2    [free, entri, in, 2, wkli, comp, to, win, fa, ...
>Name: stemmed_tokens, dtype: object
>
>After Lemmatizing:
>
>0    [go, jurong, point, , crazi, avail, onli, in, ...
>1                     [ok, lar, , joke, wif, u, oni, ]
>2    [free, entri, in, 2, wkli, comp, to, win, fa, ...
>Name: lem_tokens, dtype: object
>   label  ...                                         lem_tokens
>0    0.0  ...  [go, jurong, point, , crazi, avail, onli, in, ...
>1    0.0  ...                   [ok, lar, , joke, wif, u, oni, ]
>2    1.0  ...  [free, entri, in, 2, wkli, comp, to, win, fa, ...
>
>[3 rows x 8 columns]

```
This is good. Our token list are almost as clean as they can be without us treating individual instances (for a perfectly clean dataset, you will 
probably have to do this with hand-built lists of token mappings). One last step we will take is to remove blank tokens. These are just a consequence
of our earlier removal of punctuation and special characters. We didn't tehnically remove them, rather we replaced them with whitespace. So for tokens
that were just a special character or punctuation mark, we are now left with tokens that are pure whitespace (blanks).

The code below removes these blank tokens. At the same time, we've been keeping each preprocessing step as its own column. There's no need to keep
these intermediate series, so I drop them, leaving only label, raw text, and the final processed tokens.
```python
##### Remove blank tokens (empty strings) #####
def remove_blank_tokens(df, colname):
  '''Remove blank tokens from df'''
  # Instantiate  list of non-blank tokens
  no_blanks = []
  # Loop through data
  for row in df[colname]:
    no_blanks.append([t for t in row if t != ''])
  df['tokens'] = no_blanks
  
  return df

data = remove_blank_tokens(data, 'lem_tokens')

##### Drop intermediate columns #####
def remove_inter_cols(df, keep_col):
  '''Remove intermediate cols from df'''
  data = df[['label', 'message', keep_col]]
  return data

data = remove_inter_cols(data, 'tokens')

print(data.head(3))

#-------------------------------------------------
>   label  ...                                             tokens
>0    0.0  ...  [go, jurong, point, crazi, avail, onli, in, bu...
>1    0.0  ...                       [ok, lar, joke, wif, u, oni]
>2    1.0  ...  [free, entri, in, 2, wkli, comp, to, win, fa, ...
>
>[3 rows x 3 columns]
```
Now our dataset is pretty clean. Note that we could painstakingly continue to clean individual tokens on a case-by-case basis (if this were a project or 
application going into production, you should definitely do this), but for our purposes, the steps we have taken will be enough to achieve good performance.
Up until now, I've been processing the data as one set, but, as with any model-fitting technique, now is the time to split the data into training and 
test sets (and possibly a validation set if you're experimenting with several modelling options). Packages like `scikit-learn` have excellent built-in
functions to split data this way, and I will use that here.
```python
import numpy as np
from math import *
from sklearn.model_selection import train_test_split

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(data['tokens'], data['label'], test_size = 0.1)
print(x_train.head(5))
print(y_train.head(5))
#------------------------------------------------------------------------------------

>526     [today, s, offer, claim, ur, å150, worth, disc...
>4834    [oh, rite, well, im, with, best, mate, pete, w...
>1253    [mum, say, wan, to, go, then, go, then, can, s...
>3595             [good, morn, princess, happi, new, year]
>4217     [actual, m, wait, 2, week, when, start, put, ad]
>Name: tokens, dtype: object
>526     1.0
>4834    0.0
>1253    0.0
>3595    0.0
>4217    0.0
>Name: label, dtype: float64
```

Now we're ready to fit the model. But first, let's look at exactly how the Naive Bayes model works.

## Naive Bayes Classification

We're interested in classifying our documents (lists of tokens) into two classes. Let's call these classes \\( S \\) and \\( H \\), for Spam and Ham. How might we use our tokens to predict the class?
- __Idea__: Let's try to model the probabilities of each class __conditional on the tokens in the document__. Suppose our message has D tokens. It might not be obvious how to do this, but we can rely on Bayes Rule:

{% raw %}

$$ Pr(S|w_1 , \cdots , w_D)  = \frac{Pr(w_1 , \cdots , w_D | S) Pr(S)}{Pr(w_1 , \cdots , w_D)} \propto Pr(w_1 , \cdots , w_D | S) Pr(S) $$

{% endraw %}

So the probability that our message is spam is equal to the probability that we get these words given the message is spam, multiplied by the probability of any message being spam (this is pretty intuitive). We call the left term the __likelihood__, and the right term the __prior__. 

-__Problem__: How in the world do we estimate \\( Pr(w_1 , \cdots , w_D \vert S) \\)?

-__Solution__: We use a __naive__ assumption. Let's assume that words are independent of each other conditional on class (this is obviously not true in reality, but it makes our lives easier). Then we can rewrite our probability:

{% raw %}

$$ Pr(S|w_1 , \cdots , w_D)  = \propto \Pi_{i=1}^{D}Pr(w_i | S) Pr(S) $$

{% endraw %}

Now we have something we can estimate. Assume V is the set of all words in your training corpus. The __Naive Bayes__ approach does the following:

__i)__ Estimate \\( \hat{Pr}(S) = \frac{N_s}{N} \\), where \\( N_s \\) is the number of documents labelled as S. Similarly estimate \\( \hat{Pr}(H) = 1 - \hat{Pr}(S) \\).

__ii)__ For every token in every document, estimate \\( \hat{Pr}(w_i \vert S) = \frac{count(w_i \vert S)}{\Sigma_{w \in V} count(w_i \vert S)} \\), where \\( count(w_i\vert S) \\) is the number of times token  \\( w_i \\) appears __in all spam documents__, and  \\( \Sigma_{w \in V} count(w_i \vert S) \\) is the total number of words in all spam documents. Do a similar computation for Ham documents and tokens. 

__iii)__ Once we have these probabilities, we can compute \\( \hat{Pr}(S \vert w_1 , \cdots , w_D) \\) and \\( \hat{Pr}(H \vert w_1 , \cdots , w_D) \\) for any new document. Then we simply label that document as Spam or Ham, depending on which probability is larger.

__One small problem__: If a word in the test corpus does not appear in the training corpus, it will have a count (and thus a probability) of zero. This will make the entire \\( \hat{Pr}(S \vert w_1 , \cdots , w_D) = 0 \\), even if other words in the document still have positive probability. To deal with this, we use __smoothing__, assigning every word an arbitrarily low probability. There are different ways to do this, but here is a technique called __Laplace Smoothing__:

{% raw %}

$$\hat{Pr}(w_i|S) = \frac{count(w_i|S) + 1}{\Sigma_{w \in V} (count(w_i|S) + 1)}$$

{% endraw %}


This probability estimation approach is called __Bag of Words (BOW)__, because we treat classes and documents as collections of words, where order does not matter.
Note that the estimators above (before smoothing) are simply the maximum likelihood estimators of the word probabilities, if we assume the word counts in sentences
follow a __multinomial distribution__. Thus, this approach is called __Multinomial Naive Bayes__, and could easily be extended for the case of more than 2 classes.
Although there are packages in python that can fit this kind of model automatically (one great example is the `MultinomialNB()` transformer from `scikit-learn`), 
just like the preprocessing steps, I will build this from scratch so you can see how it works.

The goal of the next few functions is to separate the documents (tokens lists) into bags of words (1 for ham, 1 for spam). I start by computing the two class priors, using the formula given earlier. Technically we're computing log probabilities, because it is easier to work with sums than products:
```python
###### Create Priors ######

###### Create Priors ######

def get_log_priors(X, y):
  '''Compute prior probabilities for each class'''
  assert len(X) == len(y)
  # Total number of document
  n_doc = len(X)
  # Spam documents
  n_spam = len(X[y == 1.0])
  # Ham documents
  n_ham = len(X[y == 0.0])

  # Quick sanity check
  assert n_ham + n_spam == n_doc

  # Vocab of a class is the union of all words of class C
  spam_prior = log(n_spam/n_doc)
  ham_prior = log(n_ham/n_doc)

  return spam_prior, ham_prior

# Get prior class probabilities
spam_prior, ham_prior = get_log_priors(x_train, y_train)
```
Next I want to turn my pandas dataframe of messages into bags of words. It no longer makes sense to store these two collections of tokens in DF format, so
the resulting function returns two lists of tokens. One list contains all tokens in all spam documents, and the other contains all tokens in all ham documents. Note that I only want the tokens from the training data, since the probabilities generated with the bags of words cannot contain information from the test data (this would be an example of __data leakage__, and it would make our test error useless as a measure of generalization).
```python
##### Get BOWs for each class ######

def get_bags_of_words(X, y):
  '''Convert df into bags of words for each class.
  Inputs: X - features (tokens)
  y - labels (binary).
  Returns: One list of tokens for each class.'''
  # Separate ham and spam into two dfs
  spam_obs = X[y == 1.0]
  ham_obs = X[y == 0.0]
  print('Number of spam and ham observations:',len(spam_obs), len(ham_obs))

  # Single list of all spam tokens (includes duplicates)
  spam_token_lists = spam_obs.values.flatten().tolist()
  spam_bow = [t for l in spam_token_lists for t in l]

  # Single list of ham tokens (includes duplicates)
  ham_token_lists = ham_obs.values.flatten().tolist()
  ham_bow = [t for l in ham_token_lists for t in l]
  
  print('Size of spam and ham bags of words:',len(spam_bow), len(ham_bow))

  return spam_bow, ham_bow

s_bow, h_bow = get_bags_of_words(x_train, y_train)

#---------------------------------------------------------------

>Number of spam and ham observations: 676 4338
>Size of spam and ham bags of words: 13523 45373
```
So now we have two bags of words containing all tokens in each of the two classes. Our next step is to convert these larger token lists into
counts of tokens. A convenient data structure to store this information is the _dictionary_, with keys being tokens, and values being their counts. The following code produces token count dictionaries for both ham and spam classes:
```python
###### Convert tokens into counts ######
def get_token_counts(spam_bow, ham_bow):
  '''Convert class token lists to dictionaries of counts'''
  # Create counts of spam tokens
  spam_counts = dict()
  for t in spam_bow:
    spam_counts[t] = spam_counts.get(t,0)
    spam_counts[t] += 1
  
  # Create counts of ham tokens
  ham_counts = dict()
  for t in ham_bow:
    ham_counts[t] = ham_counts.get(t,0)
    ham_counts[t] += 1

  print(len(spam_counts), len(ham_counts))
  return spam_counts, ham_counts

s_counts, h_counts = get_token_counts(s_bow, h_bow)

# Print a few tokens from each class
print('Five token counts from Spam:\n',list(s_counts.items())[:5])
print('Five token counts from Ham:\n',list(h_counts.items())[:5])

#---------------------------------------------------------------------

>2585 5662
>Five token counts from Spam:
> [('today', 35), ('s', 78), ('offer', 39), ('claim', 106), ('ur', 125)]
>Five token counts from Ham:
> [('oh', 104), ('rite', 17), ('well', 100), ('im', 69), ('with', 247)]
```
We can see (intuitively) that some words appear much more frequently than others. Now we want to convert these counts into probabilities 
according to the maximum likelihood estimate given earlier. The following code converts the dictionary of counts to a dictionary of probabilities.
To give you a sense of what this looks like, I print the first 10 entries (alphabetically) from the spam dictionary.
```python
from math import *
import json 

#### Convert BOW counts into probabilities ####
def get_token_probs(count_dict):
  '''Convert docs (token lists) to class probs'''
  prob_dict = {}
  # Extra term is for laplace smoothing (denominators of posterior estimates)
  sum_counts = sum(count_dict.values()) + len(count_dict.keys())
  for key in count_dict.keys():
    prob_dict[key] = log((count_dict[key] + 1)/sum_counts)

  # Define a default probability for unseen test tokens
  default_prob = log(1/sum_counts)
  # return the dictionary of log probabilities
  return prob_dict, default_prob

# Create prob dictionaries for each class
spam_probs, default_prob_spam = get_token_probs(s_counts)
ham_probs, default_prob_ham =  get_token_probs(h_counts)

# Print some spam probs to see (10)
d = dict(list(spam_probs.items())[0:10])
print(json.dumps(d, indent=4, sort_keys=True))

#----------------------------------------------------------------------

>{
>    "2": -4.584781932701026,
>    "comp": -7.388142313607561,
>    "cup": -7.898967937373551,
>    "entri": -6.4718515817334055,
>    "fa": -8.081289494167507,
>    "free": -4.412612747371089,
>    "in": -5.456620902004347,
>    "to": -3.270732478454464,
>    "win": -5.665375715866457,
>    "wkli": -6.982677205499396
>}
```
Recall these are log probabilities, so the numbers will be negative, but a larger number indicates a higher frequency. 

Now that we have our probabilities (generated based on the training data), we can use them to compute probabilities for entire
observastions in both the training and testing sets. The posterior class probabilities for each email will simply be the sum of 
the log probabilities of each token in each email, plus the class prior. Remember the prior must also be computed based on the training 
data. We compute probabilities for the ham and spam classes, and the prediction will simply be the higher of the two posterior class probabilities:
```python
#### Convert observations in original df into probabilities ####
def get_doc_probs(X, spam_probs, ham_probs, def_prob_spam, def_prob_ham):
  '''Convert docs (token lists) to posterior probabilities.
  Input: X - The dataframe of features
  spam_probs - The dictionary of probabilities for spam tokens.
  ham_probs - The dictionary of probabilities for ham tokens.
  def_prob_ham - The default prob of ham class for unseen tokens
  def_prob_spam - The default prob of spam class for unseen tokens
  Output: A dataframe of predictions, with both ham and spam post. probs'''

  # Instantiate probabilities for each doc for each class
  p_spam = []
  p_ham = []
  for row in X:
    # Compute log likelihood probabilities for each token in each doc
    token_probs_spam = [spam_probs.get(t, def_prob_spam) for t in row]
    token_probs_ham = [ham_probs.get(t, def_prob_ham) for t in row]

    # Append posterior probabilities (likelihood + prior) 
    p_spam.append(sum(token_probs_spam) + spam_prior)
    p_ham.append(sum(token_probs_ham) + ham_prior)

  print(len(p_ham))
  print(len(p_spam))
  # Add posterior probabilities as columns
  df = pd.DataFrame(p_spam, columns = ['p_spam'])
  print(df.columns)
  df['p_ham'] = p_ham
  df['prediction'] = (df['p_spam'] > df['p_ham']).astype(float)

  return df

# Predict on training data
train_preds = get_doc_probs(x_train, spam_probs, ham_probs, default_prob_spam, default_prob_ham)

# Predict on testing data
train_preds = get_doc_probs(x_test, spam_probs, ham_probs, default_prob_spam, default_prob_ham)
```
Now we have two data frames (one for training, and one for testing data), each with the probabilities for each class, and the associated prediction.
We can use these dataframes to compute simple training and testing accuracy, which is defined (in this setting) as the proportion of predictions 
(in both classes combined) that are correct.
```python
# Generate training error
print('---------- Training Accuracy ----------')
train_error = sum((train_preds['prediction'] == y_train))/len(y_train)
print(train_error)

# Generate testing error
print('---------- Test Accuracy ----------')
test_error = sum((test_preds['prediction'] == y_test))/len(y_test)
print(test_error)

#------------------------------------------------------------------
>---------- Training Error ----------
>0.9900279218189071
>
>---------- Test Error ----------
>0.9587813620071685
```
This is pretty great! We can see that our training accuracy is slightly higher than the testing accuracy, which is an indication of overfitting (ie the algorithm
is capturing some patterns unique to the data but not true for the spam/ham relationship in general). However, given the class imbalance (87% were ham), a
classification accuracy of almost 96% is quite good. Some state of the art models (ie neural networks) could probably achieve close to 100% accuracy on the 
test set, however given how simple our model is, this performance is promising, and serves as a strong baseline. Note also that more preprocessing before running
this model would probably also improve performance slightly, however given the many steps we already took, and the diminishing returns of further preprocessing,
that additional improvement would likely not have been worth it.

## Further Reading

- Perhaps the best text on NLP in general is [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Jurafsky and Martin. It contains an entire chapter dedicated to Naive Bayes, and also addresses many of the preprocessing steps done here.

- The `Scikit-learn` documentation for [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) is certainly worth reading.

- [This](https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/) is a similar article, also providing python code.
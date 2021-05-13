---
layout: post
title: "The Surprising Effectiveness of K Nearest Neighbours"
date: 2020-12-10
---

## The Effective Simplicity of KNN

_Prerequisite Math: Calculus (Derivatives), Intermediate Statistics (Prior, Likelihood, Posterior)_
_Prerequisite Coding: Python(Numpy, Matplotlib)_

In today's post, I'll talk about one of the oldest and simplest Machine Learning algorithms, called __K-Nearest Neighbors__. It is a supervised learning technique, meaning in order to use it, we must have a data that is labelled (ie contains human-specified target values for the variable we wish to predict). I'll go into more detail shortly, but the basic assumption behind KNN is that points similar (close to each other) in feature space should have similar target values. I'll go through the algorithm in detail, discussing pros and cons while showing a detailed example and providing python code along the way. 

The dataset I'll be using today is the famous (at least among data scientists) __Titanic Dataset__, which you can find [here](https://www.kaggle.com/c/titanic). The entire dataset contains some data on each passenger (thousands of observations), but since I just want to illustrate the algorithm, I won't spend much time analyzing or cleaning it. Note than, in practice and in competition, cleaning and engineering your data is probably the most important step, and you should take great care in doing so. To start, I load in the data (I'm using a colab notebook, but any IDE or even the command line would work):

```Python 
# Load in the data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

from google.colab import files
uploaded = files.upload()

df = pd.read_csv(io.StringIO(uploaded['train.csv'].decode('latin-1')))
df = df.dropna(axis = 0)
# Take a subset of two columns
X, y = df[['Age', 'Fare']].to_numpy(), df['Survived'].to_numpy()

# X = X[~np.isnan(X).any(axis = 1)]
y = y.reshape((X.shape[0], 1))

print(X.shape)
print(y.shape)
```

So there's a few things going on here. First, I load the dataset, which is actually a subset in itself (891 passengers with labels). In our case, there are 11 different features, but with distance-based algorithms, we need numerical features. Since visualizing is easier in 2D, I decided to use just Age and Fare as my two features. Also, I remove any row for which one or more of the feature values is missing (this is true for many of them). In the end, this leaves me with 183 observations (so N = 183), which I convert to numpy array format. Notice that our label is binary (0 = Died, 1 = Survived).

## The Algorithm

As mentioned above, the key idea behind KNNs is that points with the same label should be near each other in feature space. To measure nearness, it is common to use a specific distance measure between any two points. So suppose we have a set of features, \\( X_{N \times D}\\), where \\( N \\) is the number of data points (rows), and \\( D \\) is the number of features (columns). You can think of this as a __training set__, although in K Nearest Neighbors classification, there really is not much training; we just use the existing data to classify new points. Along with our features, we also have class labels, \\( y_{N \times 1} \\). Suppose we have a new query point (ie one observation of a set of features, or an \\( 1 \times m \\) vector), and we want to predict it's class label. The K-Nearest Neighbors classification algorithm does the following:

1. Compute the distance between our query point, and each point in the training set (n distances total).
2. Select the k-closest points; in other words, select the points corresponding to the k smallest distances.
3. Assign to our query point the label that occurs most frequently among the k nearest neighbors.

At its core, that is essentially all there is too it. Simple right? In fact, it's surprisingly effective despite its simplicity, as we'll soon see.

## The Distance Metric Matters

Now I've told you that the K-Nearest Neighbors are chosen by distance to the query point, but I've not specified exactly which distance you should use. There's no right answer here; no distance is proven to be consistently better than any other, but there are a few common choices you might consider. Below I provide the mathematical formula as well as the python code for several common distance functions. For query point \\( w \\):

__Manhattan Distance__ (also called L1 Norm):

$$  \sum_{k=1}^{m} \mid x_k - w_k \mid $$

__Euclidean Distance__ (also called L2 Norm):

$$ ( \sum_{k=1}^{m} (x_k - w_k )^2 )^{1/2} $$

__Minkowski Distance__ (a generalization of the two distances above). Note that when \\( p \rightarrow \infty \\), we simply get the maximum.

$$ ( \sum_{k=1}^{m} (x_k - w_k )^p )^{1/p} $$

__Cosine Similarity__. This measures the angle between the two data points (as vectors from the origin). Note that the numerator is the inner product of the two points:

$$ cos(\theta) = \frac{ X^T W }{ \| X \| \| W \| } $$

Note that different distances can lead to different decision boundaries. With a large enough training set, and assuming the training set labels are balanced, this shouldn't be too much of an issue, but it is worth keeping in mind when predicting new points. The following code computes three of the distances defined above (which three should be apparent). Note that I designed the code to work on `numpy` arrays, which represent data points as column vectors, but the code can easily be modified for row vectors (transposes).

```python
import numpy as np
# Compute distances for a single query point w, and training point x

# Manhattan distance
def manhattan_distance(x, w):
  # Take elementwise differences (should be col vectors)
  c = np.subtract(x, w)
  # Take the absolute values of differences and sum
  d = np.sum(np.absolute(c))
  return d

# Euclidean Distance
def euclidean_distance(x, w):
  # Take elementwise differences
  c = np.subtract(x,w)
  # Take sum of squares of differences
  d = np.sum(np.square(c))
  return d

# Cosine Similarity/Something else
def cos_similarity(x, w):
  # Take dot product(numerator)
  num = np.dot(x, w)
  # Take magnitudes (denominator)
  mag_x, mag_w = np.linalg.norm(x), np.linalg.norm(w)
  # Construct Cosine
  cos = num/(max_x*mag_w)
  return cos
```
In the classification object we will build shortly, we now have the means of using several different distance metrics, which will be helpful in determining how robust our results are.

## Don't Forget to Scale

Before we compute any distance using our data points, it is important to note that it is good practice in any distance calculation to scale the inputs. In our case, we want to normalize each of our features so that they have approximately the same domain. If we do not do this, then one feature containing high numeric values relative to the other columns will dominate the distance computation (and thus the assignment of label to query points), even though there may be other, more important features. For example, if we use population (which is often in the thousands or higher) and number of universities (often a low natural number) to predict the income of a US state, the population will dominate the distances between points, even though the universities clearly matter in determining income. To avoid this, we can subtract each column by its mean, and divide the result by that column's standard deviation. This process is called __standardization__, and the code below accomplishes it:
```python
# Scale/subset the data

def scale_features(x):
  # Compute mean and standard deviation
  m = np.mean(x, axis = 0)
  s = np.std(x, axis = 0)
  # Standardize the features
  z = (x - m)/s
  # Return standardized scores and parameters
  return m, s, z
```
Notice that the code above also returns the mean and sample standard deviation of the input dataset. This is intentional - when we split our data into training and test sets later, we want to scale the training and test sets based on the mean and standard deviation of the training set only. This is to avoid what is called __data leakage__. We do not want any information from our 'unknown' data entering the training stage of the algorithm, since this would be equivalent to having access to the future, and we might mistakenly believe our algorithm is better than in reality.

# Implementing Prediction

To start, I'm going to consider the case I described above, where we have a single query point. You can think of this as a test set of size one, although later we'll expand our code to include any arbitrary test set. Our goal here is to product a prediction given the training features and labels, and a query point. The following code accomplishes this:
```python
def KNN_predict_point(X_train, y_train, X_query, k = 3):
  '''X_train is (N,D)
  y_train is (N,1)
  X_query is (D,1)
  Result is (1,1)'''
  # Scale train features
  X_train, m, s = scale_features(X_train)
  # Scale test features in an identical way avoids data leakage
  X_query = (X_query - m)/s

  # Compute distances between train points and query point
  d = euclidean_distance(X_train, X_query)
  # Extract k closest distances
  neighbor_idx = np.argsort(d)[:k]

  # Majority vote for label
  y_votes = y_train[neighbor_idx]
  y_votes, y_counts = np.unique(y_votes, return_counts = True)
  # Get most frequent label
  max_vote = np.argmax(y_counts)
  pred_y = y_votes[max_vote] 

  return pred_y
```
You can see that the training and test data are identically scaled. Then the function computes the distance between each training point and the query point. Lastly, the k closest points are identified, and the majority label among them becomes the predicted label. Note that I chose Euclidean distance here, but you could use one of the other distance functions I showed above (or you could write a completely different one if you like). So now we have the ability to predict labels for new query points.

The next step is to generalize this to a larger matrix of query points, ie a test set. This is easily done row by row, since numpy arrays are row-iterable. Here I use a simple for loop, which is not very computationally efficient. However there are ways of speeding up such a computation.
```python
def KNN_pred_multi(X_train, y_train, X_test, k = 3):
  # Instantiate list to hold predictions
  N = X_train.shape[0]
  M = X_test.shape[0]
  preds = []
  # Generate individual labels for each row
  for row in X_test:
    preds.append(KNN_predict_point(X_train, y_train, row.T, k = 3))
  # Convert list to numpy array
  y_hat = np.asarray(preds).reshape((M,1))
  return y_hat
```

## How to Choose K?

The decision boundary we imply for new points depends on our choice of K. Thus K is considered a __hyperparameter__. How many nearest neighbors do we want? If we choose K to be small, then the boundary will be more flexible. This means that

## Computational Concerns

## Watch out for Imbalanced Classes

## The Curse of Dimensionality

## Further Reading

- I highly recommend the book _Machine Learning: An Applied Mathematics Introduction_, by Paul Wilmott. It provides an excellent introduction to not only KNNs, but also most other foundational ML algorithms.
- There is a very detailed section (including the curse of dimensionality argument on which mine is based) in the text _Elements of Statistical Learning_ by Tibshirani et al.
- Blog Sites like _Medium_ likely contain a wealth of knowledge on this algorithm and most others. 
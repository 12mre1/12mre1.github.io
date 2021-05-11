---
layout: post
title: "The Surprising Effectiveness of K Nearest Neighbours"
date: 2020-12-10
---

## The Effective Simplicity of KNN

_Prerequisite Math: Calculus (Derivatives), Intermediate Statistics (Prior, Likelihood, Posterior)_
_Prerequisite Coding: Python(Numpy, Matplotlib)_

In today's post, I'll talk about one of the oldest and simplest Machine Learning algorithms, called __K-Nearest Neighbors__. It is a supervised learning technique, meaning in order to use it, we must have a data that is labelled (ie contains human-specified target values for the variable we wish to predict). I'll go into more detail shortly, but the basic assumption behind KNN is that points similar (close to each other) in feature space should have similar target values. I'll go through the algorithm in detail, discussing pros and cons while showing a detailed example and providing python code along the way.

## The Algorithm

As mentioned above, the key idea behind KNNs is that points of the same label should be near each other in feature space. To measure nearness, it is common to use a specific distance measure between any two points. So suppose we have a set of features, \\( X_{n \times m}\\), where \\( n \\) is the number of data points (rows), and \\( m \\) is the number of features (columns). You can think of this as a __training set__, although in K Nearest Neighbors classification, there really is not much training; we just use the existing data to classify new points. Along with our features, we also have class labels, \\( y_{n \times 1} \\). Suppose we have a new query point (ie one observation of a set of features, or an \\( 1 \times m \\) vector), and we want to predict it's class label. The K-Nearest Neighbors classification algorithm does the following:

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

## How to Choose K?

The decision boundary we imply for new points depends on our choice of K. Thus K is considered a __hyperparameter__. How many nearest neighbors do we want? If we choose K to be small, then the boundary will be more flexible. This means that

## Computational Concerns

## Watch out for Imbalanced Classes

## The Curse of Dimensionality

## Further Reading

- I highly recommend the book _Machine Learning: An Applied Mathematics Introduction_, by Paul Wilmott. It provides an excellent introduction to not only KNNs, but also most other foundational ML algorithms.
- There is a very detailed section (including the curse of dimensionality argument on which mine is based) in the text _Elements of Statistical Learning_ by Tibshirani et al.
- Blog Sites like _Medium_ likely contain a wealth of knowledge on this algorithm and most others. 
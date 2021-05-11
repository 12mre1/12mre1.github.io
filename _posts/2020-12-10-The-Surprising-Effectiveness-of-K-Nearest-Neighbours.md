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

As mentioned above, the key idea behind KNNs is that points of the same label should be near each other in feature space. To measure nearness, it is common to use a specific distance measure between any two points. So suppose we have a set of features, \\( X_{n \times m}\\), where \\( n \\) is the number of data points (rows), and \\( m \\) is the number of features (columns). You can think of this as a __training set__, although in K Nearest Neighbors classification, there really is not much training; we just use the existing data to classify new points. Along with our features, we also have class labels, \\( y_{n \times 1} \\). 

## The Distance Metric Matters

## Don't Forget to Scale

## How to Choose K?

## Computational Concerns

## Watch out for Imbalanced Classes

## The Curse of Dimensionality

## Further Reading

- I highly recommend the book _Machine Learning: An Applied Mathematics Introduction_, by Paul Wilmott. It provides an excellent introduction to not only KNNs, but also most other foundational ML algorithms.
- There is a very detailed section (including the curse of dimensionality argument on which mine is based) in the text _Elements of Statistical Learning_ by Tibshirani et al.
- Blog Sites like _Medium_ likely contain a wealth of knowledge on this algorithm and most others. 
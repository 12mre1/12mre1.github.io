---
layout: post
title: "Some Common Clustering Techniques"
date: 2020-11-09
---

## Some Common Clustering Algorithms

_Prerequisite Math: Calculus (Derivatives), Intermediate Statistics (Prior, Likelihood, Posterior)_
_Prerequisite Coding: Python(Sci-kit Learn)_

Sometimes, when building a ML model, we don't want have a target value or class we want to predict. Instead, we just want to identify some possible structure in the data. When would you want to do this? An obvious example would be for customer segmentation, if a company wants to identify groups of similar customers. Another example would be the grouping of purchases to identify fraudulent charges. This are just a couple of examples, but the general idea is to look for patterns in a set of unlabelled (or unsupervised) data. We call this type of learning __clustering__ because the structure we are looking for groups the data into clusters. There are many different types of clustering algorithms, but today I'm going to introduce you to the three most popular:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

We'll look at how each is implemented, then we'll run each algorithm on the same dataset. Finally, we'll walk through the tradeoffs associated with each type of algorithm.

## K-Means Clustering
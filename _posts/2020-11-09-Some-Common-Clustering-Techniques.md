---
layout: post
title: "Some Common Clustering Techniques"
date: 2020-11-09
---

## Some Common Clustering Algorithms

_Prerequisite Math: Calculus (Derivatives), Intermediate Statistics (Prior, Likelihood, Posterior)_

_Prerequisite Coding: Python (Sci-kit Learn)_

Sometimes, when building a ML model, we don't want have a target value or class we want to predict. Instead, we just want to identify some possible structure in the data. When would you want to do this? An obvious example would be for customer segmentation, if a company wants to identify groups of similar customers. Another example would be the grouping of purchases to identify fraudulent charges. This are just a couple of examples, but the general idea is to look for patterns in a set of unlabelled (or unsupervised) data. We call this type of learning __clustering__ because the structure we are looking for groups the data into clusters. There are many different types of clustering algorithms, but today I'm going to introduce you to the three most popular:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

We'll look at how each is implemented, then we'll run each algorithm on the same dataset. Finally, we'll walk through the tradeoffs associated with each type of algorithm. Though I will go into depth on how each algorithm works, the code itself will be highly abstracted, coming from python's `scikit-learn` ML API. 

## Clustering

I've talked about _clustering_ in previous posts, but I'll review it again here. Recall that in Machine Learning, __clusters__ are just groups of unlabelled data points that share feature similarity. In other words, these collections of points are close together in feature space. The goal of clustering algorithms then, is to take an arbitrary set of points, and identify these hidden groups. Unlike in the __supervised learning__ setting, where our dataset has labels with which to evaluate the performance of our algorithm using some kind of loss, clustering is almost always __unsupervised__, so it can be difficult to find a _correct_ answer. However most examples rely on domain knowledge to inform what constitutes a reasonable clustering assignment. Visualization can also be extremely helpful - you should be able to see the different groups clearly. 

There are many different examples of clustering, but here are some common applications:

- Grouping customers by purchasing patterns
- Deciding optimal placement of city parks by identifying clusters of foot traffic
- Optimizing Neck Size and Arm Length of shirt sizes

In all of these cases, and in virtually all clustering algorithms, the points are represented by vectors, with features as entries. The algorithm's job is to __assign each point__ to a specific cluster. A warning before we get started - always remember to scale your features when using distance-based algorithms, so as to avoid one feature dominating the others. `Scikit-learn` transformers will often do this for you, but it never hurts to double check.

## K-Means

The K-Means algorithm is an iterative process that splits the dataset into K non-overlapping subsets without any cluster-internal structure. Each group has an associated __center of mass__ (centroid), and it is the distance from these centroids that determines whether a point belongs to a given cluster. The goal of this algorithm in particular is two-fold: to determine the optimal assignment of points for a given collection of clusters, and to determine the optimal placement of cluster centres given the collection of points. This brings up a curious chicken egg problem - if we knew the true cluster assignments, we could easily compute the cluster centres; But if we knew the true cluster centres, finding the optimal assignment would be just as easy. So how do we deal with this problem? We use the following iterative approach:

1. Randomly initialize K cluster centres in feature space (K is a hyperparameter - more on this later).
2. Compute the distances between each point and each cluster center.
3. Assign each point the cluster label corresponding to the closest cluster center.
4. Holding constant the cluster assignments, recompute the centroids of each cluster.
5. Repeat steps 2-4 until convergence (ie until cluster centroids and/or cluster assignments no longer change).

Sounds impossibly simple right? Well most great ML algorithms are. This one has its strengths and weaknesses too. One good consequence is that, under this approach, the within-cluster variance is minimized while the between-cluster variance is maximized. Like points share a group, and unlike points do not. A more subtle result involves the shape of the clusters. Since we use distance from a centroid as our guide to find optimal assignment, the clusters of K-Means are perfect circles (or hyperspheres in higher dimensions). Depending on the application, this may not be desirable. I should also mention that K-Means is not guaranteed to converge to a global optimum; you will get a locally optimal set of clusters, but it is not necessarily going to give the best result. However, there are some ways to prevent this. You can (and should):

- Run the algorithm multiple times with different initializations
- Randomly break and combine clusters between iterations

Recall that I mentioned we typically do not have labels. Though this is true, we can still use a __cost function__ which will tell us which assignments are better than others. Typically we use the average distance between a point and its cluster centroid. Notice that, the more clusters we choose to identify (the higher K we pick), the smaller that average distance will be. Note also that as the algorithm iterates, this cost will decrease. In the extreme, if \\( K = N \\), we end up with a cost of zero, since each data point represents its own cluster (and cluster centroid). One corollary of this is that we can run the algorithm for a number of different values of \\( K \\), and plot cost against \\( K \\). This is called a __scree plot__, and the natural choice of K would present itself as an elbow, or a kink in this plot.

## Hierarchical Clustering

Hierarchical clustering is a newer, and very different approach. Unlike in K-Means where clusters were non-overlapping, the goal of __hierarchical clustering__ is to generate a series of clusters that have a hierarchical (tree-like) relationship. In this tree, each node is a cluster that consist of the clusters of its daughter nodes. Essentially what this algorithm does is start from raw data points, and build a tree of hierarchical clustering assignments. There are two main ways to build such a tree:

- From the top down (called __Divisive__, because we start with one cluster, and split it up)
- From the bottom up (called __Agglomerative__, because we start with several clusters, and combine them)

Both work well, but agglomerative has been more popular in recent years. Once the entire tree is built, we can visualize the assignments in a special graph called a __dendrogram__. To get disjoint clusters, we just cut the dendrogram horizontally (like cutting a tree by the trunk). We'll see one shortly, but first let's formalize the hierarchical clustering algorithm. It follows the following steps:

1. Create \\( N \\) clusters, one for each data point. 
2. Compute the __proximity matrix__, which measures the distances between each cluster and every other cluster
3. Repeat until \\( K = 1\\):
    i) Merge the two closest clusters
    ii) Update the proximity matrix, storing the old version
4. Return all cluster assignments

Like K-Means, the above algorithm is surprisingly simple. But there are a couple of questions that come to mind. What distance metric should be used? Also, two clusters have multiple points, how do we measure the distance between them? The first question does not have a best answer. Depending on the application, you may find different metrics do better. Some examples include euclidean distance, manhattan distance, and cosine similarity (I encourage you to read my post on K-Nearest Neighbors to see more information on such metrics). However the second question has garnered a number of different approaches, and it too does not have one best answer. There are several ways of computing distances between multipoint clusters, such as:

- __Single Linkage Clustering__: Uses the minimum point-to-point distance between clusters.
- __Complete Linkage Clustering__: Uses the maximum point-to-point distance between clusters.
- __Average Linkage Clustering__: Uses the average point-to-point distance between clusters.
- __Centroid Linkage Clustering__: Uses the distance between cluster centroids.

## DBSCAN

## Conclusions



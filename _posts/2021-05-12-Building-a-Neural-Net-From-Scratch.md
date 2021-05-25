---
layout: post
title: "Building a Neural Net From Scratch"
date: 2021-05-12
---
_Prerequisite Math: Calculus_

_Prerequisite Coding: Python (basic)_

## Building a Neural Net Using Just Numpy

I remember reading an article the other day about how someone had attempted to make a BLT completely from scratch. They milked the cow, grew the lettuce and tomato, and even butchered their own bacon. They made their own mayonnaise, and baked their own bread. When all ways said and done, the price of making that sandwich by hand turned out to be well in excess of 1000 dollars. This is shocking to some, given that you can buy a BLT from many sandwich shops for about 8 dollars as of the writing of this post. This is because advances in technology bring significant cost reductions, that people often take for granted.

Many ML practitioners take a similar view when building models. Using state of the art frameworks like pytorch, tensorflow, and keras, it can seem like building
deep learning models is just a matter of typing a handful of lines of code. With so much of the process fully automated, one can lose sight of what actually goes on under the hood. With this in mind, I'm going to build a simple neural net from scratch, relying only on python's `numpy` package to store and manipulate arrays. The network itself is just a simple 3-input, 1-output model with one hidden layer. Here is a picture:

<center><img src="/img/nn-from-scratch.png" alt = "basic-nn"></center>

You can see that I've used a single hidden layer with 4 nodes. Now before I write a single line of code, I'm going to make sure I have all the equations for the network mapped out. I'll also make sure I know the dimensions for each parameter (or parameter matrix), since dimension errors are one of the most frequent bugs found in deep learning code. To begin, let me define a few quantities of interest:

$$ N $$ (note the capitalization) is the number of data points we have. When training the network, this will be the size of our training set.

$$ D $$ is the number of features we have. This corresponds to the number of nodes in our input layer, and is also the number of columns in our training set.

$$ n_h $$ is the number of nodes in our hidden layer. This is the middle layer of our network, and we can see from the picture that \\( n_h = 4\\). 

## Vector Operations

As I mentioned above, we have only one hidden layer. Here are the equations that govern the network, with \\( X_0 \\) representing our raw data, which has dimension (\\( N, D \\) ):

$$ Z_1 = X_0 W_1 $$
$$ X_1 = Relu(Z_1) $$
$$ y = X_1 W_2 $$
$$ L = \frac{1}{2} (y - t)^2 $$

Pretty simple right? Note that I have removed any bias terms that you might see in other networks (just for simplicity). Now in order to derive the dimensions for the weights, we can first get the dimensions of our inputs and outputs, then work between them. Let me show you what I mean. I know (by definition) that our data must have dimension equal to the number of observations (rows) by the number of features (columns). Thus, \\( X_0 \in (N, D) \\). I also know that the dimension of the hidden layer's input just converts from feature space (3) to hidden-layer space (4), so it must be that \\( X_1 \in (N, n_h) \\). Knowing that the Relu activation works elementwise (does not change the dimension), it must be that \\( W_1 \in (D, n_h) \\), otherwise the dimensions would not match. In any dot product, the columns of the left matrix must equal the rows of the right matrix.

## Our Data

## Forward Propagation

## Dropout

## Backward Propagation

## Visualization
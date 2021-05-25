---
layout: post
title: "Building a Neural Net From Scratch"
date: 2021-05-12
---
_Prerequisite Math: Calculus_

_Prerequisite Coding: Python (Numpy)_

## Building a Neural Net Using Just Numpy

I remember reading an article the other day about how someone had attempted to make a BLT completely from scratch. They milked the cow, grew the lettuce and tomato, and even butchered their own bacon. They made their own mayonnaise, and baked their own bread. When all ways said and done, the price of making that sandwich by hand turned out to be well in excess of 1000 dollars. This is shocking to some, given that you can buy a BLT from many sandwich shops for about 8 dollars as of the writing of this post. This is because advances in technology bring significant cost reductions, that people often take for granted.

Many ML practitioners take a similar view when building models. Using state of the art frameworks like pytorch, tensorflow, and keras, it can seem like building
deep learning models is just a matter of typing a handful of lines of code. With so much of the process fully automated, one can lose sight of what actually goes on under the hood. With this in mind, I'm going to build a simple neural net from scratch, relying only on python's `numpy` package to store and manipulate arrays. The network itself is just a simple 3-input, 1-output model with one hidden layer. Here is a picture:

<center><img src="/img/nn-from-scratch.png" alt = "basic-nn"></center>

You can see that I've used a single hidden layer with 4 nodes. Now before I write a single line of code, I'm going to make sure I have all the equations for the network mapped out. I'll also make sure I know the dimensions for each parameter (or parameter matrix), since dimension errors are one of the most frequent bugs found in deep learning code. To begin, let me define a few quantities of interest:

$$ N $$ (note the capitalization) is the number of data points we have. When training the network, this will be the size of our training set.

$$ D $$ is the number of features we have. This corresponds to the number of nodes in our input layer, and is also the number of columns in our training set.

$$ n_h $$ is the number of nodes in our hidden layer. This is the middle layer of our network, and we can see from the picture that \\( n_h = 4\\). 

## Our Data

Today is less about achieving state of the art results, and more about building the network. With this in mind, I'll use the famous [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html), which uses 13 different features to try and predict median home value in 506 different suburbs of Boston. For this demonstration, I'll just use three features: Property tax per 10,000 dollars, average pupil-teacher ratio, and crime-rate per capita. The labels are listed in thousands of dollars. Here is the code for loading and extracting the features and labels. I also define some parameters we'll use shortly.

```python
mport numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

features, labels = load_boston(return_X_y = True)

# Take only three features, and the labels
X0 = features[:,[0,9,10]]
t = labels

# Define parameters
N = X0.shape[0]
D = X0.shape[1]
n_h = 4 
```
With the data prepped and parameters chosen, let's look at how to code the network itself.

## Vector Operations

As I mentioned above, we have only one hidden layer. Here are the equations that govern the network, with \\( X_0 \\) representing our raw data, which has dimension (\\( N, D \\) ):

$$ Z_1 = X_0 W_1 $$

$$ X_1 = Relu(Z_1) $$

$$ y = X_1 W_2 $$

$$ L = \frac{1}{2} (y - t)^2 $$

Pretty simple right? Note that I have removed any bias terms that you might see in other networks (just for simplicity). Now in order to derive the dimensions for the weights, we can first get the dimensions of our inputs and outputs, then work between them. Let me show you what I mean. I know (by definition) that our data must have dimension equal to the number of observations (rows) by the number of features (columns). Thus, \\( X_0 \in (N, D) \\). I also know that the dimension of the hidden layer's input just converts from feature space (3) to hidden-layer space (4), so it must be that \\( X_1 \in (N, n_h) \\). Knowing that the Relu activation works elementwise (does not change the dimension), it must be that \\( W_1 \in (D, n_h) \\), otherwise the dimensions would not match. In any dot product, the columns of the left matrix must equal the rows of the right matrix.

Similarly, we can deduce the dimensions of the other variables:

$$ X_0 - (N, D) \ , \ W_1 - (D, n_h) $$

$$ Z_1 - (N, n_h) \ , \ X_1 - (N, n_h) $$

$$ W_2 - (n_h, 1) \ , \ y - (N, 1) $$

Now that we have the equations to follow, and the dimensions are computed, we can code the forward pass of our network. The code for this is below. I also include the random initialization of the weights:

```python
# Initialize weight matrices
W1 = 2*np.random.random((D, n_h)) - 1
W2 = 2*np.random.random((n_h, 1)) - 1

# Create activation function
def relu(x):
  return (x > 0)*x

## Forward Propagation

def forward_pass(X0, t, W1, W2):
  '''Computes the forward propagation of the 
  network and stores parameters.
  X0: the training data (N,D)
  t: the labels (N,1)
  W1: 1st weight matrix (D,n_h)
  W2: 2nd weight matrix (n_h, 1)'''
  # Run through network per equations given
  Z1 = np.dot(X0, W1)
  X1 = relu(Z1)
  y = np.dot(X1, W2)
  # Compute and print error (MSE)
  error = np.sum(0.5*(y - t)**2)
  print('Error: ', error)
  print('---------------')
  # Return values
  return X1, y
```

## Backward Propagation

The __backpropagation__ step requires that we attribute the overall change in our loss (error) to each of the weights. In order to do this, we need to know how sensitive is our loss to changes in each weight. In other words, we want to compute the derivatives with respect to our two weight matrices ( \\( W_1, W_2 \\) ). We do this by using the chain rule. But there are many relationships to keep track of, even in this simple network. To keep things organized, and to help with derivative computation, I like to turn our equations into a __computation graph__. This is a structure that identifies all the relationships in our network, and here is what ours looks like:

<center><img src="/img/nngraph.png" alt = "basic-nn"></center>

There aren't a lot of free computation graph illustrators, so I just drew this one by hand. But beyond showing the relationships in our network, such a graph is very useful for computing derivatives using the __chain rule__. One of the more interesting aspects of coding up a neural network is that __we don't need closed expressions for each derivative, we only need rules for computing the derivatives we care about__. What does this mean exactly? Well, suppose we want to compute the derivative of the loss with respect to the first weight matrix, \\( W_1 \\). You can see there are a lot of intermediate variables whose derivatives we would need to find first. Instead of combining all these individual relationships to get a closed-form expression for the derivative we want, we work backwords through the graph. Start with loss, and take its derivative only with respect to variables connected to it (ie the variables whose arrows point to it). Once we've computed those, we move one level backward, taking the derivative of y w.r.t the variables connected to it, and so on and so forth. Here's how this looks when done on the whole graph. Note that \\( t \\) is our labels, also called targets, which aren't really variables, since they will not change. So I don't bother with that derivative.

$$ dL = \bar{L} = \frac{dL}{dL} = 1 $$

$$ dy = \bar{y} = \frac{\partial L}{\partial y} = (y - t) $$

$$ dx1 = \bar{X_1} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial X_1} = (y - t) W_2^T  = \bar{y}  W_2^T $$

$$ dw2 = \bar{W_2} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial W_2} = X_1^T (y - t) = X_1^T \bar{y} $$

$$ dz1 = \bar{Z_1} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial X_1} \frac{\partial X_1}{\partial Z_1} = I(Z_1 > 0) \bar{X_1} $$

$$ dx0 = \bar{X_0} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial X_1} \frac{\partial X_1}{\partial Z_1} \frac{\partial Z_1}{\partial X_0}  = \bar{Z_1} W_1^T $$

$$ dw1 = \bar{W_1} = \bar{Z_1} \frac{\partial Z_1}{\partial W_1} = X_0^T \bar{Z_1} $$

Perfect! We now have the derivatives we need to perform backpropagation. I typically use the 'bar' notation you see above to express pieces of the chain in a longer derivative. Notice also apart from the first two derivations, we never needed to express any of these derivatives in there full form. Not only is this easier when deriving these with pen and paper, but it is also faster computationally. I actually computed more derivatives than we needed here, but I just wanted to stress the advantage of using the computation graph.

You might be wondering why I used the matrix transpose in certain places, and how I knew when to do this. One very helpful rule of thumb is that __the matrix of derivatives must have the same dimension as the variable itself__. For example dx1 must have the same shape as \\( X_1 \\), dw2 the same shape as \\( W_2 \\), and so on. So I use the transpose wherever it is needed to preserve the dimension. When we execute backprop, even though we only need the derivatives of the weights (since they are what is being updated as the network learns), preserving the shape across all intermediate variables is good practice.

Here is the code for backpropagation:
```python
def backward_pass(X0, t, W1, W2, X1, y):
  '''Computes the backward propagation of the
  network and returns the updated weights.
  X1: The data in the hidden layer (N, n_h)
  y: The predicted target (N,1)'''
  # Compute derivatives shown above
  dy = (y - t)
  dX1 = np.dot(dy, W2.T)
  dW2 = np.dot(X1.T, dy)
  dZ1 = (Z1 > 0)*dX1
  dX0 = np.dot(dZ1, W1.T)
  dW1 = np.dot(X0.T, dZ1)
  # Update the weights
  W1_new -= alpha*dW1
  W2_new -= alpha*dW2
  # Return updates
  return W1_new, W2_new
```

## Visualization

## Dropout
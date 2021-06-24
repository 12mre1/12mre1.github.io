---
layout: post
title: "Optimizing Deep Neural Nets with Tensorflow"
date: 2021-06-15
---

## Introduction

Most of today's state-of-the-art machine learning models are some kind of deep neural network. But while such models are excellent at fitting complex functions to model relationships between features (X) and targets (y), one of their biggest downsides is that they can take a very long time to train. There are several ways you can mitigate this such as (this list is not exhaustive):
- Weight Initialization (e.g. Glorot-He Initialization)
- Non-saturating activation functions (e.g. ReLU)
- Batch or Layer Normalization
- Pretraining (particularly in the NLP setting)

I encourage you to experiment with as many of these hyperparameters as possible to see what works best. However in this post, I'm going to talk about one of the biggest ways to improve training time: the __optimizer__. Most deep learning frameworks (and indeed many non-neural modelling solutions) use some version of __Gradient Descent__ as the default optimizer. While this is an excellent way to generate optimal weights in your network, researchers have in the past decade or so, come up with a variety of clever ways to modify this procedure to ensure faster and smoother convergence than is typically found in SGD. Specifically, I'm going to walk you through the following alternative procedures:
1. Momentum
2. Nesterov Accelerated Gradient
3. RMSProp
4. AdaGrad
5. Adam
6. Nadam

We'll discuss how each of these is computed, as well as the intuition behind them. Then we'll train a fixed model, and use __tensorboard__ to visualize the convergence of the weights, so you can see the differences. If you've never used (or heard of) tensorboard, don't worry. I'll explain it in more detail a bit later.

## Review of SGD

__Gradient Descent__ is an iterative optimization algorithm that is designed to find the minimum of a surface by repeatedly taking steps in the opposite direction of the direction of steepest increase. By definition, the direction of greatest increase at a point on a surface is the __gradient__, or the vector of partial derivatives evaluated at that point. For example, if the surface is given by the function \\( f = 5x^2 + 2xy + 3y^2 \\), then the gradient is given by \\( \nabla_{f} = [ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} ]^T = [10x +2y , 2x + 6y] \\). If we're at a specific point, say \\( (x = 1, y = 1)\\), then the gradient has real entries, \\([12, 8]^T \\). 

Most neural networks update weights (and biases) according to the following formula:

$$ w := w - \alpha \frac{\partial L}{\partial w}$$

Where \\( w \\) typically denotes the weight matrix. Doing this for a number of epochs will usually result in convergence to the minimum of the loss function (ie the weights converge to their optimal values). The following graph shows this process of iteratively descending the loss surface until the optimal weights are found. I'd also like to mention that I never claimed to be an artist: 

<center>
<figure>
    <img src='/img/sgd.png' alt='missing' />
    <figcaption>Source: rasbt.github.io</figcaption>
</figure>
</center>

In the formula above, \\( \alpha \\) denotes the __learning rate__, which is a hyperparameter designed to control step size. You can see that eventually, repeated iterations of weight updates reach the globabl minimum of the loss function. The dotted lines represent the gradients at fixed points along the way. Notice also that we're assuming the surface is differentiable (and continuous) for the entire domain of weights. However, the loss function does not necessarily have to look 'pretty'. There may be many local optima and saddle points in which regular GD can get stuck. Some of the technique's we'll discuss shortly are also great at avoiding this problem.

One final thing I'd like to mention is that Gradient descent is considered a __first-order__ optimization algorithm, because we only ever work with the first derivatives of the loss surface. Although higher-order derivatives may work well in theory, in practice, this involves computing the __Hessian__ (matrix of second derivatives) of the loss, which for an entire matrix containing millions of weights, involves far too many parameters to be feasible in practice. In future I may dedicate a separate post to higher-order optimization methods, but for now just know that they do exist (and they're a fascinating area of research).

## Today's Example Data 

## Momentum

For those familiar with Newtonian physics, momentum measures the quantity of motion of an object. It is computed as the product of an object's mass and velocity, and it is this physical interpretation that inspired the Momentum Optimization variant. Regular Gradient descent does not care about the gradients in previous update iterations - if the current value is small, so will be the update. This can sometimes cause convergence to take a very long time. Conversely, Momentum optimization cares a lot about earlier gradients. At each iteration, instead of subtracting the gradient directly, it subtracts the gradient from a __momentum vector__, and updates the weights by adding the momentum vector to the previous instance. Here are the equations defining Momentum:

$$ V_{w} := \beta V_{w} + (1 - \beta) \nabla_{w} J(w) $$

$$ w := w - \alpha V_{w} $$

There would be identical equations for updating the bias term as well. Notice that we now have an additional hyperparameter: \\( \beta \in (0,1) \\). This is called the _momentum_ parameter. You can think of this new term as a friction mechanism: when small, there is little momentum, and the update is controlled almost exclusively by the gradient. However, when the momentum coefficient is large, the optimizer moves very quickly towards the minimum. In practice, it is common to set \\( \beta = 0.9 \\). We want a good deal of friction, otherwise the optimizer will quickly reach the minimum, but will overshoot it, continuing to oscilate and taking longer than we would like. 

A couple of other things to note. Often in the literature, you will see the first equation written as \\( V_{w} := \beta V_{w} + \nabla_{w} J(w) \\), which has the effect of scaling the equation by \\( 1 - \beta \\), but the principle is identical. Also, some papers don't bother using this on the bias term. Notice too that in these equations, the gradient acts not as the velocity, but as the acceleration ( \\( V_{w} \\) is akin to the velocity). If there is a drawback in using Momentum, it's that you now have another hyperparameter to tune (yay!). However in certain situations, such as when there is no batch normalization, and features can have very different units, this addition can save a lot of time.

So how do we implement this in Tensorflow? Luckily, it's incredibly easy:

## Nesterov Accelerated Gradient

## RMSProp

## AdaGrad

## Adam

## Nadam

## Conclusion

## Further Reading
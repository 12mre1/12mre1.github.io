---
layout: post
title: "An Intro to OLS Regression"
date: 2020-11-15
---

## OLS Regression: An ML Classic

_Prerequisite Math: Calculus (Derivatives), Introductory Statistics (Expectation, Normal Distribution)_

_Prerequisite Coding: Basic Python_


When we want to use a set of features X, to predict some response Y, one of the simplest (and oldest) approaches is to assume that
the predictor-response relationship is linear (ie can be modelled well with a straight line). This approach is intuitive, because
our brains deal with linear relationships every day. For example, many people are paid in wages (a linear relation between hours and pay), 
and most of us at least roughly obey the speed limit (a linear relation between distance and time) while driving. Another benefit of 
assuming a linear relationship is that the math becomes much more manageable, as we will see shortly.

### What is Ordinary Least Squares?

OLS stands for Ordinary Least Squares. We sometimes call this Linear Least Squares, or just Linear Regression. The names comes from the fact 
that we're choosing our line (the linear model) in order to minimize the sum of the squared errors of our predictions. Why this is a good thing
will become clear a bit later. For now, let's formally define our model. Suppose we have a dataset of n observations, where each observation is 
a datapoint (x,y). What we're trying to do is learn the general relationship between the two variables (X and Y). In this post, I will follow
the convention of using capitals to denote random variables, and lowercase letters to denote the realizations of those variables. For example,
if I wanted to model a linear relationship between height and weight, X and Y would be height and weight generally, while x,y would be specific
values (say, 152 cm and 170 lbs). Assuming a linear relationship, our model would take the following form:  

\\[ y_i = \beta_0 + \beta_1 x_i + \epsilon_i \\]

In this case, the model is simply

### How Do We Estimate $\beta$?

### Newton's Method

### Properties of the OLS Estimator

### Computational Concerns

### Application: Predicting Student Test Scores

### Other Extensions

- Weighted Least Squares
- Generalized Least Squares
- Non-linear Least Squares
- Regularized Regression

### Further Reading


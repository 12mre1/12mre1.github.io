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

OLS stands for Ordinary Least Squares. We sometimes call this Linear Least Squares, or just Linear Regression. The names come from the fact 
that we're choosing our line (the linear model) in order to minimize the sum of the squared errors of our predictions. Why this is a good thing
will become clear a bit later. For now, let's formally define our model. Suppose we have a dataset of n observations, where each observation is 
a datapoint (x,y). What we're trying to do is learn the general relationship between the two variables (X and Y). In this post, I will follow
the convention of using capitals to denote random variables, and lowercase letters to denote the realizations of those variables. For example,
if I wanted to model a linear relationship between height and weight, X and Y would be height and weight generally, while x,y would be specific
values (say, 152 cm and 170 lbs). Assuming a linear relationship, our model would take the following form:  

\\[ y_i = \beta_0 + \beta_1 x_i + \epsilon_i \\]

In this case, the model is simply a sum of three terms. \\( \beta_0 \\) is the __intercept__ of our line, while \\( \beta_1 \\) is the 
__slope coefficient__. These together make up the line that our model uses to predict new, unseen data. The third term, \\( \epsilon_i \\) 
is called the __error__ . It reflects the fact that, on an observation to observation basis, our prediction might be a little bit off. In 
other words, our line will not pass perfectly through all the points in our dataset. There will be some difference between our predicted
value of y, and the true value of y. In this case, we use subscript i, \\( i = 1,...,n \\) to denote the fact that each observation
has its own error, but note that the slope and intercept are constant across all observations. In this model, we call Y the __dependent variable__, 
and X the __independent variable__. This is because we generally want to predict Y given X (not the other way around), and
this terminology is used for other models as well. Sometimes, in the regression setting we use the terms __regressor__ and __regressand__ 
to denote X and Y respectively. 

We often make additional assumptions on our error term, \\( \epsilon_i \\). First, we assume that the errors are \\( iid \\). This means
that errors are independent across observations, and that each of the errors comes from the same Data Generating Process (DGP). All
this means is that, after accounting for the linear shape of the relationship between X and y, there is no other trend or relationship
that we have not accounted for. As we will see later, this assumption is not always realistic. We also assume that errors are mean zero.
In other words, our predictions will be perfect on average. Consider, for a moment, if the errors were not mean zero. If errors had an 
average value of say, 2, we could simply add 2 to our intercept and force the errors down to zero mean! 

The case described above is called __simple linear regression__, because we are only studying the relationship between Y and one other 
variable, X. We can generalize the math above to account for multiple independent variables (indeed, we will do so later). When this happens, 
the model takes the form not of a line, but of a _plane_ (or a _hyperplane_ for > 3 dimensions). But we'll come back to that later. 
Regardless of the number of variables, we make one additional assumption about our errors: that they are independent of our regressor(s). 
Formally, this means that \\( E(\epsilon_i|x_i) = 0 \\). We call this assumption __exogeneity__ of errors, and it is crucial. If the
errors and the regressors were not independent, there would be some way of using X to influence the errors. When trying to find the true
relationship between Y and X, we would never be able to isolate X alone, since any changes to X would impact Y both through X
directly, and through the error. By definition, we want our error to be unpredictable. If the error were predictable, it would mean our
linear model is __incorrectly specified__, or unreflective of the true Data Generating Process.

### How Do We Estimate \\( \beta \\)?

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


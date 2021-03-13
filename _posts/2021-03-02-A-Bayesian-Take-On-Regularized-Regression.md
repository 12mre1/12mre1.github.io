---
layout: post
title: "A Bayesian Take on Regularized Regression"
date: 2021-03-02
---

## The Bayesian Foundations of Regularized Regression

_Prerequisite Math: Calculus (Derivatives), Intermediate Statistics (Prior, Likelihood, Posterior)_

_Prerequisite Coding: None. This is post is entirely theory._

Converting the given definitions into vector notation, we have:

{% raw %}
$$
\begin{align*}
    X = Z\beta + \epsilon \ \ \epsilon \sim N(0, \sigma^2 I)  \\
    \hat{\beta}_{Ridge} = argmin_{\beta} \{ (X - Z\beta)^T (X - Z\beta) + \lambda ||\beta||_{2}^{2} \} \\
    \hat{\beta}_{LASSO} = argmin_{\beta} \{ (X - Z\beta)^T (X - Z\beta) + \lambda ||\beta||_{1} \} 
\end{align*}
$$
{% endraw %}

If the priors for our \\( \beta_{j} \\) are zero-mean and independent (ie diagonal covariance matrix), then we have \\( \beta \sim N(0, u^2 I) \\) for some constant \\( u \\). We can compute the posterior:

{% raw %}
$$
\begin{align*}
    &\propto exp(-\frac{1}{2}(\beta - 0)^T \frac{1}{u^2}I(\beta - 0)) \cdot exp(-\frac{1}{2}(X - Z\beta)^T \frac{1}{\sigma^2}I(X - Z\beta)) \\
    &= exp(-\frac{1}{2\sigma^2} (X - Z\beta)^T (X - Z\beta)  - \frac{1}{2u^2} \beta^T\beta) \\
    &= exp(-\frac{1}{2\sigma^2} (X - Z\beta)^T (X - Z\beta)  - \frac{1}{2u^2}||\beta||_{2}^{2})
\end{align*}
$$
{% endraw %}

We can ignore the constants, since they do not affect  the maximization of the posterior (ie finding the mode). Taking the log, then multiplying by 2, we get that the mode is equal to:

{% raw %}
$$
\begin{align*}
    \hat{\beta} &= argmax_{\beta} \{ -\frac{1}{2\sigma^2} (X - Z\beta)^T (X - Z\beta)  - \frac{1}{2u^2}||\beta||_{2}^{2} \} \\
    &= argmax_{\beta} \{ (X - Z\beta)^T (X - Z\beta)  - \frac{\sigma^2}{u^2}||\beta||_{2}^{2} \}
\end{align*}
$$
{% endraw %}

Note that this is identical to the Ridge Regression estimate, with tuning parameter \\( \lambda = \frac{\sigma^2}{u^2} \\).

\

If we have normally distributed \\( \epsilon \\), but this time we use a double-exponential (or Laplace) prior for our  \\( \beta_{j}s \\), then (still assuming independence among individual priors):

{% raw %}
$$
\begin{align*}
    f(\beta) = \Pi_j f(\beta_j) = \Pi_{j} \frac{1}{2\tau} exp(- \frac{|\beta_j|}{\tau}) \propto exp(- \Sigma_j \frac{|\beta_j|}{\tau})
\end{align*}
$$
{% endraw %}

We can again derive the posterior, ignoring constants:

{% raw %}
$$
\begin{align*}
    p(\beta | X, Z) &\propto p(\beta) \dot p(X|Z, \beta) \\
    &\propto exp(- \Sigma_j \frac{|\beta_j|}{\tau}) \cdot exp(-(X - Z\beta)^T \frac{1}{\sigma^2}I(X - Z\beta)) \\
    &= exp(-\frac{1}{\sigma^2}(X - Z\beta)^T(X - Z\beta) - \frac{1}{\tau} ||\beta||_{1})
\end{align*}
$$
{% endraw %}

Just like before, we compute the MAP estimate of \\( \beta \\), which is equivalent to finding the posterior mode:

{% raw %}
$$
\begin{align*}
    \hat{\beta} &= argmin_{\beta}\{ exp(-\frac{1}{2\sigma^2}(X - Z\beta)^T(X - Z\beta) - \frac{1}{\tau} ||\beta||_{1}) \} \\
    &= argmin_{\beta} \{ -\frac{1}{\sigma^2}(X - Z\beta)^T(X - Z\beta) - \frac{1}{\tau} ||\beta||_{1}\} \\
    &= argmin_{\beta} \{ (X - Z\beta)^T(X - Z\beta) - \frac{\sigma^2}{\tau} ||\beta||_{1}\}
\end{align*}
$$
{% raw %}

This is identical to the LASSO estimate, with tuning parameter  \\( \lambda = \frac{\sigma^2}{\tau} \\).
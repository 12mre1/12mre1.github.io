---
layout: post
title: "Spam Classification From Scratch with Naive Bayes"
date: 2020-12-30
---

## Spam Classification From Scratch With Naive Bayes

_Prerequisite Math: Calculus (Derivatives), Intermediate Statistics (Prior, Likelihood, Posterior)_

_Prerequisite Coding: None_

One of the all time classic uses for classification algorithms is labelling email messages as Spam(bad) or Ham(good). Doing so allows us to automatically filter out messages we know are not actually important. But determining what exactly separates the ham from the spam is not something we can easily define. Although humans are great at determining whether a given email spam, the goal with Machine Learning is to learn the __decision rule__ that separates the classes, so that any message which is spam can be filtered without humans having to read it.
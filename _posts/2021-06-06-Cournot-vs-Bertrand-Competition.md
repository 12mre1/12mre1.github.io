---
layout: post
title: "Cournot vs Bertrand Competition"
date: 2021-06-06
---

_Prerequisite Math: Probability (Chain Rule, Bayes Rule)_

_Prerequisite Coding: Python (Functions)_

## On Cournot vs Bertrand Pricing Models

Today's post is going to be a bit longer than usual, and a bit more theoretical (no code, just math). We're going to take a close look at two types of models used to predict producer (firm) behavior in the marketplace: __Cournot__ and __Bertrand__. Both are named for their inventors, and both model competition among homogeneous firms in a competitive market, however there are advantages and disadvantages to each. We'll discuss the assumptions underlying each, and work through some of the derivations in a game theoretic setting. Here is a rough table of contents:

- Introduction
- Cournot Competition
    - Technical Assumptions
    - Profit Maximization
        - The Cournot Theorem
        - Nash Equilibrium and Best Response Functions
    - Example: Quadratic Costs
    - The Case of Heterogeneous Costs
- Bertrand Competition
    - Technical Assumptions
    - Profit Maximization

## Introduction

Within any industry, the firm must constantly make decisions in order to compete in the market. This note compares two of the earliest models of firm dynamics in a competitive market: __Cournot competition__, and __Bertrand competition__. In the former, firms compete on the basis of quantity produced, while in the latter, they compete on price.

## Cournot Competition

Invented by __Antoine Augustin Cournot__ in the mid 17th century, this model relies primarily on the belief that firms compete on the basis of quantity of goods produced. In this setting, firms are treated solely as a producer (among other identical producers), so there are no demand-side considerations beyond the market price, which is solely a function of quantity demanded.

### Technical Assumptions

This model is an obvious simplification of a true economy, but it does still yield useful properties. Under Cournot Competition, we assume the following rules:

1. ***There are a large number of firms, or at least enough for pure competition to be feasible.*** Under this setting, any deviation from market demand would result in any one firm being priced out of the market. Thus, in equilibrium (more on this later), all firms will have the same price.

2. ***Firms choose their quantity of supply with the sole objective of profit maximization***. In reality, there are many other aspects that factor into a firm's governance, but in this setting, the only thing that matters to each firm is profit. Also behind the scenes here is the assumption that all firms behave rationally.

3. ***Firms make their production decisions simultaneously.*** Though, in reality, firms within any industry constantly make decisions at different times, in our setting, every firm chooses their quantity at the exact same time. This prevents any one firm from gaining a first-mover or late-mover advantage.

4. ***Firms have market power, and the number of firms is fixed.*** In this setting, the total quantity supplied (which in turn, drives price through the market demand curve), is comprised of the sum of quantities chosen by each firm. No one firm has any more market power than any other, except (as we shall see) insofar as their superior cost function may allow for them to produce more than others.

5. ***There is no collusion between firms.*** In a competitive setting, there is always an incentive for larger firms to collude in order to maintain high prices for increased profit, or to keep prices low (ie below fixed costs) in order to deter entry into the market. Here however, we assume this is not possible. Firms decide their quantities independently. This is not to say they do not consider the decisions of competitors; only that cooperation is not allowed.

6. ***There is no product differentiation.*** This is a more implicity assumption, but in order to justify pure competition among firms, all firms must be selling the exact same good or service. There can be no differences that may account for differences in price or cost. Thus, we assume all firms produce exactly the same product. 

### Profit Maximization
### The Cournot Theorem
### Nash Equilibrium and Best Response Functions
### Example: Quadratic Costs
### The Case of Heterogeneous Costs
## Bertrand Competition
### Technical Assumptions
### Profit Maximization
---
layout: post
title: "Stable Matching with the Gale-Shapley Algorithm"
date: 2021-02-10
---

## Stable Matching with the Gale-Shapley Algorithm

_Prerequisite Math: Set Theory, Utility Theory (Basic)_

_Prerequisite Coding: Python (Basic)_

In this writeup, I'll be discussing one of the first big contributions to result from the combination of economics and computer science. This post will be a bit short and narrower in scope than usual, since its content will be somewhat rigorous. In 1962, mathematical economists Lloyd Shapley and David Gale published an algorithm that solved what is known as the __Stable Matching Problem__, an important problem in economics as well as many other disciplines. This algorithm, appropriately called the __Gale-Shapley Algorithm__, is designed to find a stable matching between two sets of candidates, when each candidate has a defined set of preferences over possible partners. If that explanation seemed vague, don't worry! I'll go into more detail shortly, but before I do, let me mention that although GS were the first to formally publish and prove certain properties about this algorithm, it was in use as early as 1950 by the United States National Residency Matching Program (NRMP) to match residency applicants to hospitals. Consequently this matching of hospitals and residences will be the running example I use for the remainder of this post. 

## The Stable Matching Problem

So what exactly did I mean earlier when I mentioned matching? Suppose we have $n$ medical residents, $R = \{ r_1 , \cdots \ r_n \}$ who are applying for residencies, and there are $n$ hospitals, $H = \{ h_1 , \cdots \ h_n \}$, who receive residency applications. Assume also that each hospital can hire only one resident, and each resident can only work for one hospital. Assume also that each resident has a __preference ordering__ over hospitals, and each hospital has a __preference ordering__ over applicants (residents). A preference ordering just means a clear ranking - if a resident receives two job offers, it is clear to that resident which offer they prefer (and a similar statement applies for each hospital). Note that preference orderings needn't be the same between hospitals or residents (in fact they almost never are).
## The Algorithm

## The Pros And Cons

## Conclusion

## Further Reading
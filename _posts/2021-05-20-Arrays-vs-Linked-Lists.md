---
layout: post
title: "Arrays vs Linked Lists"
date: 2021-05-20
---
_Prerequisite Math: None_

_Prerequisite Coding: Python (Classes and Methods)_

## Comparing Arrays and Linked Lists in Python

In today's post, I'm going to be talking about two fundamental data structures used in computer science: __arrays__ and __linked lists__. I'll explain what each is, what advantages each has compared to the other, then we'll code each of these structures in python. Even though python has basic structures in place that very much resemble what I'm about to show, I find it's very good practice to implement these yourselves. Doing so helped me understand them when I first learned about data structures and algorithms.

Let's start with a metaphor. Suppose you and your friends are trying to find seats for a hockey game. The memory of a computer looks a lot like the arena seats you and your friends would search for - a series of blocks (seats), each able to store one item (seat one person), and each block/seat having a unique address. And just like a computer storing information, there are a number of ways you and your friends might choose where to sit. An __array__ is a method for storing information that uses contiguous (sequential) slots. In this case, this means that you and your friends all sit side-by-side. What's unique about an array is that each position is numbered. In one dimension, this means that each position is given an index from left to right. Be warned, in many languages, this starts at zero, not at one (python is zero-indexed). Because the array is a set of continuous blocks, the size must be defined before any items are placed in memory. This means you choose how many seats to reserve before you and your friends show up on game day.

<center><img src="/img/array.png" alt = "digits"></center>
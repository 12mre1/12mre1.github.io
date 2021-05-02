---
layout: post
title: "The Basic Reinforcement Learning Framework"
date: 2021-02-17
---

## The Basic Framework of Reinforcement Learning

_Prerequisite Math: None_

_Prerequisite Coding: None_

This post is dedicated to the basics of __Reinforcement Learning__, which is a subdomain of Machine Learning more generally. Many machine learning techniques involve teaching computers to perform a task by giving the algorithm a series of examples, so the computer can attempt to deduce patterns underlying proper(optimal) behavior. Often (though not always), these examples come with labels made by humans , that allow for evaluative feedback (loss) to guide the machine towards the optimal solution given certain data. However RL operates in a fundamentally different way.  Instead of explicitly teaching the computer using human-driven feedback, we give the reinforcement learning agent a direct connection to its __environment__, so that it may __explore possible strategies through trial and error__. In this sense, RL agents learn from experience, and from this experience can learn to evaluate the consequences of actions.

Because of the sequential nature of environment exploration, and the fact that the agent can have a very large number of interactions through time, the goal of any Reinforcement agent is simply to maximize a __reward signal__ across time. This brings a number of challenges, since the action of the agent early in time may have a large impact on possible future rewards later. Thus, in evaluating possible actions, the agent must somehow think beyond its current situation. In other words, the agent must plan ahead. So how do we formally define such a framework? Fortunately in the RL community, the notation used to define the components I will go through shortly is fairly universal.

I begin here by defining the __action space__ \\( A \\), which is simply the set of all possible actions an agent may take. The set of actions can be remarkably simple - for example in the game of tic-tac-toe, which traditionally uses a 3 x 3 board, your action space will be a set of at most 9 choices. Whatever the agent decides at each step in the sequence, it must be that \\( a_t \in A \\), where \\( t = 1, \cdots, T \\) denotes current period in the sequence. Note that the action space may change as the agent progresses through time. Also the action space can be continuous or discrete. The tic-tac-toe example is discrete (meaning there are a countable number of choices), but in a different scenario - say, the agent must fly a helicoptor in high winds, or learn to walk - the action space can be continuous, or infinite. Note that the actions available to the agent also depend on what has happened up to the current point in the game. How do we keep track of this information? 

Perhaps the most important component in the RL framework is the concept of __state__. The state contains __all the relevent information about what has occurred thusfar__.
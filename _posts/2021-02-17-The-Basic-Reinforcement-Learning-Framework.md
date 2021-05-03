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

I begin here by defining the __action space__ \\( A \\), which is simply the set of all possible actions an agent may take. The set of actions can be remarkably simple - for example in the game of tic-tac-toe, which traditionally uses a 3 x 3 board, your action space will be a set of at most 9 choices. Whatever the agent decides at each step in the sequence, it must be that \\( a_t \in A \\), where \\( t = 0, \cdots, T \\) denotes current period in the sequence. Note that the action space may change as the agent progresses through time. Also the action space can be continuous or discrete. The tic-tac-toe example is discrete (meaning there are a countable number of choices), but in a different scenario - say, the agent must fly a helicoptor in high winds, or learn to walk - the action space can be continuous, or infinite. Note that the actions available to the agent also depend on what has happened up to the current point in the game. How do we keep track of this information? 

Perhaps the most important component in the RL framework is the concept of __state__. The state contains __all the relevent information about what has occurred thusfar__. In some cases, this representation is intuitive - the chess board configuration alone is enough for an agent trying to win the game of chess. But in some cases, the state can be quite subtle. An example that comes to mind here is the robotic manipulation task of picking up and putting down a cup. In this case, prior velocity and position from several previous states may be required to fully inform the agent in the current state. The important thing to note is that the state is the __mechanism through which the agent interacts with the environment__. The state keeps the agent aware of how the game/task changes, which is crucial for understanding the consequences of actions. We similarly denote the state space by \\( S \\). Sometimes the state space is small too. For instance, in tic-tac-toe there are \\( 3^9 \\) possible board configurations, many of which are duplicated by symmetry, so the number of states is around 600. However for larger games like Go, or other tasks like learning to walk, the state space can be very large, or even infinite.

Now, in a perfect world, our RL agent would know exactly which action is best in each possible situation that might occur. In general, we call a mapping from state to action a __policy__, and the goal in any RL task is to find the policy that maximizes reward. We call such a policy the __optimal policy__, and often denote policies in general with the greek letter \\( \pi \\). Mathematically, this means \\( \pi : X \rightarrow A \ \forall x \in X, \ a \in A \\). How exactly the agent deduces the optimal policy is something I discuss later on in the post.

And now for the last fundamental component: __the reward signal__. One the axioms of reinforcement learning is called the __reward hypothesis__, which basically says that the objective of any task can be boiled down to a single continuous signal that the agent will try to maximize during its interaction with the environment. In other words, this hypothesis says that the feedback needed to learn the optimal policy in any situation can be expressed as a single continuous number. This might seem controversial, and non-obvious, but while there is still no formal proof that this statement is true, in practice it often holds. For example, in many games, there is often some small, fixed reward amount for most of the game stages, with a large reward (or punishment) occuring only when one player has won (or loss). To be more concrete, you can think of a possible reward function for the game tic-tac-toe, where the agent receives a reward of 1 for an action that results in a win, -1 for an action that results in a loss or draw, and zero for every other action. It becomes clear that only a handful of states will be highly valued, even if the agent learns to think ahead. One of the underlying assumptions I have yet to make clear is that, in the RL setting, __the agent only sees the reward after it has chosen the action__. Thus, it is only through experience that the agent learns to modify its behavior. Formally, we denote the reward signal, or reward space by \\( R \\), with the reward generated at each time step \\( r_t \in R \\). As mentioned above, reward is a scalar.

## The Big Picture

The above text contained a lot of new information. Let's briefly review what we've covered. The general RL agent relies on the following components:

- __States__ contain all information about the task/game/learning process up to the current time. The state is an observation the agent receives about the current condition of the environment.
- __Actions__ are the possible choices the agent considers. The set of feasible actions may change depending on the state. In general, the agent wants to choose the action that maximizes potential future rewards.
- __Rewards__ are received by the agent after every action, and provide feedback as to the quality of the choice. The reward is a scalar, and the agent is constantly looking ahead to evaluate the current state and action while balancing immediate and long-term rewards.
- __Policy__: What the agent is ultimately after is the action at each state that will result in the greatest long-term reward. Such a mapping is called the optimal policy, which can be difficult to find, since the state-action space can be very large. 

The following picture captures the continual interaction between agent and environment quite nicely, accounting for the order in which each piece of information is received:

<center><img src="/img/rl-flow.png" alt = "RL Flow Chart"></center>

So the sequence between agent and environment goes as follows:
1. Agent begins in state \\( x_t \\)
2. Agent chooses action \\( a_t \\)
3. Agent receives reward \\( r_{t+1} \\)
4. Agent moves to state \\( x_{t+1} \\)

In general, beginning at time \\( t = 0 \\), this generates the following sequence of realizations:

$$ X_0, A_0, R_1 , X_1, A_1, R_2, X_2, A_2, R_3, \cdots $$

We usually assume some initial state. Sometimes it's obvious, like a blank checkers board, or the ground/entrance for 3D problems, while other times we can begin in a random state.

Until now, all I have said about the RL agent's objective is that it involves maximizing long-term reward. What exactly does this mean? Well using the above sequence generated by experience, we end up with a sum of rewards:

$$ R_1 + R_2 + \cdots + R_T $$

It is exactly this sum of rewards that we want the agent to maximize through its choice of actions. But simply maxizing the sum of rewards above treats future rewards identical to those experienced near the current state. This is certainly not how humans would value future rewards (ie time value of money - all else held constant, you would prefer something today to receiving it tomorrow). And in many cases, we would like our agent to place more weight on current reward than on future rewards. This is accomplished by using a __discounted sum of rewards__, which we also call the __return__ from a given state:

$$ G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots $$
$$ \Rightarrow G_t = \doteq R_{t+1} + \gamma ( R_{t+2} + \gamma R_{t+3} + \cdots) $$
$$ \Rightarrow G_t = \doteq R_{t+1} + \gamma G_{t+1} $$

The above sequence is easily computed, treating \\( \gamma \\) as a hyperparameter that controls how myopic or near-sighted the agent is. For a finite number time steps, we get that \\( G_t = \sum_{t+1}^{T}\gamma^{k-t-1}R_{k} \\). Note that the recurrent relation formed makes it easy to start at an end (terminal) state, which would have a return of 0,  and work backwards to the current state. This is one of the hallmarks of __dynamic programming__, which underlies much of the state evaluation we will see soon. But where do these rewards actually come from? Recall above, w
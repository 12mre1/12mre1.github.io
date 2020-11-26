---
layout: post
title: "A Formal Notion of Games"
date: 2020-09-13
---

#### *Prerequisite Math*: None
#### *Prerequisite Coding*: None

If you asked someone to define a __game__, there are several examples that come to mind. Many people would imagine a sport, like baseball or hockey, while others would imagine boardgames or videogames. Someone else might think of cardgames, like blackjack or poker. All these answers are correct, but they are very different in many ways. So what makes a game a game?

Since the 1930s, mathematicians have analyzed games using a very broad definition. Their goal was to predict the strategic decisions of players in various social situations. This field of research is called __game theory__ , and it has far-reaching applications, some of which I'll discuss shortly. Roughly speaking, in order for a situation to be considered a game, it needs to have the following:

1.  ***Players***. This first one is obvious. How could we analyze behavior without actors to make decisions? We don't always refer to these participants as players, because not all situations encompass the recreational nature we associate with play. They are more commonly called _actors_, or _agents_. Note that we don't necessarily need people to be the agents. For instance, biologists often study animals (e.g. predator-prey dynamics), and economists study firms (e.g. market competition).

2. ***Strategies***. This one is more subtle. It can only be a game if agents are allowed to choose. If every agent in the game had only one option, then the outcome of the game would be entirely predetermined, and we would need no analysis. How fun would it be to play baseball if the result was always the same? But let me be clear. Just because we have choices, it is not necessarily true that the choices are the same for every agent. We'll see some examples of this later on. So how do agents know which strategy to select? The third requirement is arguably the most important...

3. ***Payoffs***. These are usually called _values_ or _utilities_, but they refer to the costs or benefits received by agents when they choose a certain strategy. One interesting aspect is that payoffs are usually not determined only by the choice of one agent, but by the strategies of every agent in the game. Payoffs are important because they allow us to model an agent's preferences over different outcomes. In order to predict what strategies agents will choose, we need to know what they consider 'good' and 'bad'. We sometimes use money to denote the payoffs, because it is an intuitive representation, but not always. 

There is one more assumption we make when analyzing games. You've probably been taking it for granted until now, but for completeness, I should state it explicitly: __We assume agents choose strategies to maximize their payoffs__. In other words, we assume agents play to win (or at least to do their best, since the notion of winning can be vague). If you've ever heard of the term _rational agent_, this is exactly what it means. In the context of our games, where self-benefit is optimal for each agent, all agents will behave in whatever way they believe most increases their payoff.  

Now that we have a formal framework for specifying games, let's go through an example. To keep things simple, I'll use only two players:

<center>
<table>
<thead>
  <tr>
    <th>Player 1 \ Player 2</th>
    <th>Heads</th>
    <th>Tails</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Heads</td>
    <td>1,-1</td>
    <td>-1,1</td>
  </tr>
  <tr>
    <td>Tails<br></td>
    <td>-1,1</td>
    <td>1,-1</td>
  </tr>
</tbody>
</table>
  <i>Figure 1: A Game Matrix for Matching Pennies</i>
</center>



The matrix above is the most common representation for games. It is called **normal form** . This game is called **Matching Pennies**, and it is usually played between two people. Both players have a coin, and they flip at the same time. If the two results match (ie HH or TT), the first player wins. If the two results are different (TH or HT), the second player wins. Typically, the player who wins gets to keep both coins. In normal form representation, it is convention to write the strategies of player one as rows, and those of player 2 as columns. The payoffs associated with each pair of strategies are listed as an ordered pair (row player payoff, col player payoff). So how do the payoffs match our description of the game?

  Consider Player 1's choices (rows). Given that Player 2's coin has come up Heads, the payoffs should reflect the fact that P1 would prefer to also have Heads. Here, we give P1 a payoff of 1 if this happens (HH), and -1 if it does not (TH). Thus, P1 prefers the outcome HH to TH. Similarly if P2's coin landed Tails, P1 would prefer TT to HT, indicated by the payoff 1 > -1. For the column player, the opposite is true. P2's payoffs are greater for TH and HT (1) than they are for HH and TT (-1). Note that in this case, the magnitude of the payoffs does not determine whether the matrix fits the game we described. Only the direction matters; positive numbers represent gain, and negatives represent losses. While this is not true for all games, there are many classes of games that use similar value structures.

  So what is the benefit of using normal form representation? For starters, it is a very compact way of representing a game. It is much shorter than, say, a written description like the one I gave earlier. But more importantly, it **preserves information**. What I mean by this is that everything I told you about how the game works (and everything we need to analyze outcomes) can be understood just by reading the matrix. One other benefit, one that isn't as obvious from this example, is that we can use the matrix to put ourselves into the minds of the players, and *evaluate which strategies are good and bad*. In the case of Matching Pennies, it's not obvious how to do this, since the payoffs are **symmetric**, and the outcomes are purely determined by chance. In other words, it's not clear ahead of time which outcome (HH,HT,TH, or TT) will happen. But there are other games, like our next example, that have more interesting strategy profiles. Games where one player's loss is the other player's gain (ie Matching Pennies) are called **zero-sum games**, because the payoffs in each outcome add to zero.
  
Let's consider another game, called the **Deadlock**. It works like this:

   - Two nuclear superpowers are trying to dissolve their arsenals of nuclear weapons
   - Each country has a choice: Cooperate in the dissolution, or renege and keep their arms
   - If both countries renege, they receive the same payout
   - Each country gets a higher individual benefit when they renege and their opponent cooperates (they become relatively stronger)
   - If both countries cooperate, they are both weaker (you can think of this compared to the rest of the world)

Here is the normal form representation of this game:

<center>
<table>
<thead>
  <tr>
    <th>Country 1 \ Country 2</th>
    <th>Cooperate</th>
    <th>Renege</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Cooperate</td>
    <td>1,1</td>
    <td>0,4</td>
  </tr>
  <tr>
    <td>Renege</td>
    <td>4,0</td>
    <td>3,3</td>
  </tr>
</tbody>
</table>
  <i>Figure 2: A Game Matrix for Deadlock</i>
</center>

We can see the payoffs in this case are structured a bit differently. Let's consider the motives of C1 (rows). When evaluating strategies, it is often useful to see what one player would do if the other's choice was already known. Assuming that C2 was cooperating, we can see that C1's best option is to renege, since 4 > 1. What happens if C2 were insteading choosing to renege? Well, C1's preference would still be to renege, since 3 > 0. Stop and consider this for a second. We have found that *no matter what C2 does, C1 will always prefer to choose renege instead of cooperate*. Moreover, we could do a similar analysis for C2, and assume hypothetical strategies for C1. However, since the payoffs of this game are symmetric, we know that C2 will be in the exact same situation, and will also always prefer to renege. Since it never makes sense for either country to dissolve their arsenal, the only stable outcome of this game is for both players to renege. 

More generally, a player is said to be playing their **best response** when they are achieving their best outcome taking the strategies of others as given. Identifying the best response for each player in each situation is exactly what we've just done. When all players in the game are playing their best responses, the outcome is called a **Nash Equilibrium**. Invented by mathematician John Nash, it describes the outcome or outcomes of a game where, given the strategies of all other players, no player has any incentive to deviate to another strategy. This concept is crucial because it characterizes stability in the game. In other words, it lets us predict what will happen in the game without having to play it. 

Note that not all games have a Nash Equilibrium like this one. Think about Matching Pennies. No matter what outcome occurs, the player that lost will always wish they had deviated to the other strategy (ie if HT, P1 would have preferred TT, or similar for P2). Thus, while every player will always have a best response (there may be more than one if different strategies have identical payoffs, in which case players are indifferent between choices), there may not be an outcome where all players play their best responses together. Note that the Renege strategy in the Deadlock Game is always a best response, no matter what the other players are doing. A strategy that is always better regardless of the opponents' choices is called a **dominant strategy**. Similarly, a strategy that is never preferred (regardless of opponent strategies) is called **dominant**. In the case of Deadlock, Renege dominates Cooperate. Thus, cooperation does not happen.

I should also point out that both the examples I've shown you have several properties don't necessarily hold for all games. The following list is not exhaustive, but it does hint at some possible extensions to what we've already seen:

- ***Both players make choices at the same time***. We call this a **simultaneous game**. Alternatively, we could have set up the game so that P1 plays first, P2 sees P1's choice, then P2 gets to choose. we call games like this **sequential**, because strategies are played one after another. We usually represent sequential games with a tree, whose different levels of depth represent different stages of time. This is called **extensive form** representation.

- ***Each agent knows the other's payoffs (and their own) for every outcome***. When all players know all strategies and payoffs, it is called **complete information**. Can you think of some common games where you might not know all of your opponent's choices or payoffs? Many card games do not meet this criteria, and would instead be called games of **incomplete information**.

- ***The game is played only once.*** Consider a twist on the Deadlock game above, where the payoffs for *{Renege,Renege}* and *{Cooperate,Cooperate}* are switched. It would still make sense for both players to always renege, which would give them payoffs of (1,1). But what if they both knew they were playing the game more than once? Could they come to some sort of implicit agreement to play {Cooperate,Cooperate}? In repeated games, it is possible for agents to play strategies that do not match the single-game Nash Equilibria. 

- ***Players choose only one strategy.*** In our Matching Pennies game, both strategies are equally best for both players. This is because the outcome is due to chance. It doesn't really make sense to choose one strategy (called a **pure strategy**), unless you knew the other player was choosing only one. Intuitively, we would rather choose H 50% of the time, and T 50% of the time. Choices like this are called **mixed strategies**, because they are probability distributions over pure strategies. It is possible to have Nash Equilibria with mixed strategies.

We've covered a lot in this post, but everything is centered around our original three criteria for games (players,strategies, payoffs). We've looked at how to analyze behavior and determine how games might end before they are played. This type of framework is used extensively in many disciplines (mathematics, economics, biology, computer science, etc..) and is extremely flexible. I encourage you to try and find your own examples of games, and to analyze them. The results might surprise you.

## Further Reading

- [Here](https://www.academia.edu/3868220/John_von_Neumann_and_Oskar_Morgenstern_Theory_of_Games_and_Economic_Behavior_Sixtieth_Anniversary_Edition) is a link to the book *Theory of Games and Economic Behavior* by Jon Von Neumann and Oskar Morgenstern. It was written in the 1940s, and is considered to be the foundation of modern game theory.

- [Here](https://library.princeton.edu/special-collections/sites/default/files/Non-Cooperative_Games_Nash.pdf) is John Nash's 1950 PhD thesis, where he first introduces the concept of Nash Equilibria.

- I strongly recommend [this biography](https://www.amazon.ca/Beautiful-Mind-Sylvia-Nasar/dp/1451628420/ref=sr_1_2?dchild=1&keywords=A+Beautiful+Mind&qid=1600342567&sr=8-2) of John Nash, written by Sylvia Nassar, called *A Beautiful Mind*. It has also been made into a film.

- [This book](https://www.amazon.ca/Art-Strategy-Theorists-Success-Business/dp/0393337170/ref=sr_1_2?dchild=1&keywords=The+art+of+strategy&qid=1600342748&sr=8-2), called *The Art of Strategy*, contains many practical examples of game theory.


  

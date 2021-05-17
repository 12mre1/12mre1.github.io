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

So what exactly did I mean earlier when I mentioned matching? Suppose we have \\(n\\) medical residents, \\(R = { r_1 , \cdots \ r_n } \\) who are applying for residencies, and there are \\( n \\) hospitals, \\( H = { h_1 , \cdots \ h_n } \\), who receive residency applications. Assume also that each hospital can hire only one resident, and each resident can only work for one hospital. Assume also that each resident has a __preference ordering__ over hospitals, and each hospital has a __preference ordering__ over applicants (residents). A preference ordering just means a clear ranking - if a resident receives two job offers, it is clear to that resident which offer they prefer (and a similar statement applies for each hospital). Note that preference orderings needn't be the same between hospitals or residents (in fact they almost never are).

Suppose that every hospital sends out their initial job offer at the exact same time. Suppose also that each resident can accept or reject any offer they are given. You can see right away that it is very unlikely that each initial offer will be accepted - in order for this to happen, the preferences of each resident would have to mesh perfectly (ie each resident has a different _favorite_ hospital), and each hospital would have to coincidentally send their offer to the applicant that prefers them. What is more likely is that some residents will turn down offers for others, and initial pairings will end up switching as long as there is still time. In other words, the process is not __self-enforcing__; If everyone acts in their own self-interest, the process could fail. Moreover, many residents and hospitals could be very unhappy. The Gale Shapley Algorithm gives us a way of finding a set of __perfect matchings__, ie resident-hospital pairs where no one will switch.

In terms of the sets we defined above, a __Matching__ is a set of ordered pairs \\( S \\) such that each \\( r \in R, \ h \in H \\)
appears at most once in \\( S \\). A __Perfect Matching__ is a similar set, where each \\( r \in R, \ h \in H \\) appears _exactly once_. In terms of our example, a perfect matching only happens if every resident is placed in exactly one hospital, and every hospital has exactly one resident. But we don't just want to find a perfect matching. We also want to find one that is stable. A matching is __stable__
if:

(i) It is perfect, and 

(ii) There is no instability with respect to \\( S \\). In other words, given some matching \\( S \\), no unmatched pair wishes to deviate to a different matching. More on this later.

Given our definition of stable, is it possible to construct an algorithm that guarantees a stable matching? Yes - The GS algorithm does just that (we will prove so shortly).

## The Algorithm

First, i'll walk through the algorithm at a high level. Then I present the python code. It works as follow. Initially, every resident and every hospital is unmatched. An unmatched hospital \\( h \\) chooses the resident \\( r \\) that ranks highest on their preference list, and sends them a job offer. At some point, another hospital \\( h' \\) might also offer \\( r \\) a residency spot, but in the mean time, \\( r \\) _verbally commits_ to \\( h \\). This verbal agreement represents an intermediate state that is necessary for the algorithm to produce a stable matching. You can think of it like an engagement between a couple that usually turns into marriage.

While some hospitals and residents are free (not matched), an arbitrary free hospital \\( h \\) chooses their highest ranked resident \\( r \\) to whom they have not already sent an offer, and sends them a residency offer. If that resident is free, the two enter a verbal commitment. Otherwise, \\( r \\) has already committed to the residency at \\( h' \\), and decides based on preferences whether to abandon that commitment in favor of matching with \\( h \\). If \\( r \\) decides to remain in current commitment, then \\( h \\) simply sends an offer to their next highest-rated resident, and the process repeats itself. The algorithm terminates when there are no free hospitals or residents.

Here is the python code for this algorithm:

```python
### Gale-Shapley Algorithm ###
import numpy as np
import json
N = 10 # Number of Ordered Pairs/ Hospitals/ Residents

# Define freedom lists (initially everyone is free)
r_free = ['f' for key in range(N)]
h_free = ['f' for key in range(N)]

# Define Preference Lists (10 x 10 array for both R and H)
h_pref = [np.random.choice(range(N),(1,N), replace = False).flatten().tolist() for i in range(N)]
r_pref = [np.random.choice(range(N),(1,N), replace = False).flatten().tolist() for i in range(N)]

# Define list to track proposals
h_offers = [[] for i in range(N)]

# Define list to store pairs
pairs = {}
```
To start, I define several data structue. Lists keep track of which hospitals and residents are unmatched. I use `numpy`'s random number generation to instantiate preferences, which I then convert to nested lists. I also use a nested list to keep track of offers made by hospitals to residents. Finally a dictionary stores the resident-hospital pairings as key-value pairs. This is intentional - knowing that in our algorithm, and existing pair can only be broken if the resident decides so, access by resident was easier than by hospital. Hence the use of residents, and not hospitals as keys for the dictionary. Here is the main algorithm:

```python
# Track the number of offers to terminate the loop below
n_offers = len([item for sublist in h_offers for item in sublist])

# While there is a free hospital that hasn't proposed to everyone
while 'f' in h_free and n_offers < N*N:
  # Find an available hospital
  h = h_free.index('f') # index method finds first occurence
  # Define r as highest ranked resident in h's preferences to whom h has not offered
  opts = [pref for idx, pref in enumerate(h_pref[h]) if idx not in h_offers[h]]
  r = np.argmax(opts)
  # If r is free
  if r_free[r] == 'f':
    # h and r match
    print('Pair ({},{}) Created'.format(h,r))
    pairs[r] = h
    r_free[r] = 'm'
    h_free[h] = 'm'
    # h can only propose to r once
    h_offers[h].append(r)
  # Else if r is already committed to h'
  else:
    # Find r's current partner
    h_prime = pairs[r]
    # If r prefers h' to h
    if r_pref[r][h_prime] >= r_pref[r][h]:
      # h remains free, but still can only propose to this r once
      h_offers[h].append(r)
    # Else r prefers h to h'
    else:
      # r and h commit
      print('Pair ({},{}) Created'.format(h,r))
      pairs[r] = h
      # h is no longer free
      h_free[h] = 'm'
      h_free[h_prime] = 'f'
      # h can only propose to r once
      h_offers[h].append(r)
  # Update number of offers
  n_offers = len([item for sublist in h_offers for item in sublist])
```
I've provided clear comments at each step, but I'll walk through what happens again:
1. All residents and hospitals begin free (unmatched)
2. While there is a hospital that is unmatched and has not sent offers to every resident: do
3. Choose such a hospital \\(h \\) (the first in the array by default)
4. Find the highest-ranked resident in \\( \\)'s preferences \\( r \\) to whom \\( h \\) has not sent an offer
5. If \\( r \\) is free, then (\\( h, r \\)) become partners
6. Else \\( r \\) is committed to \\( h' \\)
7. If \\( r \\) prefers \\( h' \\) to \\( h \\):
8. \\( h \\) remains free
9. Else (\\( h, r \\)) become partners, and \\( h' \\) goes free
10. Return the set of matched pairs.

Notice above, i've defined the algorithm for a general number of pairings, N. Above, I set this to 10, and we get the following output:
```python
Pair (0,0) Created
Pair (1,9) Created
Pair (2,5) Created
Pair (3,8) Created
Pair (4,1) Created
Pair (5,1) Created
Pair (4,3) Created
Pair (6,6) Created
Pair (7,7) Created
Pair (8,9) Created
Pair (1,2) Created
Pair (9,4) Created
{0: 0, 9: 8, 5: 2, 8: 3, 1: 5, 3: 4, 6: 6, 7: 7, 2: 1, 4: 9}
```
Note that, due to the randomness in preference generation, your output may look slightly different.

## The Pros And Cons

It is fairly straightforward to show that the GS Algorithm returns a stable and perfect matching (I will not run through proofs here, but you can see Kleinberg and Tardos in the Further Reading Section for a great proof). But even so, there are a number of interesting consequences that may not be so obvious:

__1. Given a specific set of preferences, all executions of this algorithm will yield the same matching.__

This may not be obvious, but let's think about it further. Suppose a given hospital would prefer resident \\( r_1 \\) to all others. Since the hospitals make the offer, but the residents decide whether to remain committed, the only way \\( r_1 \\) does not end up completing his/her residency at the given hospital is if \\( r_1 \\) prefers another hospital __and gets an offer from that hospital__. If this happens, then clearly the current hospital resident pairing is not a stable matching. Using the same logic for the other hospitals, in order for the algorithm to terminate, it must be the case that none of the other hospitals are unhappy with their residents (no hospitals want to trade). Since the ordering of offers and committments is based on a fixed set of preferences, offers go out in an identical order, and thus, the eventual matchings are constant (as long as preferences do not change).

__2. The runtime of the Gale-Shapley algorithm is \\( O(N^2) \\)__.

This one makes sense. In the worst-case scenario, each of N hospitals has to send offers to each of N different residents. This results in N offers for each resident, or \\( N^2 \\) iterations of our while loop. Note that the algorithm is often faster than this, but it serves as the best known upper bound.

__3. The algorithm above yields a stable matching that is the best for all hospitals among all stable matchings__.

This point is subtle, but has tremendous consequences. What is says is that each hospital receives the best resident they can hope to have after all offers are settled. To see why this is true, let's assume the opposite, and show a contradiction. Suppose some hospital is paired with a resident who is not their _best valid partner_ (highest preference among stable outcomes). Since hospitals send offers in decreasing order of preference, this would mean that the hospital was rejected by some valid partner earlier (a resident who received a better offer). But in order for this to have happened, the earlier resident must have received an offer from a hospital they prefer, meaning the pairing of current hospital to earlier resident is not a stable matching. Thus the algorithm could not have returned it, and that earlier resident is not a valid partner of the current hospital (contradiction). This we have that every hospital gets the best possible resident they could hope for in a stable outcome. Note that if the residents were sending offers instead of hospitals, the same property would hold for the residents. In this algorithm, the proposing side gets an advantage.

__4. The algorithm above yields a stable matching that is the worst for all residents among all stable matchings__.

Using similar logic to that above, we can see that the residents are at a disadvantage for having to wait for offers from the hospitals. Among all possible hospitals who might send them offers, they end up with the worst possible option. In this sense, the algorithm displays __unfairness__.

## Conclusion

This is a key algorithm in the fields of economics and computer science, with many applications like scheduling, facility location, social media recommendations, and urban planning. I recommend you look more into some of these use cases. I may write a later post on the more general problem of Stable Matching, which goes into more detail about Graph Theory (don't worry if you don't know what that is). For now, here are the key takeaways:

- The GS algorithm is designed to compute stable matchings between two sets of elements, based on preferences of individual elements
- A stable matching is one where no pairs wish to deviate from their current mapping
- The algorithm runs fairly quickly, with a polynomial-time upper bound
- The GS algorithm is biased towards the set that proposes partnership

## Further Reading

- I recommend the first chapter of _Algorithm Design_ by Kleinberg and Tardos. It has an excellent runthrough of GS.
- The Wikipedia page on the Gale-Shapley Algorithm is actually quite helpful, especially in placing the Stable Matching problem in a larger context
- There is a great set of slides by Kevin Wayne (though a bit old now), who teaches algorithms at Princeton. Just google these slides, they should be publically available.
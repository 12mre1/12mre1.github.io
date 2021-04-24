---
layout: post
title: "Tic-Tac-Toe in 100 Lines of Python"
date: 2021-04-22
---

## Tic-Tac-Toe in 100 Lines of Python 

_Prerequisite Math: None_

_Prerequisite Coding: Python (Numpy)_

One of the most popular board games worldwide is __Tic Tac Toe__, also known to some as __Naughts and Crosses__. Today I'm going to walk through how you can set up a simple two player game using (roughly) 100 lines of Python (not accounting for spaces of course). I think you'll find that the trick to making games like this is not so much the rules of the game, but how you represent them using data structures. I believe this code can be easily implemented in another language, if you would like to try building it yourself (I prefer python, but I know others may have their go-to as well). Before we get started with the coding of the game, let's just refresh ourselves on the rules:

    - 1. The game is traditionally played on a 3x3 board
    - 2. One player is represented by Xs, and the other by Os (hence the game's title)
    - 3. Players take turns placing either an X or an O on the board
    - 4. The first player to achieve 3 of their own pieces in a row, column, or diagonal is the winner
    - 5. It is possible to tie (in my code that follows, there will simply be no options for placing pieces)

My goal in doing this was to make a simple working version of the game that covers these rules, and (this is important for any game) was not difficult to play. My game is contained in a single python file, designed to be run on a bash shell (but you could easily run it on another system that can execute python files). So where to begin?

There are many different places to start a small project like this, but to me, it seemed obvious to begin with how to represent the game board. Remember we want a way to represent a 3x3 board (grid) that allows us to act on specific board positions. There are several ways you might do this. I use a __numpy array__, which is essentially a 3x3 matrix that is sliceable (meaning we can access specific entries by position). As an aside, NumPy is a scientific computing library designed to enable fast calculations between multi-dimensional arrays, and is great for turning algebraic equations straight into code. (Shameless plug) I encourage you to visit the teaching materials section of this site, for an in-depth tutorial on numpy and several other popular python packages. But back to the game. To start, I import the numpy package, and define a board:

```python
import numpy as np

# Instantiate game board (0 = blank, 1 = X, 2 = O)
board = np.zeros((3,3))
```
Running this code instantiates a 3x3 game board (grid) that we can modify later depending on player input. This naturally extends to our next problem: How will players refer to (and possibly choose) different spots on our board? We could use some sort of text-based system (e.g. 'Place an X on the top left square'), but this is tedious, and not very efficient. What I'll do instead is define a different 3x3 array, with each cell labelled as a floating point number (0-8, since python is zero-indexed), that players of the game can refer to when deciding where to place their pieces.
```python
# Define game board positions
board_positions = np.array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]])
```
Note that later on, we can print this array to remind players which number corresponds to which board position. Now you might be thinking: we've defined these numbers (0-8) somewhat arbitrarily. How can we make sure they correspond to the correct position on our game board? Essentially what we need is a __mapping__ from these floats representing board positions, to the actual locations within our game board. Remember how I told you that numpy arrays are sliceable? This means that I can access a certain element using index numbers. So to access the top left entry of the board, I would use the (0,0) tuple, which stands for first row, first column. Similarly, I could use (0,1) to represent the first row, second column, and so on and so forth for the remaining positions on our game board. So we're looking for a mapping from the floats we defined above, to these indices corresponding to our actual game board positions. Good news! Python's __dictionary__ data structure is perfect for this. It maps a set of keys (the floats) to values (the board indices), without tracking the order of the entries. This makes it faster, and excellent for our purposes, since the only time we'll use the mapping is to convert player input (their moves) to board ajdustments. I define this mapping below. One other thing I do is create a similar dictionary that stores the player symbols as indices (X will be denoted 1, and O as 2), for reasons that I'll explain shortly.

```python
# Maps player choice (float) to board position
board_idx = {0.0: (0,0), 1.0: (0,1), 2.0: (0,2),
            3.0: (1,0), 4.0: (1,1), 5.0: (1,2),
            6.0: (2,0), 7.0: (2,1), 8.0: (2,2)}
# Maps player symbol to index number (also board value)
player_idx = {'X': 1.0, 'O': 2.0}
```
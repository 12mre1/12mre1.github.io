---
layout: post
title: "Tic-Tac-Toe in 100 Lines of Python"
date: 2021-04-22
---

## Tic-Tac-Toe in 100 Lines of Python 

_Prerequisite Math: None_

_Prerequisite Coding: Python (Numpy)_

One of the most popular board games worldwide is __Tic Tac Toe__, also known to some as __Naughts and Crosses__. Today I'm going to walk through how you can set up a simple two player game using (roughly) 100 lines of Python (not accounting for spaces of course). I think you'll find that the trick to making games like this is not so much the rules of the game, but how you represent them using data structures. I believe this code can be easily implemented in another language, if you would like to try building it yourself (I prefer python, but I know others may have their go-to as well). Before we get started with the coding of the game, let's just refresh ourselves on the rules:

    1. The game is traditionally played on a 3x3 board
    2. One player is represented by Xs, and the other by Os (hence the game's title)
    3. Players take turns placing either an X or an O on the board
    4. The first player to achieve 3 of their own pieces in a row, column, or diagonal is the winner
    5. It is possible to tie (in my code that follows, there will simply be no options for placing pieces)

My goal in doing this was to make a simple working version of the game that covers these rules, and (this is important for any game) was not difficult to play. My game is contained in a single python file, designed to be run on a bash shell (but you could easily run it on another system that can execute python files). So where to begin?

There are many different places to start a small project like this, but to me, it seemed obvious to begin with how to represent the game board. Remember we want a way to represent a 3x3 board (grid) that allows us to act on specific board positions. There are several ways you might do this. I use a __numpy array__, which is essentially a 3x3 matrix that is sliceable (meaning we can access specific entries by position). As an aside, NumPy is a scientific computing library designed to enable fast calculations between multi-dimensional arrays, and is great for turning algebraic equations straight into code. (Shameless plug) I encourage you to visit the teaching materials section of this site, for an in-depth tutorial on numpy and several other popular python packages. But back to the game. To start, I import the numpy package, and define a board:


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
So we have a basic representation of our board, and we know (roughly) how to access specific elements if we need to. From here, I'll define several functions that we'll used to execute the game. The first of these functions is designed to accept and validate user input. We know the game will start with a blank board, so we'll need to prompt a given player to enter their response, and accept, validate, and store their input. Recall earlier that we defined a grid of numbers with those floating point numbers. These floats can be used by players to dictate their moves in the game. 

```python
def get_valid_move(game_board):
    '''Generate and validate user input.'''
    # Generate list of valid moves
    valid_moves = np.where(game_board.flatten() == 0.0)[0]
    print('Possible valid moves:\n\n', valid_moves,'\n')
    x = float(input('Enter a value (0-8):'))
        # Check that that position is available
    if x in valid_moves:
        return x
    else:
        print('Sorry, you can only play on available board positions.\n\n')
        x = get_valid_move(game_board)
        return x
```
Note that the function above is intermediate - I will use it later on as a piece of the game play pipeline. But here's what's going on in this code. Recall that, when we initialized the game board earlier, we start with a matrix of zeros - thus, zero is used to represent a blank space on the board. The `valid_moves` object in the function above represents an array containing all positions that are blank (the game board value is zero). The list of valid moves is printed for the player. Next I use the built-in `input` function, which will take the choice of the player and store it. If the player's input is valid (lies within the list of valid moves), we allow that choice to be returned. If not, we run the function again.

So now we have a way of collecting and validating player input. What we'll need next is a way of updating our board based on player choices. The following function accomplishes this by accepting as an argument the user input we collected above. Note that in this case, how we update the board will depend on which player gave their input. Notice that we update the board by updating the specific entry indicated through the mapping we defined earlier.

```python
def adjust_board(user_input, game_board, player):
    '''Takes user input and updates the game board'''
    idx = board_idx[user_input]
    # print('Index you chose:', idx)
    if game_board[idx] == 0:
        game_board[idx] = player_idx[player]
    else:
        print('Sorry, you must choose a blank square.')
    return game_board
```
Recall also that I defined a mapping from player symbol to the associated value. This is useful, because it gives us a very easy way of changing a blank space on the board from a zero to a one (if X) or two (if O). So now, we have the ability to take player input, and use it to update our game. What's missing? Well, for starters, it may not be easy for players to make decisions using a board of numbers. So in order to make the user interface a bit nicer, we should have a way of displaying our game board with blanks, Xs and Os instead of zeroes, ones and twos. To do this, we can create a simple substitution function that makes a copy of the current (floating point) game board, and replaces those values with their symbolic counterparts.

```python
def symbol_board(game_board):
    '''Print game board with symbols instead of numbers'''
    new_board = game_board.astype(str)
    new_board[new_board == '0.0'] = ' '
    new_board[new_board == '1.0'] = 'X'
    new_board[new_board == '2.0'] = 'O'
    return new_board
```
The above code is fairly straightforward, so I won't expand on it too much. Note numpy arrays have homogeneous datatypes, so we need to coerce the entire boards' entries into strings before making the substitiutions you see above. Another function I use is just a wrapper function designed to engage the board update and prompt the user to input their choice. The following function just uses two of the functions we defined above:

```python
def player_move(player, game_board):
    '''Update board according to player choice
    player - the player symbol (string): X or O'''
    # Remind user of current board
    print('Here is the current board:')
    print('--------------------------\n')
    print(symbol_board(game_board), '\n')
    # State which player is moving
    print('Player',player,'where would you like to put an',player,'?\n')
    # Remind player of board positions
    print('Here are the board positions you might choose from:')
    print('---------------------------------------------------\n')
    print(board_positions, '\n')
    # Take and validate user input
    user_input = get_valid_move(game_board)
    # print('You selected:', user_input)
    # Update board 
    game_board = adjust_board(user_input, game_board, player)
```
If you look at the details of the above function, you'll see it's mostly designed to inform the player how exactly to make their decision. At each new turn, we print the current game board (using symbols, not floats), we list the available possibilities, and we identify whose turn it actually is. Then we take the user input, and update the board accordingly. One other detail that I haven't mentioned up to this point is that players will alternate turns, and traditionally player O begins the game. To account for this, I define the following small function:

```python
# Update whose turn it is
def update_turn(player):
    if player == 'X':
        player = 'O'
    elif player =='O':
        player = 'X'
    return player
```
This just alternates the player variable depending on whose turn it is. So let's review what we have so far. We have our game board, represented by a numpy array, with an accompanying array showing players which float corresponds to which board position. We have functions designed to take and validate user input (the choice of float), and another function to update to game board accordingly. We also have a function designed to alternate player O and X, with O starting first. We almost have enough for the complete game, but we're missing one important detail - how do we know when a player wins? Recall the rules I mentioned in the beginning - we need to check to see if a given player has three consecutive pieces. If they do, we should declare them the victor. The following function accomplishes this:

```python
def check_victory(game_board, player, win):
    '''Check to see if a player has won'''
    # Whether we look for 1s or 2s depends on player
    test_cond = np.ones((3,1))*player_idx[player]
    # Check board
    # Board transpose needed to check off diagonal
    board_flip = np.transpose(game_board)
    for i in range(3):
        # Check rows
        row_compare =  np.transpose(game_board[i]) == test_cond
        if row_compare.all():
            win = True
        # Check cols
        col_compare = np.transpose(game_board[:,i]) == test_cond
        if col_compare.all():
            win = True
        # Check diags
        diag_compare = np.transpose(np.diagonal(game_board)) == test_cond
        if diag_compare.all():
            win = True
        trans_compare = np.diagonal(board_flip) == test_cond
        if trans_compare.all():
            win = True
        
    if win == True:
        print('-----------------------------------\n')
        print('\nCongratulations',player,'You won!')
    return win
```
This code is designed to be run after each player's turn. Looks at the current state of the game, checks the victory conditions listed above, and if it finds that one of the victory conditions is true, it declares a winner. Note the presence of the boolean variable `win`, which we instantiate as false, and will only be true if one of the players makes a winning move. As I mentioned before, it is possible to draw, in which case the game needs to be manually reset. So we have everything we need. Let's run the game! I use a simple while loop to execute the above functions repeatedly:
```python
# Define win flag
win = False
# Os goes first
player = 'O'

print('\nWelcome to Naughts and Crosses. As is customary, Player O will go first. Good luck!\n')
print('--------------------------------------------------------------------------------------')
# While no-one has won
while win == False:
    # Execute move
    player_move(player, board)
    # Check board for victory
    win = check_victory(board, player, win)
    # Swap players
    player = update_turn(player)
```
Running this python file will give the following output (I use a bash shell, but you can execute this in whatever environment you normally use):
```bash
Welcome to Naughts and Crosses. As is customary, Player O will go first. Good luck!

--------------------------------------------------------------------------------------
Here is the current board:
--------------------------

[[' ' ' ' ' ']
 [' ' ' ' ' ']
 [' ' ' ' ' ']]

Player O where would you like to put an O ?

Here are the board positions you might choose from:
---------------------------------------------------

[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]

Possible valid moves:

 [0 1 2 3 4 5 6 7 8]

Enter a value (0-8):
```
Suppose the first move from player 1 is to go top right (position 0). Here's what the update looks like:
```python
Here is the current board:
--------------------------

[['O' ' ' ' ']
 [' ' ' ' ' ']
 [' ' ' ' ' ']]

Player X where would you like to put an X ?

Here are the board positions you might choose from:
---------------------------------------------------

[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]

Possible valid moves:

 [1 2 3 4 5 6 7 8]

Enter a value (0-8):
```
That's all there is to it! I encourage you to build this, or a similar game yourself. It's quite fun, and you'd be surprised what you can get done in a short amount of time. You might extend what I've done, to count the number of moves, or to use a larger board. One of the reason's I built this myself was because I plan to build a Reinforcement-Learning Agent to play against. Tic-tac-toe is one of the simplest controlled environments to train computers on. Whatever you chose, I recommend using python, because of its simplicity and power.
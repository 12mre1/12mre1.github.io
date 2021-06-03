---
layout: post
title: "Arrays vs Linked Lists"
date: 2021-05-20
---
_Prerequisite Math: Big-O Notation_

_Prerequisite Coding: Python (Classes and Methods)_

## Comparing Arrays and Linked Lists in Python

In today's post, I'm going to be talking about two fundamental data structures used in computer science: __arrays__ and __linked lists__. I'll explain what each is, what advantages each has compared to the other, then we'll code each of these structures in python. Even though python has basic structures in place that very much resemble what I'm about to show, I find it's very good practice to implement these yourselves. Doing so helped me understand them when I first learned about data structures and algorithms.

## The Array

Let's start with a metaphor. Suppose you and your friends are trying to find seats for a hockey game. The memory of a computer looks a lot like the arena seats you and your friends would search for - a series of blocks (seats), each able to store one item (seat one person), and each block/seat having a unique address. And just like a computer storing information, there are a number of ways you and your friends might choose where to sit. An __array__ is a method for storing information that uses contiguous (sequential) slots. In this case, this means that you and your friends all sit side-by-side. What's unique about an array is that each position is numbered. In one dimension, this means that each position is given an index from left to right. Be warned, in many languages, this starts at zero, not at one (python is zero-indexed). Because the array is a set of continuous blocks, the size must be defined before any items are placed in memory. This means you choose how many seats to reserve before you and your friends show up on game day.

<center><img src="/img/array.png" alt = "digits"></center>

Why index every position? Well this makes it very easy to find any element in the array (by index). For example, what if we want to find the 5th element in the array? The general formula for finding the i-th element is:

$$ array_address + elem_size \times (i - first_index) $$

The only information we need to completely specify the array is the address of the first element, and the array size. In the above formula we also need the element size, but this is usually 1 (1 item per slot in memory). Some rare applications may require you to sum up different sizes to find items later in the array, but for now we just assume, like the indices, that each item has 1 slot. So to find the 5th item, we look in position  0 + 1*(5-1) = 4. Thus, index 4 contains the 5th item in our array. In general, arrays are very good at __reading__ or retrieving items, thus the read operation using arrays is \\( O(1) \\). What about adding an element to an array? Typically items are added from left to right. Adding an item to a list is called __pushing__, and removing an item is called __popping__. 

What happens if we want to add an item to the back of an array? As long as the array is not full, that's no problem. We just assign our new item to the earliest open slot allocated to the array. Thus, back insertion is \\( O(1) \\). What about adding to the front? Remember, every item in our array is indexed, so adding to the front involves re-indexing (moving) every item already in the list. In the worst case, this operation is \\( O(N) \\). For deletion, we have a similar story. Deleting from the back is no problem - this is \\( O(1) \\). But from the front, we run into the same problem. When the first element is removed, all remaining elements drop one in index. This involves computation on every element in the array, which is \\( O \\). 

One last problem with arrays is that in order for them to be defined, there must be an available sequence of memory slots equal to or greater than the size you request. For an array of size 10, this is not likely to be an issue, but for arrays with hundreds of elements, it could be that the array just won't fit. So in some sense, this structure is only feasible in certain situations. So how can we implement this in python? Unfortunately python does not have its own version of an array (though the list data type is very similar), but we can define our own. Here is a simple array class, that accepts size as an initialization parameter:

```python
class Array:
    ''' An array data structure '''
    def __init__(self, size):
        # instantiate the empty array
        self.array = []
        self.length = 0
        self.size = size

    def __repr__(self):
        return '{}, array of length {}'.format(self.array, self.length)
    
    def push_back(self, elem):
        ''' add an element to the back of the array '''
        if self.length == self.size:
            raise Exception('Array is full')
        self.array += [elem]
        self.length += 1

    def push_front(self, elem):
        ''' add an element to the front of the array '''
        if self.length == self.size:
            raise Exception('Array is full')
        new_array = []
        new_array += [elem]
        for elem in self.array:
            new_array += [elem]
        self.array = new_array
        self.length += 1

    def pop_back(self):
        ''' Remove last item from list, and return it'''
        if self.length == 0:
            raise Exception('Sorry, nothing to pop')

        # Get the last item
        pop_elem = self.array[-1]
        # Instantiate new array
        new_array = []
        # add all but last items to new array
        for elem in self.array[:-1]:
            new_array += [elem]
        # Change old array to new array
        self.array = new_array
        # Modify length
        self.length -= 1
        # Return popped element
        return pop_elem

    def pop_front(self):
        ''' Remove and return the front element for the list '''
        if self.length == 0:
            raise Exception('Sorry, nothing to pop')

        front_elem = self.array[0]
        new_array = []
        for elem in self.array[1:]:
            new_array += [elem]
        self.array = new_array
        self.length -= 1


```
Now, let's talk through each of the methods. The first three methods are sometimes refered to as __dunder__ (double-underscore) or __magic__ methods, and they are built into python's class definition procedures. I use two of them - the initialization method tells us what to include and declare when instances of the class are created, while the representation of the class specifies how instances of the class will be represented. In other words, we can customize what happens when we print an Array object. Notice also that I define a length parameter, which will track how many items are in the array, so we know when the array is full.

## The Linked List

## Conclusion

## Further Reading 
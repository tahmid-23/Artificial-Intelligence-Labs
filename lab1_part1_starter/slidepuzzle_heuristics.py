# Lab 1, Part 2a: Heuristics.
# Name(s): 
from search_heuristics import *
from slidepuzzle_problem import *

INF = float('inf')

#### Lab 1, Part 2a: Heuristics #################################################

# Implement these two heuristic functions for SlidePuzzleState.

""" Return the Hamming distance (number of tiles out of place) of the SlidePuzzleState """
def slidepuzzle_hamming(state : SlidePuzzleState)  -> float:
    size = len(state.tiles)
    ret = 0
    for x in range(size):
        for y in range(size):
            if size * x + y != state.tiles[x][y]:
                ret += 1
    return ret

""" Return the sum of Manhattan distances between tiles and goal of the SlidePuzzleState """
def slidepuzzle_manhattan(state : SlidePuzzleState)  -> float:
    size = len(state.tiles)
    ret = 0
    for y in range(size):
        for x in range(size):
            thing_thats_there = state.tiles[x][y]
            xdiff = thing_thats_there // size
            ydiff = thing_thats_there % size
            score = abs(xdiff - x) + abs(ydiff - y)
            ret += score
    return ret

SLIDEPUZZLE_HEURISTICS = {
    "Zero" : zero_heuristic, 
    "Arbitrary": arbitrary_heuristic, 
    "Hamming" : slidepuzzle_hamming,
    "Manhattan" : slidepuzzle_manhattan
    }


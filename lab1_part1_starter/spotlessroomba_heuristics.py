from search_heuristics import *
from spotlessroomba_problem import *

"""
In order to clear all tiles, the roomba must first clear a single dirty tile.
From the position of the first dirty tile, the roomba must eventually reach the farthest dirty tile.
This heuristic computes the minimum sum of distances.

This is admissible because the roomba must reach both the first tile and the farthest tile from that first tile, 
so it optimistically predicts that no intermediate tiles must be cleaned and computes the distance to travel to both tiles.


This is not consistent: counterexample

We can represent the roomba problem as a metric space in R2 using the manhattan distance as a metric.
Dirty spots at (-1, -1), (1, 2), (3, 3)
Suppose A = (0, 0), B = (0, 1)

h(A) = d((0, 0), (-1, -1)) + d((-1, -1), (3, 3)) = 2 + 8 = 10
h(B) = d((0, 1), (1, 2)) + d((1, 2), (-1, -1)) = 2 + 5 = 7
cost(A, B) = 1

h(A) - h(B) = 3, cost(A, B) = 1, but 3 > 1
"""
def spotlessroomba_min_spot_plus_farthest_heuristic(state : SpotlessRoombaState)  -> float:
    min_dist = INF

    for r1 in range(state.get_height()):
        for c1 in range(state.get_width()):
            if state.get_terrain(Coordinate(r1, c1)) in (DIRTY_CARPET, DIRTY_FLOOR) : 
                max_dist = -INF
                for r2 in range(state.get_height()):
                    for c2 in range(state.get_width()):
                        if r2 == r1 or c2 == c1:
                            continue

                        max_dist = max(max_dist, abs(r2 - r1) + abs(c2 - c1))

                min_dist = min(min_dist, abs(r1 - state.position.row) + abs(c1 - state.position.col) + max_dist)

    return min_dist

"""
"""
def spotlessroomba_second_heuristic(state : SpotlessRoombaState)  -> float:
    pass

# TODO if you wish, implement more heuristics!

# TODO Update heuristic names and functions below. If you make more than two, add them here.
SPOTLESSROOMBA_HEURISTICS = {"Zero" : zero_heuristic,
                        "Arbitrary": arbitrary_heuristic, 
                        "Custom Heur. 1": spotlessroomba_min_spot_plus_farthest_heuristic,
                        "Custom Heur. 2" : spotlessroomba_second_heuristic
                        }

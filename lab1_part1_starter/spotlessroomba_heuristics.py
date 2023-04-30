import heapq
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

Edge case: no dirty tiles
If so, return 0

Edge case: only one dirty tile
If so, use regular manhattan
"""
def spotlessroomba_min_spot_plus_farthest_heuristic(state : SpotlessRoombaState)  -> float:
    if len(state.dirty_locations) == 0:
        return 0

    if len(state.dirty_locations) == 1:
        return abs(state.dirty_locations[0].row - state.position.row) + abs(state.dirty_locations[0].col - state.position.col)
    
    min_dist = INF

    for i, dirty_tile1 in enumerate(state.dirty_locations):
        max_dist = -INF

        for j, dirty_tile2 in enumerate(state.dirty_locations):
            if i != j:
                max_dist = max(max_dist, abs(dirty_tile2.row - dirty_tile1.row) + abs(dirty_tile2.col - dirty_tile1.col))

        min_dist = min(min_dist, abs(dirty_tile1.row - state.position.row) + abs(dirty_tile1.col - state.position.col) + max_dist)


    return min_dist

"""
This first computes the minimum manhattan distance to any single dirty spot.
That heuristic is already known to be admissible and consistent.

Then, we run Prim's algorithm to find the edge weights of the minimum spanning tree of the graph where
V = dirty tiles
E = manhattan path between dirty tiles
Edge weights are added to the heuristic as the algorithm runs.

Traversing the MST is admissible and consistent for the same reason that the closest dirty tile heuristic is

Edge case: no dirty tiles
If so, return 0
"""
def spotlessroomba_manhattan_mst(state : SpotlessRoombaState)  -> float:
    if len(state.dirty_locations) == 0:
        return 0

    h = INF

    for dirty_tile in state.dirty_locations:
        h = min(h, abs(dirty_tile.row - state.position.row) + abs(dirty_tile.col - state.position.col))

    pq = []
    visited = [False] * len(state.dirty_locations)
    visited[0] = True

    root_tile = state.dirty_locations[0]
    for end_index, end_tile in enumerate(state.dirty_locations):
        if end_index != 0:
            heapq.heappush(pq, (abs(root_tile.row - end_tile.row) + abs(root_tile.col - end_tile.col), end_index, end_tile))

    edge_count = 0
    while edge_count < len(state.dirty_locations) - 1:
        (dist, end_index, end_tile) = heapq.heappop(pq)
        
        if visited[end_index]:
            continue

        visited[end_index] = True

        for next_index, next_tile in enumerate(state.dirty_locations):
            if not visited[next_index]:
                heapq.heappush(pq, (abs(end_tile.row - next_tile.row) + abs(end_tile.col - next_tile.col), next_index, next_tile))
        
        h += dist
        edge_count += 1

    return h

# TODO if you wish, implement more heuristics!

# TODO Update heuristic names and functions below. If you make more than two, add them here.
SPOTLESSROOMBA_HEURISTICS = {"Zero" : zero_heuristic,
                        "Arbitrary": arbitrary_heuristic, 
                        "Custom Heur. 1": spotlessroomba_min_spot_plus_farthest_heuristic,
                        "Custom Heur. 2" : spotlessroomba_manhattan_mst
                        }

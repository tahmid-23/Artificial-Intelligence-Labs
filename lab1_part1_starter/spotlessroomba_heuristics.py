# Lab 1, Part 2c: Heuristics.
# Name(s): Maxwell Gong, Tahmid Zaman
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

Proof that manhattan distance is a metric:
Let p1 = (x1, y1), p2 = (x2, y2), p3 = (x3, y3), p1 != p2, p2 != p3, p1 != p3

1. 
a. d(p1, p2) != 0
d(p1, p2) = |x1 - x2| + |y1 - y2|
x1 != x2, y1 != y2 -> |x1 - x2| != 0, |y1 - y2| != 0 -> d(p1, p2) != 0

b. d(p1, p1) = 0
d(p1, p1) = |x1 - x1| + |y1 - y1| = 0

2. d(p, q) = 0
d(p1, p2) = |x1 - x2| + |y1 - y2| = |-(x2 - x1)| + |-(y2 - y1)| = |x2 - x1| + |y2 - y1| = d(p2, p1)

3. d(p1, p3) <= d(p1, p2) + d(p2, p3) (triangle inequality)
d(p1, p3) = |x1 - x3| + |y1 - y3| = |(x1 - x2) + (x2 - x3)| + |(y1 - y2) + (y2 - y3)|
By the triangle inequality, d(p1, p3) <= |x1 - x2| + |x2 - x3| + |y1 - y2| + |y2 - y3| = d(p1, p2) + d(p2, p3)


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
def spotlessroomba_closest_plus_farthest_heuristic(state : SpotlessRoombaState)  -> float:
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
This runs Prim's algorithm to find the edge weights of the minimum spanning tree of the graph where
V = dirty tiles + start tile
E = manhattan distance between all pairs of dirty tiles
Edge weights are added to the heuristic as the algorithm runs.

The minimum cost to reach each dirty tile is no smaller than the sum of the edge weights between all nodes.
Therefore, the heuristic is admissible.


Define MST(X) as the sum of the edge weights of the graph with vertices in X and all edges between vertices.
Let s, s' be two possible start nodes. Use the same metric space as in the previous heuristic.

Define (V_s, E_s) and (V_s', E_s') as the aforementioned V and E with s and s' as the respective starting nodes. 
Since we are considering a tree, if we ignore the first cost from the root to a dirty tile, the sum of the edge weights are equal.
In other words, MST(V_s \ s) = MST(V_s' \ s').
Define x_s as the node adjacent to s in the minimum spanning tree of V_s, and x_s' similarly.
x_s is closest to s, and x_s' is closest to s' since these edges are a result of adding an edge from s to an element of V_s (or s' to an element of V_s' respectively).
MST(V_s) = MST(V_s \ s) + d(s, x_s)
MST(V_s') = MST(V_s' \ s') + d(s', x_s')
h(s) = MST(V_s) and h(s') = MST(V_s') as the definition of our heuristic.

h(s) - h(s') = MST(V_s \ s) + d(s, x_s) - (MST(V_s' \ s') + d(s', x_s')) = d(s, x_s) - d(s', x_s')

We want to prove that d(s, x_s) - d(s', x_s') <= d(s, s').
We know that d(s, x_s) <= d(s, x_s'), since x_s must not have been farther to s than x_s'.
d(s, x_s') <= d(s, s') + d(s', x_s') by the triangle inequality
d(s, x_s) <= d(s, x_s') <= d(s, s') + d(s', x_s') by combining the two inequalities
If we now consider the ends of the inequality,
d(s, x_s) - d(s', x_s') <= d(s, s')
d(s, s') is defined as the cost between s and s', c. Therefore,
d(s, x_s) - d(s', x_s') <= c
h(s) - h(s') <= c
QED

Edge case: no dirty tiles
If so, return 0
"""
def spotlessroomba_manhattan_mst(state : SpotlessRoombaState)  -> float:
    if len(state.dirty_locations) == 0:
        return 0
    
    h = 0
    pq = []
    visited = [False] * (len(state.dirty_locations) + 1)
    visited[0] = True

    for end_index, end_tile in enumerate(state.dirty_locations, 1):
        if end_index != 0:
            heapq.heappush(pq, (abs(state.position.row - end_tile.row) + abs(state.position.col - end_tile.col), end_index, end_tile))

    edge_count = 0
    while edge_count < len(state.dirty_locations):
        (dist, end_index, end_tile) = heapq.heappop(pq)
        
        if visited[end_index]:
            continue

        visited[end_index] = True

        for next_index, next_tile in enumerate(state.dirty_locations, 1):
            if not visited[next_index]:
                heapq.heappush(pq, (abs(end_tile.row - next_tile.row) + abs(end_tile.col - next_tile.col), next_index, next_tile))
        
        h += dist
        edge_count += 1

    return h

SPOTLESSROOMBA_HEURISTICS = {"Zero" : zero_heuristic,
                        "Arbitrary": arbitrary_heuristic, 
                        "Closest-Farthest": spotlessroomba_closest_plus_farthest_heuristic,
                        "Minimum Spanning Tree" : spotlessroomba_manhattan_mst
                        }

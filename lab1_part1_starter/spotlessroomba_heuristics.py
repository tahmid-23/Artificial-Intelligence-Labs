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
This is similar to the previous heuristic.
However, to compute the farthest tile, it uses dijkstra's algorithm.

This heuristic does not compute actual distances due to walls or carpets, so it is still a heuristic, if perhaps a slower one.

This is admissible since the roomba must first visit a closest tile, and then it must visit multiple tiles on its path.
This concept is basically the same as the previous heuristic.

The previous counterexample applies here.
"""
def spotlessroomba_dirty_dijkstra(state : SpotlessRoombaState)  -> float:
    if len(state.dirty_locations) == 0:
        return 0

    if len(state.dirty_locations) == 1:
        return abs(state.dirty_locations[0].row - state.position.row) + abs(state.dirty_locations[0].col - state.position.col)
    
    min_tile_index = None
    h = INF

    for i, dirty_tile in enumerate(state.dirty_locations):
        dist = abs(dirty_tile.row - state.position.row) + abs(dirty_tile.col - state.position.col)
        if dist < h:
            min_tile_index = i
            h = dist

    distances = [INF] * len(state.dirty_locations)
    visited = [False] * len(state.dirty_locations)

    distances[min_tile_index] = 0

    for _ in range(len(state.dirty_locations)):
        min_tile_index = None
        min_tile = None
        min_dist = INF

        for i, tile in enumerate(state.dirty_locations):
            dist = distances[i]
            if not visited[i] and dist < min_dist:
                min_tile_index = i
                min_tile = tile
                min_dist = dist

        visited[min_tile_index] = True

        for i, tile in enumerate(state.dirty_locations):
            if not visited[i]:
                new_dist = min_dist + abs(min_tile.col - tile.col) + abs(min_tile.row - tile.row)
                if new_dist < distances[i]:
                    distances[i] = new_dist

    max_dist = -INF
    for dist in distances:
        max_dist = max(max_dist, dist)

    h += max_dist
    return h

SPOTLESSROOMBA_HEURISTICS = {"Zero" : zero_heuristic,
                        "Arbitrary": arbitrary_heuristic, 
                        "Closest-Farthest": spotlessroomba_closest_plus_farthest_heuristic,
                        "Closest-Dijkstra" : spotlessroomba_dirty_dijkstra
                        }

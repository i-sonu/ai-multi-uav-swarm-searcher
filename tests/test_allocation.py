"""Unit tests for Multi-Agent Task Allocation (MATA)."""

import numpy as np
from src.agents.agent import Agent
from src.mapping.occupancy_grid import OccupancyGrid
from src.planning.allocation import allocate, build_cost_matrix, build_utility_matrix
from src.planning.astar import astar


def test_allocation_shapes_and_validity():
    grid = OccupancyGrid(height=20, width=20, resolution=0.25)
    # Set a block of unknown space
    grid.grid[5:15, 5:15] = -1

    agent1 = Agent(agent_id=0, pose=(0.0, 0.0, 0.0))
    agent2 = Agent(agent_id=1, pose=(1.0, 1.0, 0.0))
    agents = [agent1, agent2]

    frontiers = [(5, 5), (10, 10), (14, 14)]

    cost_matrix = build_cost_matrix(agents, frontiers, grid, planner_fn=astar)
    assert cost_matrix.shape == (2, 3)

    utility_vector = build_utility_matrix(grid, frontiers, sensor_range=3.0)
    assert utility_vector.shape == (3,)

    # Test baseline (none)
    assign_none = allocate(cost_matrix, utility_vector, method="none")
    assert len(assign_none) == 2

    # Test greedy allocation
    assign_greedy = allocate(cost_matrix, utility_vector, method="greedy")
    assert len(assign_greedy) == 2
    assert assign_greedy[0] != assign_greedy[1]  # distinct assignment

    # Test Hungarian allocation
    assign_hungarian = allocate(cost_matrix, utility_vector, method="hungarian")
    assert len(assign_hungarian) == 2
    assert assign_hungarian[0] != assign_hungarian[1]  # distinct assignment

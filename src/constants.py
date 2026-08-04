"""Project-wide constants.

Occupancy encoding is duplicated in ``configs/default.yaml`` for the record, but
these module-level integers are the single source of truth used by the hot
inner loops (sensor ray casting, frontier detection) where importing from a dict
every call would be wasteful. The two must always agree.
"""

# --- Occupancy grid cell encoding (CLAUDE.md §3, use these exact values) ---
UNKNOWN = -1
FREE = 0
OCCUPIED = 1

# --- Ground-truth map encoding (map_generator output) ---
# The ground-truth map is binary: a cell is either traversable or a wall.
GT_FREE = 0
GT_WALL = 1

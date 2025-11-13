import math

""" Geometry utilities for football pitch calculations """
# Pitch in meters
PITCH_X, PITCH_Y = 120.0, 80.0
GOAL_X, GOAL_Y = 120.0, 40.0  # center of goal on x-max line


def distance_to_goal(x, y):
    """Calculate distance to center of goal in meters."""
    dx, dy = GOAL_X - float(x), GOAL_Y - float(y)
    return math.hypot(dx, dy)


def shot_angle(x, y, goal_width=7.32):
    """Caculate the shot angle in radians."""
    y_upper = GOAL_Y + goal_width / 2.0
    y_lower = GOAL_Y - goal_width / 2.0
    a = math.atan2(y_upper - y, GOAL_X - x)
    b = math.atan2(y_lower - y, GOAL_X - x)
    return abs(a - b)

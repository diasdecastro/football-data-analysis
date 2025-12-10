import numpy as np

""" Geometry utilities for football pitch calculations """

PITCH_X, PITCH_Y = 120.0, 80.0  # Pitch in meters
GOAL_X, GOAL_Y = 120.0, 40.0  # center of goal on x-max line


def distance_to_goal(x, y):
    """Calculate distance to center of goal in meters.
    Supports both scalar and array inputs (vectorized).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Check bounds - returns array of booleans for vectorized input
    out_of_bounds = (x < 0) | (x > PITCH_X) | (y < 0) | (y > PITCH_Y)
    if np.any(out_of_bounds):
        raise ValueError("Coordinates out of pitch bounds")

    dx = GOAL_X - x
    dy = GOAL_Y - y
    result = np.hypot(dx, dy)  # Hypotenuse (euclidean distance)

    # Return scalar if input was scalar
    return float(result) if result.ndim == 0 else result


def shot_angle(x, y, goal_width=7.32):
    """Calculate the shot angle in radians.
    Supports both scalar and array inputs (vectorized).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Check bounds - returns array of booleans for vectorized input
    out_of_bounds = (x < 0) | (x > PITCH_X) | (y < 0) | (y > PITCH_Y)
    if np.any(out_of_bounds):
        raise ValueError("Coordinates out of pitch bounds")

    y_upper = GOAL_Y + goal_width / 2.0
    y_lower = GOAL_Y - goal_width / 2.0
    a = np.arctan2(y_upper - y, GOAL_X - x)
    b = np.arctan2(y_lower - y, GOAL_X - x)
    result = np.abs(a - b)

    # Return scalar if input was scalar
    return float(result) if result.ndim == 0 else result

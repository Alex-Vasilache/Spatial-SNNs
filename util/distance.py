import numpy as np


def manhattan(pos1, pos2):
    """
    Calculates the Manhattan distance between two points.

    Args:
        pos1 (np.ndarray): The coordinates of the first point.
        pos2 (np.ndarray): The coordinates of the second point.

    Returns:
        int: The Manhattan distance between the two points.
    """
    return int(sum([np.abs(pos1[i] - pos2[i]) for i in range(len(pos1))]))


def euclidian(pos1, pos2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        pos1 (np.ndarray): The coordinates of the first point.
        pos2 (np.ndarray): The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(pos1 - pos2)
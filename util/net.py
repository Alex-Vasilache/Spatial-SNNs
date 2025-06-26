import numpy as np


NON_SPATIAL_PROB = 0.1


def sp_2(dist, spatial):
    """
    Calculates the connection probability based on distance, using an exponential decay function.

    Args:
        dist (np.ndarray): The distance matrix.
        spatial (bool): A flag indicating whether to use spatial properties.

    Returns:
        np.ndarray: The connection probability matrix.
    """
    dist[dist == 0] = 1
    if spatial:
        return np.power(np.exp(-dist), 2)
    else:
        return np.ones_like(dist) * NON_SPATIAL_PROB


def sp_3(dist, spatial):
    """
    Calculates the connection strength based on distance, using an exponential decay function.

    Args:
        dist (np.ndarray): The distance matrix.
        spatial (bool): A flag indicating whether to use spatial properties.

    Returns:
        np.ndarray: The connection strength matrix.
    """
    sp = np.exp(-dist + 1)
    if spatial:
        return np.power(sp, 2)
    else:
        return np.ones_like(sp)


def get_id_from_pos(pos, size):
    """
    Converts a position in a multi-dimensional grid to a unique ID.

    Args:
        pos (np.ndarray): The position coordinates.
        size (np.ndarray): The dimensions of the grid.

    Returns:
        int: The unique ID corresponding to the position.
    """
    ID = 0

    for d in range(len(size)):
        ID = ID * size[d] + pos[d]

    return ID


def get_pos_from_id(ID, size):
    """
    Converts a unique ID to a position in a multi-dimensional grid.

    Args:
        ID (int): The unique ID.
        size (np.ndarray): The dimensions of the grid.

    Returns:
        np.ndarray: The position coordinates corresponding to the ID.
    """
    pos = []

    for d in range(len(size) - 1, -1, -1):
        pos.append(int(ID % size[d]))
        ID = (ID - ID % size[d]) / size[d]
    pos.reverse()

    return np.array(pos)

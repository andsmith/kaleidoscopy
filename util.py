import numpy as np


def make_bounds(coords):
    max_vals = np.max(coords, axis=0)
    min_vals = np.min(coords, axis=0)
    return {'top': max_vals[1],
            'bottom': min_vals[1],
            'left': min_vals[0],
            'right': max_vals[0]}

import numpy as np


def center_and_scale_points(points, scale, center):
    points_centered = points - center
    points_scale = np.max(points, axis=0) - np.min(points, axis=0)
    rescale = np.max(points_scale)
    rescale_factor = scale / rescale
    points_centered *= rescale_factor
    return points
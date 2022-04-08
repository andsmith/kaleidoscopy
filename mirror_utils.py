import numpy as np


def transform_points(unit_points, scale, center=None):
    """
    :param unit_points:  N x 2 points, within unit square
    :param scale: relative size change
    :param center:  final center of points
    """
    center = center if center is not None else (np.max(unit_points, axis=0) + np.min(unit_points, axis=0)) / 2.0

    points_centered = unit_points - center
    points_centered *= scale
    return points_centered + center

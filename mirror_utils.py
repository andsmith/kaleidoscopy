import numpy as np


def unitize_shape(corner_verts):
    """
    Transform shape to be inscribed in unit square.
    :param corner_verts:  N x 2 array of 2d points
    :returns:  Shape transformed to largest that will fit in [0,1]x[0,1]
    """
    spans = np.max(corner_verts, axis=0) - np.min(corner_verts, axis=0)
    scale = np.max(spans)

    if scale == 0:
        raise Exception("Not enough non-coinciding points.")

    corner_verts /= scale
    return corner_verts - np.min(corner_verts,axis=0).reshape(1,2)


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

from abc import ABC, abstractmethod
import numpy as np


class Surface(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_intersections_and_normals(self, origins, unit_directions):
        pass


class Plane(Surface):
    def __init__(self, xyz_intersect, normal):
        self._xyz = np.array(xyz_intersect).reshape(-1)
        self._normal = (normal / np.linalg.norm(normal)).reshape(-1)

    @staticmethod
    def z_zero_plane():
        return Plane(np.array((0., 0., 0.)),
                     np.array((0., 0., 1.)))

    def get_intersections_and_normals(self, origins, unit_directions, no_points=False):
        """
        Find intersection of rays with given plane.
        :param origins:  Nx 3 array of ray origin points
        :param unit_directions:  N x 3 array of ray direction vectors
        :param no_points:  just calculate distance, not intersection points
        :return: N distances for N active rays,
                 N x 3 intersection points for N active rays, or None if no_points is True.
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            # parallel rays go to np.inf, pointing away are negative
            dists = np.dot(self._xyz - origins, self._normal) / np.dot(unit_directions, self._normal)
        dists = dists.reshape(-1, 1)
        points = None
        if not no_points:
            points = origins + dists * unit_directions
        return dists, points


class Cylinder(Surface):
    def __init__(self, xy_center, xy_rad):
        self._xy_center = xy_center
        self._xy_radius = xy_rad
        self._axis = np.array([0.0, 0.0, 1.0])

    def get_intersections_and_normals(self, origins, unit_directions):
        raise NotImplementedError("Cylindrical mirrors not implemented yet.")

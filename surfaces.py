"""
Classes to handle geometry of light bouncing off mirrors.
"""
from abc import ABC, abstractmethod
import numpy as np


class Surface(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_ray_intersections(self, origins, unit_directions):
        """
        Find intersection points of rays with surface.
        Note: no check is done to see if the intersection is in the positive direction (dists>0).

        :param origins:  N x 3 array of ray origin points
        :param unit_directions:  N x 3 array of ray direction vectors
        :return:  N x 3 intersection points for N active rays,
                  N array of distances to points (possibly negative)
        """
        pass

    @abstractmethod
    def get_ray_reflections(self, origins, unit_directions):
        """
        Find direction of rays reflected off surface.

        :param origins: N x 2 array of ray origin points
        :param unit_directions:  N x 3 array of ray directions
        :returns:  N x 3 array of new ray directions.  (new origins are intersection points)
        """


class Plane(Surface):
    def __init__(self, xyz_intersect, normal):
        self._xyz = np.array(xyz_intersect).reshape(-1)
        self._normal = (normal / np.linalg.norm(normal)).reshape(-1)

    def get_params(self):
        """The paramterization"""
        return self._xyz, self._normal

    @staticmethod
    def make_z_zero_plane():
        """
        Special Plane at z=0 (target plane of rays)
        """
        return Plane(np.array((0., 0., 0.)),
                     np.array((0., 0., 1.)))

    def get_ray_intersections(self, origins, unit_directions):
        """
        Intersect given rays with self.
        :param origins:  N x 3 (xyz) coords of ray origin points
        :param unit_directions:  Nx3 (x,y,z), direction of each ray
        :returns:  Nx3 intersection points, and N-array of intersection distances (possibly negative).

        """
        with np.errstate(divide='ignore', invalid='ignore'):
            # parallel rays go to np.inf, pointing away are negative
            dists = np.dot(self._xyz - origins, self._normal) / np.dot(unit_directions, self._normal)
        dists = dists.reshape(-1,1)
        points = origins + unit_directions * np.tile(dists, (1, 3))
        return points, dists

    def get_ray_reflections(self, origins, unit_directions):
        # new direction has component parallel to normal reversed
        delta = - 2.0 * np.dot(unit_directions, self._normal).reshape(-1, 1) \
                * self._normal.reshape(1, 3)
        new_directions = unit_directions + delta

        return new_directions


class Cylinder(Surface):
    def __init__(self, xy_center, xy_rad):
        self._xy_center = xy_center
        self._xy_radius = xy_rad
        self._axis = np.array([0.0, 0.0, 1.0])

    def get_ray_reflections(self, origins, unit_directions):
        raise NotImplementedError("Cylindrical mirrors not implemented yet.")

    def get_ray_intersections(self, origins, unit_directions):
        raise NotImplementedError("Cylindrical mirrors not implemented yet.")

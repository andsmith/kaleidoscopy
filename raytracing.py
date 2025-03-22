"""
Create the kaleidoscope's image map by raytracing backwards from the eye, through
the image plane, bouncing off the mirrors and hitting the target.

The image map can then be applied live to video to simulate the kaleidoscope effect.

Coordinate conventions:
    - eye is at (0, 0, 0), and it looks in the positive z direction
    - image plane is at z = EYE_DIST
    - target plane is at z = TARGET_DIST (TARGET_DIST > EYE_DIST)
    - mirrors are vertical, and are touching the target plane (no ray will go under the mirror to hit the target)
    - mirrors are defined by two points, p0 and p1 in the XY plane, extend indfinitely in both z directions
    - The field of view is set to

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


EYE_DIST = 0.06  # should be close enough that all points are inside mirror tube.
TARGET_DIST = 0.36


class Mirror(object):
    """
    planar, vertical mirror
    """

    def __init__(self, p0, p1):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)


class RaySet(object):
    """
    A RaySet is a group of rays that has not been broken up by bouncing off different mirrors.
    Before hitting any mirrors, there is one RaySet.
    Whenever a RaySet hits more than one mirror, it is split into multiple RaySets.
    """

    def __init__(self, origins, directions, x_ind, y_ind):
        """
        """
        self.origins = origins
        self.directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        self.x_inds = x_ind
        self.y_inds = y_ind

    @staticmethod
    def from_eyball(w, h, w_px, h_px):
        """
        Create a RaySet from the eye to the image plane.
        :param w: width of the image plane (extent in x)
        :param h: height of the image plane (extent in y)
        :param w_px: number of pixels in the width
        :param h_px: number of pixels in the height
        """
        x, y = np.meshgrid(np.linspace(-w/2, w/2, w_px), np.linspace(-h/2, h/2, h_px))
        x_inds, y_inds = np.meshgrid(np.arange(w_px), np.arange(h_px))
        z = np.ones_like(x) * EYE_DIST
        origins = np.stack([x, y, z], axis=-1)
        directions = np.stack([np.zeros_like(x), np.zeros_like(y), np.ones_like(z)], axis=-1)
        # calculate FOV in x and y:
        fov_x_deg = np.degrees(np.arctan(w / (2 * EYE_DIST))) * 2
        fov_y_deg = np.degrees(np.arctan(h / (2 * EYE_DIST))) * 2
        print("Field of view %.2f x %.2f degrees in X and y." % (fov_x_deg, fov_y_deg))
        return RaySet(origins, directions, x_inds, y_inds)

    def get_target_intersections(self):
        """
        Get the intersection points of the rays with the target plane at z=TARGET_DIST.
        """
        t = (TARGET_DIST - self.origins[..., 2]) / self.directions[..., 2]
        return self.origins + t[..., np.newaxis] * self.directions
    
    def get_mirror_intersections(self, mirror):
        """
        Get the intersection points of the rays with the mirror.
        """
        p0 = mirror.p0
        p1 = mirror.p1
        v = p1 - p0
        v = v / np.linalg.norm(v)
        n = np.array([-v[1], v[0], 0])
        n = n / np.linalg.norm(n)
        p = self.origins
        d = self.directions
        t = np.dot(p - p0, n) / np.dot(d, n)
        return p + t[..., np.newaxis] * d

class Scope(object):

    def __init__(self, mirrors, size=(640, 480)):
        self._mirrors = mirrors
        self._size = size
        self._check()

    def _check(self):
        """
        Make sure mirrors are connected?
        """
        pass

    def _get_image_plane_extent(self, w_px, h_px):
        """
        Get the extent of the image plane in x and y.
        """
        w = self._size[0] / self._size[1] * EYE_DIST
        h = EYE_DIST
        return w, h

    def trace_map(self, w_px, h_px):
        """
        Trace the image map through the mirrors to the target plane:
            1. For each ray, see which mirror (or the target) it hits first.
            2. Separate the rays into groups depending on what they hit, 
                2.a. if they hit a mirror, keep bouncing them.
                2.b. if they hit the target, assign the target xy locations as the map values for those rays.
        """
        map_x, map_y = np.zeros((h_px, w_px)), np.zeros((h_px, w_px))

        w, h = self._get_image_plane_extent(w_px, h_px)
        ray_list = [RaySet.from_eyball(w, h, w_px, h_px)]
        while len(ray_list) > 0:
            for rays in ray_list:
                closest_indices, distances = self._get_hits(rays)
                surf_inds = list(set(closest_indices))
                for surf_ind in surf_inds:
                    if surf_ind == len(self._mirrors):
                        # hit the target
                        pass

def test_ray_tracer():
    mirrors = make_iso_mirrors(30)
    scope = Scope(mirrors)
    scope.trace_map(5,5)

def make_iso_mirrors(angle_deg=30., size = 0.9):
    """
    Make a set of mirrors in an isosceles triangle pointing up, centered around the origin.
    :param angle_deg: angle of the unique angle in the triangle
    :param size: size of the triangle (maximum distance from the center to a corner)
    """
    p0 = np.array([-1, 0])
    p1 = np.array([1, 0])
    p2 = np.array([0, 1/np.tan(np.radians(angle_deg/2))])

    points = np.array([p0, p1, p2])
    points = points - np.mean(points, axis=0)  # center around the origin
    r = np.max(np.linalg.norm(points, axis=1))  # maximum distance from the origin
    points = points / r * size  # scale to the desired size
    mirrors = [Mirror(points[i], points[(i+1) % 3]) for i in range(3)]
    return mirrors


def test_iso_mirrors():
    # plot the mirrors for a 15, 30, 45, 60, 90, and 120 degree triangle
    angles = [15, 30, 45, 60, 90, 120]
    n_plot = 1
    for angle in angles:
        mirrors = make_iso_mirrors(angle)
        plt.subplot(2, 3, n_plot)   
        n_plot += 1
        for mirror in mirrors:
            plt.plot([mirror.p0[0], mirror.p1[0]], [mirror.p0[1], mirror.p1[1]],'ko-')
        plt.title("%d degrees" % angle)
        plt.axis('equal')
        plt.axis('off')
    plt.tight_layout()
    plt.show()  

if __name__ == "__main__":
    test_iso_mirrors()
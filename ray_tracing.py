import numpy as np
import cv2
import logging
import time
import matplotlib.pylab as plt
from util import make_bounds


class MirrorTube(object):
    """
    Define a ortho-prismatic_tube (shape) tube of mirrors, i.e. all perpendicular to flat, facing indwards.
    Input is an arbitrary list of 2-d polygon vertices.  (assumed clockwise, mirrors facing inwards)
    """

    def __init__(self, corners, plot_init=False):
        """
        Define mirror prism.
        :param corners:  list of 2-d coordinates (numpy arrays), i.e. mirror closes loop of polygon vertices.
        :param plot_init:  Show mirror set-up
        """
        n = len(corners)
        corner_height = 0.0  # Z-coord
        # mirror centers are midpoints between corners, and in 3d
        centers = [(corners[i] + corners[i + 1]) / 2.0 for i in range(n - 1)]
        centers.append((corners[0] + corners[-1]) / 2.0)
        centers = [np.hstack((c, [0.])) for c in centers]

        # Need two non-parallel, co-planar vectors whose cross-product will give us the normal for each mirror
        # First will connect each mirror's centers to a corner.
        co_planar_a = [centers[i] - np.hstack((corners[i], [0.0])) for i in range(n)]
        # Second will connect each mirror's first corner to a point 1cm above that corner.
        co_planar_b = [centers[i] - np.hstack((corners[i], [1.0])) for i in range(n)]
        normals = np.array([np.cross(co_planar_a[i], co_planar_b[i]) for i in range(n)])
        normals /= - np.linalg.norm(normals, axis=1).reshape(-1,1)  # negate for clockwise orient.

        corner_array = [np.hstack((corners[i].reshape(-1), [corner_height],
                                   corners[i + 1].reshape(-1), [corner_height])) for i in range(n - 1)]
        corner_array.append(np.hstack((corners[-1].reshape(-1), [corner_height],
                                       corners[0].reshape(-1), [corner_height])))

        self._bounds = make_bounds(corners)
        self._n = n
        self._centers = np.array(centers)
        self._corners = np.array(corner_array)
        self._normals = normals
        logging.info("Created %i-mirror tube." % (n,))

        if plot_init:
            self._plot_mirrors()

    def _plot_mirrors(self):
        arrow_lengths = np.linalg.norm(self._corners[:,:3] - self._corners[:,3:],axis=1)
        arrow_length = np.min(arrow_lengths)
        for i in range(self._n):
            plt.plot([self._corners[i, 0], self._corners[i, 3]],
                     [self._corners[i, 1], self._corners[i, 4]], 'bo-', linewidth=3,markersize=10)
            plt.plot(self._centers[i, 0],self._centers[i, 1], 'go', markersize=8)

            arrow_end = self._centers[i] + arrow_length * self._normals[i,:]

            plt.plot([self._centers[i, 0],  arrow_end[0]],
                     [self._centers[i, 1],  arrow_end[1]], 'k-', linewidth=1)

        plt.axis('equal')
        plt.show()

    def get_bounds(self):
        return self._bounds

    def trace(self, ray_starts, ray_dirs, back_cm, max_recurse=10):
        """
        Bounce rays through mirror tube, see where they land on other side.

        :param ray_starts:  H x W x 3 array of ray origins (i.e. eye), axis 2 is x,y,z planes
            If 1 x 3, then broadcast for all.
        :param ray_dirs:  H x W x 3 array of ray unit directions (i.e. pixel locations), axis 2 is x,y,z planes
        :param back_cm:  Z-coordinate of back, i.e. where rays "hit"
        :param max_recurse:  how many reflections before ray is considered not converging
        :return:  H x W x 3 array of ray destinations on the back plane, or NAN if divergent
        """
        out_points = ray_dirs * 0
        active = np.ones(ray_dirs.shape[0], dtype=np.uint8)

        if ray_starts.size < ray_dirs.size:
            ray_starts = np.tile(ray_starts.reshape(1, 1, 3), (ray_dirs.shape[0], ray_dirs.shape[1], 1))

        back_center = np.array([0.0, 0.0, back_cm]).reshape(1, -1)
        back_normal = np.array([0.0, 0.0, -1.0])  # towards eye

        for iteration in range(max_recurse):
            print("Ray-tracing iteration %i, %i active rays." % (iteration, active.sum()))
            if np.sum(active) == 0:
                break

            mirror_distances = [_calc_ray_plane_intersect_dists(ray_starts[active],
                                                                ray_dirs[active],
                                                                self._centers[i, :],
                                                                self._normals[i, :]) for i in range(self._n)]

            back_distances = _calc_ray_plane_intersect_dists(ray_starts[active],
                                                             ray_dirs[active],
                                                             back_center,
                                                             back_normal)
            distances = np.hstack([intersect for intersect in mirror_distances] + [back_distances])

            # not hitting = going parallel or backwards
            diverging = distances <= 0
            distances[diverging] = np.inf

            # the "hit" is the closest thing with positive distance
            hits = np.argmin(distances, axis=1)

            # hit the background
            background_hits = hits == self._n
            background_intersects = back_distances[background_hits] * ray_dirs[active][background_hits] + \
                                    ray_starts[active][background_hits]
            out_points[active][background_hits] = background_intersects
            active[active][background_hits] = 0

            # or hit a mirror.
            for mirror_i in range(self._n):
                mirror_hits = hits == mirror_i
                mirror_intersects = mirror_distances[mirror_hits] * ray_dirs[active][mirror_hits] + \
                                    ray_starts[active][mirror_hits]

                mirror_reflects = 2 * self._normals[mirror_i] * np.dot(ray_dirs, self._normals[mirror_i]) - ray_dirs
                new_ray_starts = mirror_intersects
                new_ray_dirs = mirror_reflects / np.linalg.norm(mirror_reflects, axis=2)
                ray_starts[active][mirror_hits] = new_ray_starts
                ray_dirs[active][mirror_hits] = new_ray_dirs
        return out_points


def _calc_ray_plane_intersect_dists(ray_starts, ray_dirs, plane_center, plane_normal):
    dists = np.dot(plane_center - ray_starts, plane_normal) / np.dot(ray_dirs, plane_normal)
    return dists


class IsoscelesMirrorTube(MirrorTube):
    def __init__(self, theta_deg, h_cm, **kwargs):
        theta = np.deg2rad(theta_deg)
        corners = [np.array([-np.sin(theta), 0]),
                   np.array([0, h_cm]),
                   np.array([np.sin(theta), 0]), ]

        super(IsoscelesMirrorTube, self).__init__(corners=corners, **kwargs)


def make_rays(x_span, y_span, z, res, unit=True):
    # Create image plane pixel coordinates
    px = np.linspace(x_span[0], x_span[1], res[1])
    py = np.linspace(y_span[0], y_span[0], res[0])
    x, y = np.meshgrid(px, py)
    img_coords = np.dstack((y, x, np.ones(x.shape) * z))

    # Create rays
    eye = np.zeros(3).reshape(1, 3)
    rays = img_coords - eye
    if unit:
        rays = (rays.T / np.linalg.norm(rays, axis=2).T).T  # make unit necessary ???
    return rays


def test_ray_tracing():
    mirrors = IsoscelesMirrorTube(theta_deg=10.0, h_cm=2.54, plot_init=True)
    rays = make_rays((-0.5, 0.5), (-0.5, 0.5), 1.0, (100, 100))
    eye = np.zeros(3).reshape(1, 3)

    test = mirrors.trace(eye, rays, 25.0)


if __name__ == "__main__":
    test_ray_tracing()

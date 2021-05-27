import numpy as np
import cv2
import logging
import time
import matplotlib.pylab as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from util import make_bounds


class RayBundle(object):
    """
    Represents a grid of rays, update as they reflect, etc.
    Preserve array shape using mask.
    """

    def __init__(self, ray_origins, ray_directions, init_active=True):
        """
        initialize with explicit rays
        :param ray_origins:  H x W x 3 array, x, y, z coords of ray origin points
        :param ray_directions:  H x W x 3 array, vectors indicating ray directions
        :param init_active: initialize mask to this value
        """
        self._active = np.full(ray_origins.shape[:2], init_active)

        if len(ray_origins.shape) == 2:
            ray_origins = ray_origins.reshape(-1, 1, 3)
        if len(ray_directions.shape) == 2:
            ray_directions = ray_directions.reshape(-1, 1, 3)
        self._origins = ray_origins
        self._directions = ray_directions
        self._directions /= np.linalg.norm(self._directions, axis=2, keepdims=True)  # unit-ize
        self._distances = 0 * self._origins[:, :, 0]

    @staticmethod
    def from_origin_to_plane(x_span, y_span, z, res, **kwargs):
        """
        initialize a grid from 2-d extent and distance from origin

        :param x_span:  [left, right] extent of 2-d coordinate grid
        :param y_span:  [bottom and top] extent
        :param z:  distance from origin (cm)
        :param res:  number of rays along horizontal and vertical directions
        :return: RayBundle object
        """
        px = np.linspace(x_span[0], x_span[1], res[1])
        py = np.linspace(y_span[0], y_span[1], res[0])
        x, y = np.meshgrid(px, py)
        img_coords = np.dstack((y, x, np.ones(x.shape) * z))

        # Create rays
        eye = np.zeros(3).reshape(1, 3)
        rays = img_coords - eye
        origins = 0 * rays
        return RayBundle(origins, rays, **kwargs)

    def get_active_rays(self):
        return self._origins[self._active], self._directions[self._active]

    def get_active_count(self):
        return np.sum(self._active)

    def get_mask(self):
        return self._active

    def plot_3d(self, distances=0.1, mask=None, color=(1.0, 0.5, 0.5, 0.9), ax=None):
        """
        Plot lines from ray origins to distances.
        :param distances:   length of rays, should be scalar or same shape os self._origins.shape[:2]
        :param mask: only plot active rays & this mask
        :param color: plot this color
        :param ax:  plot axis object
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        n = np.sum(mask) if mask is not None else np.sum(self._active)

        ultra_mask = self._active
        if mask is not None:
            ultra_mask = _double_index(self._active, mask)

        x = np.zeros(n * 3, dtype=np.float64)
        y = 0 * x
        z = 0 * x

        if isinstance(distances, np.ndarray):
            distances = distances.reshape(-1, 1)

        destinations = self._origins[ultra_mask] + self._directions[ultra_mask] * distances

        x[0::3] = self._origins[ultra_mask][:, 0]
        x[1::3] = destinations[:, 0]
        x[2::3] = np.nan
        y[0::3] = self._origins[ultra_mask][:, 1]
        y[1::3] = destinations[:, 1]
        y[2::3] = np.nan
        z[0::3] = self._origins[ultra_mask][:, 2]
        z[1::3] = destinations[:, 2]
        z[2::3] = np.nan

        ax.scatter(x[0::3], y[0::3], z[::3], color=color)
        ax.plot(x, y, z, color=color)

        return ax

    def intersect_active_plane_dist(self, point, normal):
        """
        Return the distance to the intersection of the current bundle with the plane
        :param point:  a point on the plane
        :param normal:  normal vector to plane
        :param mask:
        :return: H x W distances, or if mask is used, 1-dimensional array
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            # parallel rays go to np.inf
            dists = np.dot(point - self._origins[self._active], normal) / np.dot(self._directions[self._active], normal)

        if len(dists.shape) == 3:
            dists = dists.reshape(-1, 3)
        return dists

    def deactivate(self, inactive):
        self._active[_double_index(self._active, inactive)] = False

    def reflect(self, subset, intersections, plane_normal):
        self._origins[_double_index(self._active, subset)] = intersections

        new_directions = self._directions[self._active][subset] - 2.0 * np.dot(self._directions[self._active][subset],
                                                                               plane_normal).reshape(-1,
                                                                                                     1) * plane_normal.reshape(
            1, 3)

        self._directions[_double_index(self._active, subset)] = new_directions

    def get_shape(self):
        return self._directions.shape[:2]


class MirrorTube(object):
    """
    Define a ortho-prismatic_tube (shape) tube of mirrors, i.e. all perpendicular to flat, facing indwards.
    Input is an arbitrary list of 2-d polygon vertices.  (assumed clockwise, mirrors facing inwards)
    """

    def __init__(self, corners, **kwargs):
        """
        Define vertices of the mirror prism.

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
        normals /= - np.linalg.norm(normals, axis=1).reshape(-1, 1)  # negate for clockwise orient.
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

    def plot_mirrors(self):
        side_lengths = np.linalg.norm(self._corners[:, :3] - self._corners[:, 3:], axis=1)
        for i in range(self._n):
            plt.plot([self._corners[i, 0], self._corners[i, 3]],
                     [self._corners[i, 1], self._corners[i, 4]], 'bo-', linewidth=3, markersize=10)
            plt.plot(self._centers[i, 0], self._centers[i, 1], 'go', markersize=8)
            plt.annotate("Mirror %i" % (i,), self._centers[i, :2] + side_lengths[i] / 40)
        plt.quiver(self._centers[:, 0], self._centers[:, 1],
                   self._normals[:, 0], self._normals[:, 1], headwidth=2)
        plt.axis('equal')

    def plot_3d(self, base, height, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self._corners[:, 0], self._corners[:, 1], height, color=[1.0, 0, 0])
        ax.scatter(self._corners[:, 3], self._corners[:, 4], base, color=[1.0, 0, 0])
        ax.axis('equal')
        return ax

    def get_bounds(self):
        return self._bounds

    def trace(self, rays, back_cm, max_reflect=10, plot=False):
        """
        Bounce rays through mirror tube, see where they land on other side.

        :param rays:  a RayBundle object
        :param back_cm:  Z-coordinate of back, i.e. where rays "hit"
        :param max_reflect:  how many reflections before ray is considered not converging
        :return:  H x W x 3 array of ray destinations on the back plane, or NAN if divergent
        """

        back_center = np.array([0.0, 0.0, back_cm]).reshape(1, -1)
        back_normal = np.array([0.0, 0.0, -1.0])  # towards eye

        # Where each ray ends up on the background (i.e. plane back_cm from origin, in positive Z direction)
        out_points = np.zeros(shape=list(rays.get_shape()) + [3])

        # remember these so as to not hit same thing twice (-1 means invalid)
        last_hits = np.zeros(np.prod(rays.get_shape()), dtype=np.int64) - 1

        for iteration in range(max_reflect):
            ray_mask = rays.get_mask()
            n_active = ray_mask.sum()

            logging.info("Ray tracing iteration %i / %i - %i active rays." % (iteration + 1, max_reflect, n_active))
            if np.sum(n_active) == 0:
                logging.info('\tNo more active rays, trace complete.')
                break

            # distance to each mirror for each ray
            mirror_distances = [rays.intersect_active_plane_dist(self._centers[i, :], self._normals[i, :]) for i in
                                range(self._n)]
            # distance to background
            back_distances = rays.intersect_active_plane_dist(back_center, back_normal)

            distances = np.vstack(mirror_distances + [back_distances]).T

            # If the ray just hit a surface, it can't hit it again (all planes), so enforce this explicitly:

            # "turning off" = set one column of each row of distances to INF
            rows_with_col_to_turn_off = np.logical_and(ray_mask.reshape(-1), last_hits >= 0)

            FIX!!!
            distances[(last_hits>=0)[ray_mask], last_hits[last_hits] = np.inf
            logging.info("Turned off %i distances of %i surfaces." % (to_turn_off.sum(), self._n + 1))

            # not hitting = going backwards or parallel
            with np.errstate(divide='ignore', invalid='ignore'):
                diverging = np.logical_or(distances < 0, np.isinf(distances))

            # if #ALL diverging,
            all_bad = np.bitwise_and.reduce(np.isinf(distances), axis=1)
            if np.sum(all_bad) > 0:
                logging.warning("\t%s pixels hit nothing..." % (np.sum(all_bad),))
                all_bad_idx = np.where(all_bad)[0][0]
                print(rays._origins.reshape(-1, 3)[all_bad_idx, :])
                print(rays._directions.reshape(-1, 3)[all_bad_idx, :])

            # these (ray, surface) pairs cannot be the closest intersection
            distances[diverging] = np.inf
            logging.info(
                "\tIteration has %s rays diverging from %i mirrors + background." % (np.sum(diverging), self._n))

            # the "hit" is the closest thing with positive distance  (everything not hitting now is Inf away)
            hits = np.argmin(distances, axis=1)

            # these hit nothing, what to do with them?
            hits[all_bad] = -1

            ray_starts, ray_dirs = rays.get_active_rays()

            # hit the background?
            background_hits = hits == self._n
            logging.info(
                "\tBackground has %s rays hitting it first." % (np.sum(background_hits),))
            background_intersects = back_distances[background_hits].reshape(-1, 1) * ray_dirs[background_hits, :] + \
                                    ray_starts[background_hits, :]
            idx = _double_index(ray_mask, background_hits)
            out_points[idx] = background_intersects

            last_hits[idx.reshape(-1)] = self._n

            # or hit a mirror.
            ax = None
            if plot:
                # ax = rays.plot_3d(back_cm, mask=None, color=[0, 0, 0, 0.8])
                ax = self.plot_3d(back_cm, 1.0)

            # And finally, do the reflections.
            palette = cm.get_cmap('brg')
            color_indices = np.linspace(0, 1.0, self._n + 1)
            print(color_indices)
            colors = [palette(i) for i in color_indices]
            for mirror_i in range(self._n):
                mirror_hits = hits == mirror_i
                idx = _double_index(ray_mask, mirror_hits)
                last_hits[idx.reshape(-1)] = mirror_i
                logging.info(
                    "\tMirror %i has %s rays hitting it first" % (mirror_i, np.sum(mirror_hits),))
                with np.errstate(divide='ignore', invalid='ignore'):
                    mirror_intersects = mirror_distances[mirror_i][mirror_hits].reshape(-1, 1) * ray_dirs[mirror_hits] + \
                                        ray_starts[mirror_hits]

                dists = mirror_distances[mirror_i][mirror_hits]
                if mirror_i == 1:
                    rays.plot_3d(dists, mirror_hits, color=colors[mirror_i], ax=ax)

                rays.reflect(mirror_hits, mirror_intersects, self._normals[mirror_i])

            to_deactivate = np.logical_or(background_hits, all_bad)
            rays.deactivate(to_deactivate)

            if plot:
                plt.show()
                plt.imshow(last_hits.reshape(rays.get_shape()))
                plt.axis('equal')
                plt.colorbar()
                plt.show()
            ax = None
            if plot:
                # ax = rays.plot_3d(back_cm, mask=None, color=[0, 0, 0, 0.8])
                ax = self.plot_3d(back_cm, 1.0)

        return out_points


def _double_index(mask, sub_mask):
    """
    Probably could be more efficient
    """
    r = mask.copy()
    r[mask] = sub_mask
    return r


class IsoscelesMirrorTube(MirrorTube):
    def __init__(self, theta_deg, h_cm, **kwargs):
        theta = np.deg2rad(theta_deg)
        corners = [np.array([-np.sin(theta), 0]),
                   np.array([0, h_cm]),
                   np.array([np.sin(theta), 0]), ]

        super(IsoscelesMirrorTube, self).__init__(corners=corners, **kwargs)


class RectangularMirrorTube(MirrorTube):
    def __init__(self, w_cm, h_cm, **kwargs):
        hw = w_cm / 2.0
        hh = h_cm / 2.0
        corners = [np.array([-hw, -hh]),
                   np.array([-hw, hh]),
                   np.array([hw, hh]),
                   np.array([hw, -hh])]
        super(RectangularMirrorTube, self).__init__(corners=corners, **kwargs)


def test_ray_tracing():
    mirrors = RectangularMirrorTube(w_cm=2.0, h_cm=2.0)
    ray_span = 0.1
    rays = RayBundle.from_origin_to_plane((-ray_span, ray_span), (-ray_span, ray_span), 1.0, (9, 9))

    test = mirrors.trace(rays, 25.0, plot=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ray_tracing()

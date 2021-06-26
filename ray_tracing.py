import numpy as np
import cv2
import logging
import time
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from util import make_bounds, pct_str, Image
from skimage.morphology import skeletonize


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
        logging.info("Generated RayBundle with %i rays." % (self._active.size,))
        self._history = []

    @staticmethod
    def from_resolution_and_fov(resolution, image_plane_z, fov_x_deg=45.0, **kwargs):
        fov = np.deg2rad(fov_x_deg / 2.0)
        x_half_span = image_plane_z * np.tan(fov)
        xy_aspect = float(resolution[1]) / resolution[0]
        y_half_span = x_half_span / xy_aspect

        return RayBundle.from_origin_to_plane([-x_half_span, x_half_span],
                                              [-y_half_span, y_half_span],
                                              image_plane_z, resolution, **kwargs)

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
        px = np.linspace(x_span[0], x_span[1], res[1]) if res[1] > 1 else np.array(np.mean(x_span))
        py = np.linspace(y_span[0], y_span[1], res[0]) if res[0] > 1 else np.array(np.mean(y_span))

        x, y = np.meshgrid(px, py)
        img_coords = np.dstack((x, y, np.ones(x.shape) * z))

        # Create rays
        eye = np.zeros(3).reshape(1, 3)
        rays = img_coords - eye
        origins = 0 * rays
        return RayBundle(origins, rays, **kwargs)

    def get_active_rays(self):
        return self._origins[self._active], self._directions[self._active]

    def get_active_count(self):
        return np.sum(self._active)

    def get_active(self):
        return self._active

    def set_all_active(self):
        self._active = np.full(self._active.shape, True)

    def plot_3d(self, distances=0.1, mask=None, color=(0., 0., 0., 0.8), ax=None):
        """
        Plot lines from ray origins to distances.
        :param distances:   length of subset, should be scalar or same shape os self._origins.shape[:2]
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
        handle = ax.plot(x, y, z, color=color)

        return handle, ax

    def prism_intersect_dists(self, prism):
        """
        Intersection distance from ACTIVE rays to each face of prism
        NOTE:  does NOT clip at facet bounds
        NOTE:  ignores z-components, since is z-offset agnostic for a prism
        :param prism: Prism object
        :return: N x M distances, within N active rays an M mirrors
        """
        centers, normals = prism.get_mirrors()
        dists = [self.plane_intersect_dists(centers[i, :], normals[i, :]) for i in range(prism.get_n())]
        dists = np.hstack(dists)

        return dists

    def plane_intersect_dists(self, point, normal):
        """
        Return the distance to the intersection of ACTIVE rays with given.
        :param point:  a point on the plane
        :param normal:  normal vector to plane
        :return: H x 1 distances for N active rays
        """
        point = point.reshape(-1)
        normal = normal.reshape(-1)

        origins = self._origins[self._active].reshape(-1, 3)
        directions = self._directions[self._active].reshape(-1, 3)

        with np.errstate(divide='ignore', invalid='ignore'):
            # parallel rays go to np.inf
            dists = np.dot(point - origins, normal) / np.dot(directions, normal)

        return dists.reshape(-1, 1)

    def deactivate(self, inactive):
        self._active[_double_index(self._active, inactive)] = False

    def reflect(self, subset, intersections, plane_normal):

        self._origins[_double_index(self._active, subset)] = intersections.reshape(-1, 3)

        dirs = self._directions[self._active][subset]

        delta = - 2.0 * np.dot(dirs, plane_normal).reshape(-1, 1) * plane_normal.reshape(1, 3)

        new_directions = self._directions[self._active][subset] + delta
        new_directions /= np.linalg.norm(new_directions, axis=1, keepdims=True)

        self._directions[_double_index(self._active, subset)] = new_directions

    def get_shape(self):
        return self._directions.shape[:2]

    def plot_bounce_history(self, bounce_hist, ax=None, flat=True, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca() if flat else fig.gca(projection='3d')

        line_x = []
        line_y = []
        line_z = []

        line_x0 = []
        line_y0 = []
        line_z0 = []

        shape = self._directions.shape[:2]
        positions = self._origins.reshape([shape[0], shape[1], 3]) * 0

        for ray_i in range(len(bounce_hist)):
            for ray_j in range(len(bounce_hist[ray_i])):
                for b_i, bounce in enumerate(bounce_hist[ray_i][ray_j]):
                    x = [positions[ray_i][ray_j][0], bounce[0]]
                    y = [positions[ray_i][ray_j][1], bounce[1]]
                    z = [positions[ray_i][ray_j][2], bounce[2]]
                    if b_i > 0:
                        line_x.extend(x + [np.nan])
                        line_y.extend(y + [np.nan])
                        line_z.extend(z + [np.nan])
                    else:
                        line_x0.extend(x + [np.nan])
                        line_y0.extend(y + [np.nan])
                        line_z0.extend(z + [np.nan])

                    positions[ray_i][ray_j] = bounce
        if flat:
            x_coords = line_x0 + line_x
            y_coords = line_z0 + line_z
            handle = plt.plot(x_coords, y_coords, 'k-', **kwargs)
        else:
            plt.plot(line_x0, line_y0, line_z0, 'k-', alpha=.5,**kwargs)
            handle = plt.plot(line_x, line_y, line_z, 'k-', **kwargs)
        return handle


def _get_normals_from_points(c1, c2, c3):
    """
    Plane normals from three non-collinear points in the plane.
    Oriented so clockwise points away from the clock.

    :param c1: (x,y,z) point in plane
    :param c2: (x,y,z) point in plane, not equal to c1
    :param c3: (x,y,z) point in plane, not on line c2-c1
    :return: (x,y,z) normal pointing "up"
    """
    n = 3 if len(c1.shape) == 1 else 2

    co_planar_a = c2 - c1  # right-hand rule, to point inward ...
    co_planar_b = c3 - c1
    normals = np.cross(co_planar_a, co_planar_b)
    normals /= np.linalg.norm(normals)
    return normals


class Prism(object):
    """
    Class to handle geometry of a right prism.
    """

    def __init__(self, corners, height, center=True):
        """
        Init with list of 2d coordinates (i.e. closed polygon loop), representing view from the top.

        :param corners:  Nx2 array, or N-element list of (x, y) pairs, clockwise oriened corners of a N-sided polygon.
        :param height: height of prism sides
        :param center:  make avg vertex (0,0)

        """
        self._height = height
        bottom = height
        top = 0
        if not isinstance(corners, np.ndarray):
            corners = np.array(corners)
        self._n = corners.shape[0]

        if center:
            span = np.vstack([np.min(corners, axis=0),
                              np.max(corners, axis=0)])
            center_xy = np.mean(span, axis=0, keepdims=True)
            corners = corners - center_xy

        self._top_left = np.hstack((corners, np.ones(self._n).reshape(-1, 1) * top))
        self._bottom_left = np.hstack((corners, np.ones(self._n).reshape(-1, 1) * bottom))
        # shift & wrap
        self._top_right = np.hstack((np.vstack((corners[1:, :], corners[0, :])), np.ones(self._n).reshape(-1, 1) * top))
        self._bottom_right = np.hstack(
            (np.vstack((corners[1:, :], corners[0, :])), np.ones(self._n).reshape(-1, 1) * bottom))
        self._corners_2d = np.hstack((self._top_left[:, :2], self._top_right[:, :2]))

        self._z_centers = (bottom + top) / 2.0

        self._centers = (self._top_left + self._top_right + self._bottom_left + self._bottom_right) / 4.0  # rectangles
        normals = [_get_normals_from_points(self._centers[i, :],
                                            self._top_right[i, :],
                                            self._top_left[i, :]) for i in range(self._n)]
        self._normals = np.array(normals)

    def get_height(self):
        return self._height

    def get_corners(self):
        return self._corners_2d

    def get_mirrors(self):
        """
        Get mirrors coords
        :return:  center(s), normal(s)
        """
        return self._centers, self._normals

    def plot_3d(self, ax=None, z_offset=0.0, color=(0.1, .15, 1.0, .5), **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        offset_vector = np.array([0, 0, z_offset])

        top_right = self._top_right + offset_vector
        top_left = self._top_left + offset_vector

        bottom_right = self._bottom_right + offset_vector
        bottom_left = self._bottom_left + offset_vector
        all_corners = np.hstack([top_right, top_left, bottom_left, bottom_right])

        handles = plot_3d_polygon(all_corners, ax, color=color, **kwargs)
        return handles, ax

    def get_n(self):
        return self._n


def plot_3d_polygon(corners, ax, color=(0.1, .15, 1.0, .5), **kwargs):
    handles = []
    for i in range(corners.shape[0]):
        x = corners[i, ::3]
        y = corners[i, 1::3]
        z = corners[i, 2::3]
        verts = [list(zip(x, y, z))]  # list necessary python 2/3?

        poly = Poly3DCollection(verts)
        poly.set_color(color)
        handles.append(ax.add_collection3d(poly))
    return handles


class IsoscelesPrism(Prism):
    def __init__(self, theta_deg, h_cm, **kwargs):
        theta = np.deg2rad(theta_deg)
        corners = [np.array([-np.sin(theta), -h_cm / 2]),
                   np.array([0, h_cm / 2]),
                   np.array([np.sin(theta), -h_cm / 2]), ]
        corners = np.array(corners + corners[0])
        super(IsoscelesPrism, self).__init__(corners=corners, **kwargs)


class NGonPrism(Prism):
    def __init__(self, n, r, phi=0.0, **kwargs):
        logging.info("Making N-gon prism with n=%i, radius %.4f cm, angular offset %.5f." % (n, r, phi))

        theta = np.linspace(0 + phi, 2.0 * np.pi + phi, n + 1)[:-1]
        corners = np.vstack((np.cos(theta) * r, np.sin(theta) * r)).T
        super(NGonPrism, self).__init__(corners=corners, **kwargs)


class RectangularPrism(Prism):
    def __init__(self, w_cm, h_cm, **kwargs):
        hw = w_cm / 2.0
        hh = h_cm / 2.0
        corners = [np.array([-hw, -hh]),
                   np.array([-hw, hh]),
                   np.array([hw, hh]),
                   np.array([hw, -hh])]
        corners = np.array(corners + corners[0])

        super(RectangularPrism, self).__init__(corners=corners, **kwargs)


class MirrorTube(object):
    """
    Handle simulation of light in mirrors.

    Define a ortho-prismatic tube (shape) tube of mirrors, i.e. all perpendicular to flat, facing indwards.
    Input is an arbitrary list of 2-d polygon vertices.  (assumed clockwise, mirrors facing inwards)
    """

    def __init__(self, prism):
        """
        Define vertices of the mirror prism.

        :param prism:  Prism object, describing shape of mirror arrangement.
        """
        self._facets = prism
        logging.info("Created %i-mirror tube" % (self._facets.get_n(),))

    def get_height(self):
        return self._facets.get_height()

    def get_corners(self):
        return self._facets.get_corners()

    def plot_3d(self, *args, z_offset=0.0, **kwargs):
        return self._facets.plot_3d(*args, z_offset=z_offset, **kwargs)

    def trace(self, rays, ground_z_cm, scope_top_z_cm, max_reflect=10, record=False):
        """
        Bounce rays through mirror tube, see where they land on other side.

        :param rays:  a RayBundle object
        :param ground_z_cm:  when rays exit tube, how far back to ground plane / image?
        :param scope_top_z_cm:  How far from the eye (0,0,0) is the top of the 'scope?
        :param max_reflect:  how many reflections before ray is considered not hitting ground plane?
        :param record:  Save all reflections & return them, else None
        :return:  list(H x W x 3 array of ray destinations on the ground plane, or NAN if not hitting,
                       H x W array of ray distances traveled, or NAN if not hitting)
                       H x W array of number of reflections, or -1 if not hitting,
                       list of bounce history (ray intersections)
                           bounce_hist[i][j] = [(x,y,z)_0, (x,y,z)_1, ..., (x,y,z)_ground]
        """

        # list of lists, same shape as ray bundle, each list is a rays's history.
        bounce_record = [[[] for __ in range(rays.get_shape()[1])] for _ in range(rays.get_shape()[0])]
        max_dist = 2.0 * ground_z_cm  # FIX

        def accumulate_bounces(bounce_mask, intersections):
            """
            Some rays just hit something.  Record this.
            :param bounce_mask:  N element boolean array, which rays hit?
            :param intersections: np.sum(bounce_mask) x 3 (x,y,z) of hit locations
            """

            bounce_mask = bounce_mask.reshape(-1)
            if not record:
                return
            locs = np.where(bounce_mask)[0]
            if len(locs) == 0:
                return

            i_inds = (locs / rays.get_shape()[1]).astype(np.int64)
            j_inds = np.mod(locs, rays.get_shape()[1])
            for n in range(len(locs)):
                bounce_record[i_inds[n]][j_inds[n]].append(intersections[n])
                n += 1

        ground_center = np.array([0.0, 0.0, ground_z_cm]).reshape(1, -1)
        ground_normal = np.array([0.0, 0.0, -1.0])  # towards eye

        out_points = np.zeros(shape=list(rays.get_shape()) + [3])
        out_dists = np.zeros(shape=list(rays.get_shape()))
        out_reflects = np.zeros(shape=list(rays.get_shape()), dtype=np.int64) - 1

        # remember these so as to not hit same thing twice (-1 means invalid, n+1 means ground (bounce!))
        last_hits = np.zeros(np.prod(rays.get_shape()), dtype=np.int64) - 1

        # Start rays...
        _, m_normals = self._facets.get_mirrors()
        n = self._facets.get_n()
        rays.set_all_active()
        for iteration in range(max_reflect):

            active = rays.get_active()  # mask of full array, which are active in this iteration
            ray_starts, ray_dirs = rays.get_active_rays()
            n_active = active.sum()

            logging.info("Ray tracing iteration %i / %i - %s %% rays active." % (iteration + 1,
                                                                                 max_reflect,
                                                                                 pct_str(n_active, active.size)))
            if np.sum(n_active) == 0:
                logging.info('\tNo more active rays, trace complete.')
                break

            # calculate rays distances to all objects
            mirror_dists = rays.prism_intersect_dists(self._facets)
            ground_dists = rays.plane_intersect_dists(ground_center, ground_normal)
            # calculate intersections of all objects
            with np.errstate(divide='ignore', invalid='ignore'):  # some may now be inf
                ground_intersects = ground_dists * ray_dirs + ray_starts
                m_intersects = [np.tile(mirror_dists[:, m].reshape(-1, 1), (1, 3)) * ray_dirs + ray_starts for m in
                                range(n)]
                mirror_intersects = np.dstack(m_intersects)
            invalid_m_dists = np.isnan(mirror_dists)
            missed_all_mirrors = invalid_m_dists.copy()
            # Set disallowed intersections (from previous iteration) to infinity
            active_last_hits = last_hits[active.reshape(-1)]
            turn_off_mask = active_last_hits >= 0
            turn_off_idx = np.where(turn_off_mask)[0], active_last_hits[turn_off_mask]
            invalid_m_dists[turn_off_idx] = True

            # Also turn of ray/surface paris that are oriented incorrectly (non-positive distance, or too big)
            with np.errstate(divide='ignore', invalid='ignore'):  # some may now be inf
                diverging = np.logical_or(mirror_dists < 0, np.isinf(mirror_dists))
                diverging = np.logical_or(mirror_dists > max_dist, diverging)

            invalid_m_dists[diverging] = True

            # Now turn of ray/mirror intersections that are below the bottom of the mirror
            bottom_z = self._facets.get_height() + scope_top_z_cm
            too_low = mirror_intersects[:, 2, :] > bottom_z
            invalid_m_dists[too_low] = True

            mirror_dists[invalid_m_dists] = np.inf

            no_mirror = np.bitwise_and.reduce(np.isinf(mirror_dists), axis=1)
            logging.info("\t%s rays hit no mirror." % (np.sum(no_mirror),))

            # See if each ray hit a mirror first or the ground
            m_hits = np.argmin(mirror_dists, axis=1)  # ray is shortest distance from which mirror?
            closest_dists = mirror_dists[(np.arange(n_active), m_hits)]

            # Hit the ground?  record location & deactivate
            ground_hits = closest_dists > ground_dists.reshape(-1)
            logging.info("\tGround has %s rays hitting it first." % (np.sum(ground_hits),))
            if np.sum(ground_hits) != np.sum(no_mirror):
                logging.warning(
                    "The number of rays not hitting a mirror is different from the number hitting the ground.")
            # check for rays going over top
            if iteration == 0:
                over_top = np.logical_and(mirror_intersects[:, 2, :] < scope_top_z_cm,
                                          0.0 < mirror_intersects[:, 2, :])
                invalid_m_dists[over_top] = True  # mark as invalid
                missed_all_mirrors = np.bitwise_or.reduce(over_top, axis=1)  # Breaks nonconvex prisms!
                logging.info("\tFirst iteration has %i rays missing all mirrors, outside prism (%% %.3f)." % (
                    int(np.sum(missed_all_mirrors)), np.mean(missed_all_mirrors) * 100.))
                ground_hits = np.logical_or(ground_hits, missed_all_mirrors)

            idx = _double_index(active, ground_hits)
            accumulate_bounces(idx, ground_intersects[ground_hits, :])
            out_points[idx] = ground_intersects[ground_hits, :]
            out_dists[idx] = ground_dists[ground_hits].reshape(-1)
            out_reflects[idx] = iteration
            last_hits[idx.reshape(-1)] = n

            # See which rays hit which mirror first

            for mirror_i in range(n):
                mirror_hits = (m_hits == mirror_i) & np.logical_not(np.isinf(closest_dists)) & np.logical_not(
                    ground_hits)

                idx = _double_index(active, mirror_hits)
                last_hits[idx.reshape(-1)] = mirror_i  # don't hit again next time!
                logging.info(
                    "\tMirror %i has %s rays hitting it first" % (mirror_i, np.sum(mirror_hits),))

                accumulate_bounces(idx, mirror_intersects[mirror_hits, :, mirror_i])

                rays.reflect(mirror_hits, mirror_intersects[mirror_hits, :, mirror_i], m_normals[mirror_i, :])

            to_deactivate = np.logical_or(ground_hits, no_mirror)
            rays.deactivate(to_deactivate)

        return out_points.reshape(list(rays.get_shape()) + [3]), \
               out_dists.reshape(rays.get_shape()), \
               out_reflects.reshape(rays.get_shape()), \
               bounce_record

    def get_image_map(self, plot_map=False, **kwargs):
        coords, dists, bounce___hist, bounces = self.trace(**kwargs)
        if plot_map:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
            ax[0].imshow(dists)
            ax[1].imshow(bounces)
            plt.suptitle("Image map, distances in [%.3f, %.3f], bounces in [%.i, %.i]." % (
                np.min(dists), np.max(dists), np.min(bounces), np.max(bounces)))
            plt.show()

        return coords[:, :, :2], dists, bounces


def _double_index(mask, sub_mask):
    """
    Probably could be more efficient
    """
    full = mask.copy()
    for i, active in enumerate(np.where(full.reshape(-1))[0]):
        index = np.unravel_index(active, full.shape)
        full[index] = sub_mask[i]
    return full


def make_stained_glass(image, bounces, thresh=.5):
    bounces = bounces.astype(np.uint8)
    laplacian = cv2.Laplacian(bounces, cv2.CV_64F)
    grad = laplacian  # np.sqrt(sobely * sobely + sobelx * sobelx)
    """
    sobelx = cv2.Sobel(bounces, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(bounces, cv2.CV_64F, 0, 1, ksize=5)
    
    ksizes = [3, 5, 7, 9]
    sigmaxs = [0.5, 1.0, 1.5, 3.0, ]
    ind=0
    for ki, ksize in enumerate(ksizes):
        for si, sigmax in enumerate(sigmaxs):
            ind+=1
            plt.subplot(len(ksizes), len(sigmaxs), ind)
            blurred = cv2.GaussianBlur(np.uint8(laplacian > 0.001),
                                       ksize=ksize,
                                       sigmaX=sigmax,
                                       borderType=cv2.BORDER_REFLECT)
            plt.imshow(blurred)
            plt.xlabel("%i, %.2f" % (ksize, sigmax))
    plt.show()


    plt.imshow(b)
    plt.colorbar()
    plt.show()
    """
    skeleton = skeletonize(grad > thresh).astype(np.uint8)
    kern = np.ones((2, 2), dtype=np.uint8)
    skeleton = cv2.dilate(skeleton, kern, iterations=1)
    sk = np.where(skeleton > 0)
    if len(image.shape) > 2:
        channel_coord = np.zeros(sk[0].size, dtype=np.int64)
        image[(sk[0], sk[1], channel_coord)] = 0
        image[(sk[0], sk[1], channel_coord + 1)] = 0
        image[(sk[0], sk[1], channel_coord + 2)] = 0
    else:
        image[sk] = 0

    return image


def test_ray_tracing():
    # shape = RectangularPrism(w_cm=2.01, h_cm=2.01, top=2.54, bottom=50.0)
    geom = NGonPrism(n=3, r=1.012341234, top=2.54, bottom=50.0)
    # shape = IsoscelesPrism(10.0, .5, top=2.54, bottom=50.0)
    ray_span = 0.15
    ground_z_cm = 60.0
    # out_shape = (240,320)
    # out_shape = (100,100)
    out_shape = (1080, 1920)
    fov_x = 45.

    ray_span_v = float(out_shape[1]) / float(out_shape[0]) * ray_span
    rays = RayBundle.from_origin_to_plane((-ray_span_v / 2, ray_span_v / 2), (-ray_span / 2, ray_span / 2), 1.0,
                                          out_shape)

    rays = RayBundle.from_resolution_and_fov(resolution=out_shape,
                                             image_plane_z=4.0,
                                             fov_x_deg=fov_x)

    mirrors = MirrorTube(shape=geom)
    # load image
    img = Image.from_file('test_img.jpg', flip_bgr_rgb=True, px_per_cm=(150, 150))

    if False:
        # plot mirrors?
        ax = shape.plot_3d()

        # plot initial rays?
        rays.plot_3d(ax=ax, distances=30.)

        # plot image
        img.plot_3d(ax=ax, z_cm=ground_z_cm)

        plt.show()

    # ray-trace
    img_map, dists, bounce = mirrors.get_image_map(rays=rays, max_reflect=30,
                                                   ground_z_cm=ground_z_cm, plot=False)

    span = np.vstack((np.max(img_map.reshape(-1, 2), axis=0),
                      np.min(img_map.reshape(-1, 2), axis=0))).T

    pretty = img.interpolate(img_map, method='nearest')
    # pretty = img.interpolate_integer(img_map, bounce)

    plt.imshow(pretty)
    plt.axis('equal')
    plt.show()
    cv2.imwrite("K_out.jpg", pretty[:, :, ::-1])

    b_img = make_stained_glass(pretty, bounce)

    plt.imshow(b_img)
    plt.axis('equal')
    cv2.imwrite("K_out_stained_glass.jpg", b_img[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ray_tracing()

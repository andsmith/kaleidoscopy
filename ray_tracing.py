import numpy as np
import cv2
import logging
import time
import matplotlib.pylab as plt
import matplotlib.cm as cm
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
        px = np.linspace(x_span[0], x_span[1], res[1])
        py = np.linspace(y_span[0], y_span[1], res[0])

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

    def prism_intersect_dists(self, prism):
        """
        Intersection distance from ACTIVE rays to each face of prism
        :param prism: Prism object
        :return: N x M distances, withn N active rays an M mirrors
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
        self._origins[_double_index(self._active, subset)] = intersections

        new_directions = self._directions[self._active][subset] - 2.0 * np.dot(self._directions[self._active][subset],
                                                                               plane_normal).reshape(-1,
                                                                                                     1) * plane_normal.reshape(
            1, 3)

        self._directions[_double_index(self._active, subset)] = new_directions

    def get_shape(self):
        return self._directions.shape[:2]


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

    def __init__(self, corners, bottom, top):
        """
        Init with list of 2d coordinates (i.e. closed polygon loop), representing view from the top.

        :param corners:  Nx2 array, or N-element list of (x, y) pairs, clockwise oriened corners of a N-sided polygon.
        :param bottom: z-coordinate, bottom of prism
        :param top:  z-coordinate, top of prism
        """
        self._bottom = bottom
        self._top = top
        if not isinstance(corners, np.ndarray):
            corners = np.array(corners)
        self._n = corners.shape[0]
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

    def get_vertical_span(self):
        return (self._bottom, self._top)

    def get_corners(self):
        return self._corners_2d

    def get_mirrors(self):
        """
        Get mirrors coords
        :return:  center(s), normal(s)
        """
        return self._centers, self._normals

    def plot_3d(self, ax=None, color=(0.1, .15, 1.0, .5), **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        all_corners = np.hstack([self._top_right, self._top_left, self._bottom_left, self._bottom_right])
        for i in range(self._n):
            x = all_corners[i, ::3]
            y = all_corners[i, 1::3]
            z = all_corners[i, 2::3]
            verts = [list(zip(x, y, z))]  # list necessary python 2/3?

            poly = Poly3DCollection(verts)
            poly.set_color(color)
            ax.add_collection3d(poly)

        ax.set_xlim3d(np.min(self._corners_2d[:, 0]), np.max(self._corners_2d[:, 0]))
        ax.set_ylim3d(np.min(self._corners_2d[:, 1]), np.max(self._corners_2d[:, 1]))
        ax.set_zlim3d(-10.0, 100.0)

        return ax

    def get_n(self):
        return self._n


class IsoscelesPrism(Prism):
    def __init__(self, theta_deg, h_cm, **kwargs):
        theta = np.deg2rad(theta_deg)
        corners = [np.array([-np.sin(theta), -h_cm / 2]),
                   np.array([0, h_cm / 2]),
                   np.array([np.sin(theta), -h_cm / 2]), ]
        corners = np.array(corners + corners[0])
        super(IsoscelesPrism, self).__init__(corners=corners, **kwargs)


class NGonPrism(Prism):
    def __init__(self, n, r, **kwargs):
        logging.info("Making N-gon prism with n=%i, radius %.4f cm." % (n, r))

        theta = np.linspace(0, 2.0 * np.pi, n + 1)[:-1]
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

    def __init__(self, shape, **kwargs):
        """
        Define vertices of the mirror prism.

        :param shape:  Prism object, describing shape of mirror arrangement.
        """
        self._facets = shape
        logging.info("Created %i-mirror tube" % (self._facets.get_n(),))

    def get_vertical_span(self):
        return self._facets.get_vertical_span()

    def get_corners(self):
        return self._facets.get_corners()

    def trace(self, rays, ground_z_cm, max_reflect=10, plot=False, record=False):
        """
        Bounce rays through mirror tube, see where they land on other side.

        :param rays:  a RayBundle object
        :param ground_z_cm:  when rays exit tube, how far back to ground plane / image?
        :param max_reflect:  how many reflections before ray is considered not hitting ground plane?
        :param plot: show in 3d
        :param record:  Save all reflections & return them, else None
        :return:  list(H x W x 3 array of ray destinations on the ground plane, or NAN if not hitting,
                       H x W array of ray distances traveled, or NAN if not hitting)
                       H x W array of number of reflections, or -1 if not hitting,
                       list of bounce history (ray intersections)
                           bounce_hist[i][j] = [(x,y,z)_0, (x,y,z)_1, ..., (x,y,z)_ground]
        """
        import ipdb; ipdb.set_trace()

        # list of lists, same shape as ray bundle, each list is a rays's history.
        bounce_record = [[[] for __ in range(rays.get_shape()[1])] for _ in range(rays.get_shape()[0])]

        def accumulate_bounces(bounce_mask, intersections):
            """
            Some rays just hit something.  Record this.
            :param bounce_mask:  N element boolean array, which rays hit?
            :param intersections: N x 3 (x,y,z) of hit locations
            """
            if not record:
                return
            locs = np.where(bounce_mask)[0]
            if len(locs) == 0:
                return
            i_inds = (locs / rays.get_shape[1]).astype(np.int64)
            j_inds = np.mod(locs, rays.get_shape[1])
            n = 0
            for i in i_inds:
                for j in j_inds:
                    bounce_record[i][j] = intersections[n]
                    n += 1

        ground_center = np.array([0.0, 0.0, ground_z_cm]).reshape(1, -1)
        ground_normal = np.array([0.0, 0.0, -1.0])  # towards eye

        out_points = np.zeros(shape=list(rays.get_shape()) + [3])
        out_dists = np.zeros(shape=list(rays.get_shape()))
        out_reflects = np.zeros(shape=list(rays.get_shape()), dtype=np.int64) - 1

        # remember these so as to not hit same thing twice (-1 means invalid, n+1 means ground (bounce!))
        last_hits = np.zeros(np.prod(rays.get_shape()), dtype=np.int64) - 1

        # Start rays...
        n = self._facets.get_n()
        rays.set_all_active()
        for iteration in range(max_reflect):
            active = rays.get_active()
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
            dists = np.hstack([mirror_dists, ground_dists])

            # Set disallowed intersections to infinity
            active_last_hits = last_hits[active.reshape(-1)]
            turn_off_mask = active_last_hits >= 0
            idx = np.where(turn_off_mask)[0], active_last_hits[turn_off_mask]
            dists[idx] = np.inf

            # Also turn of ray/surface paris that are oriented incorrectly (non-positive distance)
            with np.errstate(divide='ignore', invalid='ignore'):  # some may now be inf
                diverging = np.logical_or(dists < 0, np.isinf(dists))
            dists[diverging] = np.inf

            # ray[i] doesn't hit anything?
            all_bad = np.bitwise_and.reduce(np.isinf(dists), axis=1)
            if np.sum(all_bad) > 0:
                logging.warning("\t%s pixels hit nothing..." % (np.sum(all_bad),))

            logging.info(
                "\tIteration has %s rays diverging from %i mirrors + background." % (np.sum(diverging), n))

            hits = np.argmin(dists, axis=1)

            # these hit nothing, what to do with them?
            # FIX hits[all_bad] = -1

            ray_starts, ray_dirs = rays.get_active_rays()
            # Hit the ground?  Save.
            ground_hits = hits == n
            logging.info("\tGround has %s rays hitting it first." % (np.sum(ground_hits),))
            intersects = ground_dists[ground_hits] * ray_dirs[ground_hits, :] + ray_starts[ground_hits, :]
            idx = _double_index(active, ground_hits)
            accumulate_bounces(idx, intersects)
            out_points[idx] = intersects
            out_dists[idx] = ground_dists[ground_hits].reshape(-1)
            out_reflects[idx] = iteration
            last_hits[idx.reshape(-1)] = n

            # for live plots
            palette = cm.get_cmap('brg')
            color_indices = np.linspace(0, 1.0, n + 1)
            colors = [palette(i) for i in color_indices]
            if plot:
                ax = self._facets.plot_3d()

            # See which rays hit which mirror first
            _, m_normals = self._facets.get_mirrors()
            for mirror_i in range(n):  # ## FIX:  Add bounds check to mirrors!
                mirror_hits = hits == mirror_i
                idx = _double_index(active, mirror_hits)
                last_hits[idx.reshape(-1)] = mirror_i  # don't hit again next time!
                logging.info(
                    "\tMirror %i has %s rays hitting it first" % (mirror_i, np.sum(mirror_hits),))
                with np.errstate(divide='ignore', invalid='ignore'):
                    mirror_intersects = dists[:, mirror_i][mirror_hits].reshape(-1, 1) * ray_dirs[mirror_hits] + \
                                        ray_starts[mirror_hits]

                if plot:
                    rays.plot_3d(dists[:, mirror_i][mirror_hits], mirror_hits, ax=ax)
                rays.reflect(mirror_hits, mirror_intersects, m_normals[mirror_i, :])
                if plot:
                    rays.plot_3d(2, mirror_hits, color=colors[mirror_i], ax=ax)

            if plot:
                ax.set_aspect('auto')
                plt.show()

            to_deactivate = np.logical_or(ground_hits, all_bad)
            rays.deactivate(to_deactivate)

        return out_points.reshape(list(rays.get_shape()) + [3]), \
               out_dists.reshape(rays.get_shape()), \
               out_reflects.reshape(rays.get_shape()),

    def get_image_map(self, plot_map=False, **kwargs):
        coords, dists, bounces = self.trace(**kwargs)
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
    r = mask.copy()
    r[mask] = sub_mask
    return r


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
    import ipdb;
    ipdb.set_trace()
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

import time
from multiprocessing import cpu_count, Pool
from threading import Thread, Lock
import logging
import numpy as np
from surfaces import Cylinder, Plane
from mirrors_rectangle import RectangularPrism  # testing
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

DEBUGGING = True


class RayTracer(object):
    _NO_HIT_Z = -1.0  # real hits are positive (?)
    _REL_TOL = 1e-6

    def __init__(self,
                 mirrors,
                 output_shape,
                 target_shape,
                 update_callback,
                 n_cores=0,
                 fov_deg_x=20.0):
        """
        :param mirrors:  Mirrors() object, set of mirrors
        :param img_shape:  H x W of output image, and of target for rays (i.e. camera image)
        :param update_callback:  function(new_map, stats) to call when results update
            new_map is dict{'mapping': HxW array, int64 of (single-index) locations of sources for each pixel,
                            'bounce_counts':  (same type) how many mirrors each ray hit
                            'distances':  (same dims, but float) how far each ray travelled}
        :param n_cores: how many cores to use during distance calculations (1 = single process)
        :param fov_deg_x:  Field of view along x-axis.
        """
        self._mirrors = mirrors
        self._n_cores = n_cores if n_cores != 0 else cpu_count() - 1  # save one for rendering?
        logging.info("Ray-tracer created running with %i cores." % (self._n_cores,))
        self._pool = Pool(processes=self._n_cores) if self._n_cores != -1 else None
        self._callback = update_callback
        self._render_thread = Thread(target=self._trace)

        # state
        self._update_result_lock = Lock()
        self._shutdown = False

        # Ray-tracing things
        self._bounce_index = 0
        self._rays = None
        self._mirror_surfaces = None  # things rays hit
        self._fov = np.deg2rad(fov_deg_x)
        self._target_shape = target_shape
        self._img_shape = output_shape
        self._target_aspect = float(target_shape[1]) / float(target_shape[0])
        self._img_aspect = float(output_shape[1]) / float(output_shape[0])

        # results
        self._bounce_counts = np.zeros(self._img_shape, dtype=np.int64)  # how many bounces for each ray
        self._ray_distances = np.zeros(self._img_shape, dtype=np.float64)  # how far did ray travel?
        self._hit_xyz_coords = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.float64) - 1.0
        self._mapping = np.zeros(self._img_shape, dtype=np.int64)
        self._stats = {'bounce_num': None,  # current bounce number
                       'rays_hit': None,  # so far
                       'ray_count': None}  # total

    def _trace(self):
        self._setup_geometry()
        self._rays = RayBundle(shape=self._img_shape,
                               direction=self._view_dir,
                               z_dist=self._img_z,
                               span=(self._img_w, self._img_h))

        while not self._shutdown and self._rays.get_active_count() > 0:
            logging.info("Ray-tracer main loop starting bounce number %i..." % (self._bounce_index,))
            self._bounce()  # advance rays to next surface, accumulate results
            self._callback(self.get_current_result(), self._stats)
        logging.info("Ray-tracer main loop complete.")

    def _setup_geometry(self, view_dir=(0.0, 0.0, 1.0), target_image_dist_m=1.0):
        """
        Determine scene layout:
           Target image width/height, Such that image subtends full window w/o mirrors
           Image plane distance, st. input & output match w/o mirrors

        Then calculate initial rays, i.e. grid on image plane pointing away from origin.

        """
        # MAX_IMG_Z = 0.1  # artificial limit
        max_ins_rad = self._mirrors.get_inscribed_radius(scaled=True)
        # origin = np.array([0.0, 0.0, 0.0])  # focus, eyeball
        self._target_z = target_image_dist_m
        self._target_w = 2.0 * np.tan(self._fov / 2.0) / self._target_z
        self._target_h = self._target_w / self._target_aspect
        self._img_w = np.sqrt(max_ins_rad ** 2.0 / (0.25 + (self._target_h / self._target_w) ** 2.0))
        self._img_h = self._img_w / self._img_aspect
        self._img_z = self._img_w * (
                self._target_z / self._target_w)  # when output rectangle just fits inscribed radius
        self._view_dir = np.array(view_dir)

    def _bounce(self):
        all_surfs = [self._target_surface] + self._mirror_surfaces

        # pairwise distance of all rays to all surfaces
        distances = self._rays.get_distances_to_surfaces(all_surfs)  # n_rays x n_surfs
        valid = distances >= 0
        distances[np.logical_not(valid)] = np.Inf
        all_invalid = np.sum(np.isInf(distances), axis=1).reshape(-1)
        if np.sum(all_invalid) > 0:
            raise Exception("Rays hitting nothing:  %s" % (np.sum(all_invalid),))
        # what hit what?
        hit_inds = np.argmin(distances, axis=1)
        hit_lists = [np.where(hit_inds == i)[0] for i in range(len(all_surfs))]
        targ_hits = hit_lists[0]

        # reflect rays hitting mirrors
        ground_intersections = None
        for s_ind, surf in enumerate(all_surfs):

            # should not depend on mirror arrangement, for debugging
            distance_traveled = distances[targ_hits[s_ind]]
            ray_indices = self._rays.get_ray_indices(targ_hits[s_ind])
            self._ray_distances[ray_indices] += distance_traveled

            intersections, surf_norms = surf.get_intersections_and_normals(self._rays.get_active_rays())
            if s_ind == 0:
                ground_intersections = intersections
                continue  # don't reflect off target

            # reflect non-target intersections (mirrors)
            self._rays.reflect(surf_norms, intersections, mask=targ_hits[s_ind])

        # collect rays hitting target
        ray_indices = self._rays.get_ray_indices(targ_hits)

        # update results
        with self._update_result_lock:
            self._bounce_counts[ray_indices] = self._n_bounce
            self._target_plane_xyz_intersect[ray_indices] = ground_intersections
            self._mapping = self._get_new_map()
            self._stats = {'rays_hit': np.sum(self._hit_xyz_coords[:, :, 0] != self._NO_HIT_Z),
                           'ray_count': np.prod(self._img_shape),
                           'bounce_num': self._n_bounce}

    def start(self):
        if not DEBUGGING:
            self._render_thread.start()
        else:
            self._trace()

    def get_current_result(self):
        with self._update_result_lock:
            return {'mapping': self._mapping.copy(),
                    'bounce_counts': self._bounce_counts.copy(),
                    'distances': self._ray_distances.copy(),
                    }

    def shutdown(self):
        self._shutdown = True

    def _get_new_map(self):
        where_hits = np.where(self._hit_xyz_coords[:, :, 2] != self._NO_HIT_Z)
        hits_xyz = self._hit_xyz_coords[where_hits]
        hits_i = hits_xyz[0]
        hits_j = hits_xyz[1]

        x_scaled = self._hit_xyz_coords[hits_i, 0] * self._target_xy_scale[0] + (self._img_shape[1] / 2.0)
        y_scaled = self._hit_xyz_coords[hits_j, 1] * self._target_xy_scale[1] + (self._img_shape[0] / 2.0)

        valid = np.logical_and(np.logical_and(x_scaled >= 0, x_scaled < self._img_shape[1]),
                               np.logical_and(y_scaled >= 0, y_scaled < self._img_shape[0]), )
        invalid = np.logical_not(valid)
        hits_i = hits_i[valid]
        hits_j = hits_j[valid]
        x_scaled = np.int64(x_scaled)
        y_scaled = np.ing64(y_scaled)
        xy_scaled = x_scaled + y_scaled * self._img_shape[0]
        mapping = np.zeros(self._img_shape, dtype=np.int64)
        mapping[(hits_i[valid], hits_j[valid])] = xy_scaled[valid]
        mapping[(hits_i[invalid], hits_j[invalid])] = 0
        return mapping


def unit_vecs(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class RayBundle(object):
    """
    Represents a grid of rays, update as they reflect, etc.
    Preserve array shape using mask.
    """

    def __init__(self, shape, span, direction, z_dist):
        """
        initialize with explicit rays
        :param shape:  tuple, H x W, number of rays
        :param span: tuple, (width, height)
        :param direction:  x,y,z view direction (center pixel)
        :param z_dist: initialize mask to this value

        """
        self._shape = shape  # h, w
        self._span = span  # x, y
        self._n_rays = np.prod(shape)
        self._ray_starts = np.zeros((shape[0], shape[1], 3), dtype=np.float64)

        ray_x = np.linspace(-span[0], span[0], shape[1])
        ray_y = np.linspace(-span[1], span[1], shape[0])
        ray_endpoints_x, ray_endpoints_y = np.meshgrid(ray_y, ray_x)
        ray_endpoints_z = 0 * ray_endpoints_x + z_dist
        ray_endpoints = np.dstack((ray_endpoints_x, ray_endpoints_y, ray_endpoints_z))

        rays = unit_vecs(ray_endpoints - self._ray_starts)
        rays = rays / np.linalg.norm(rays, axis=2)
        standard_ray = np.array([0.0, 0.0, 1.0])
        rotation = np.cross(standard_ray, direction)
        rot_mat = Rotation.from_rotvec(rotation)
        self._rays = np.dot(rays, rot_mat)
        self.plot3d()
        plt.show()

        self._active = np.full(self._shape, True)  # binary indicators

        logging.info("Generated RayBundle with %i rays." % (self._n_rays,))
        self._history = []

    def get_active_count(self):
        return self._active.size

    def reflect(self, intersections, plane_normal, subset_mask=None):

        if subset_mask is None:
            indices = self._active
        else:
            indices = self._active[subset_mask]

        # new origin is just intersection point
        new_origins = intersections.reshape(-1, 3)

        # new direction has component parallel to normal reversed
        dirs = self._ray_directions[indices]
        delta = - 2.0 * np.dot(dirs, plane_normal).reshape(-1, 1) * plane_normal.reshape(1, 3)
        new_directions = dirs + delta
        new_directions /= np.linalg.norm(new_directions, axis=1, keepdims=True)

        # get distance traveled for this update
        distances = np.linalg.norm(self._ray_origins[indices] - new_origins, axis=1)

        self._ray_origins[indices] = new_origins
        self._ray_directions[indices] = new_directions
        return distances

    def plot_3d(self, length=0.1, color=(0., 0., 0., 0.8), ax=None):
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
        n = np.sum(self._active)
        x = np.zeros(n * 3, dtype=np.float64)
        y = 0 * x
        z = 0 * x
        destinations = self._ray_starts + self._rays * length
        x[0::3] = self._ray_starts[:, 0]
        x[1::3] = destinations[:, 0]
        x[2::3] = np.nan
        y[0::3] = self._ray_starts[:, 1]
        y[1::3] = destinations[:, 1]
        y[2::3] = np.nan
        z[0::3] = self._ray_starts[:, 2]
        z[1::3] = destinations[:, 2]
        z[2::3] = np.nan
        ax.scatter(x[0::3], y[0::3], z[::3], color=color)
        handle = ax.plot(x, y, z, color=color)

        return handle, ax

    '''
    def get_prism_intersections(self, prism, **kwargs):
        """
        Intersection distance from ACTIVE rays to each face of prism
        NOTE:  does NOT clip at facet bounds
        NOTE:  ignores z-components, since is z-offset agnostic for a prism
        :param prism: Prism object
        :return: N x M distances, within N active rays an M mirrors
        """
        centers, normals = prism.get_mirrors()
        results = [self.get_plane_intersections(centers[i, :],
                                                normals[i, :],
                                                **kwargs) for i in range(prism.get_n())]
        intersects = None
        dists = np.hstack([result[0] for result in results])
        if len(results) > 0 and results[0][1] is not None:
            intersects = np.dstack([result[1] for result in results])
        return dists, intersects


    def deactivate(self, inactive):
        self._active[_double_index(self._active, inactive)] = False


    def get_shape(self):
        return self._directions.shape[:2]

    

    def plot_bounce_history(self, bounce_hist, mask=None, ax=None, flat=True, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca() if flat else fig.gca(projection='3d')

        line_x = []
        line_y = []
        line_z = []

        line_x0 = []
        line_y0 = []
        line_z0 = []

        histories = [bounce_hist[ray_i][ray_j]
                     for ray_i in range(len(bounce_hist))
                     for ray_j in range(len(bounce_hist[ray_i]))
                     if mask is not None and mask[ray_i][ray_j]]
        positions = np.zeros(shape=(len(histories), 3), dtype=np.float64)  # all rays start at origin
        for ind in range(len(histories)):
            for b_i, bounce in enumerate(histories[ind]):
                x = [positions[ind][0], bounce[0]]
                y = [positions[ind][1], bounce[1]]
                z = [positions[ind][2], bounce[2]]
                if b_i > 0:
                    line_x.extend(x + [np.nan])
                    line_y.extend(y + [np.nan])
                    line_z.extend(z + [np.nan])
                else:
                    line_x0.extend(x + [np.nan])
                    line_y0.extend(y + [np.nan])
                    line_z0.extend(z + [np.nan])

                positions[ind] = bounce
        if flat:
            x_coords = line_x0 + line_x
            y_coords = line_z0 + line_z
            handle = ax.plot(x_coords, y_coords, 'k-', **kwargs)
        else:
            plt.plot(line_x0, line_y0, line_z0, 'k-', alpha=.5, **kwargs)
            handle = ax.plot(line_x, line_y, line_z, 'k-', **kwargs)
        return handle


class MirrorTube(object):
    """
    Handle simulation of light in mirrors.
    Vectorized ray-tracing!

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

    def get_view_rad(self):
        return self._facets.get_inscribed_rad()

    def trace(self, rays, ground_z_cm, scope_top_z_cm, max_reflect=10, record=False):
        """
        Bounce rays through mirror tube, see where they land on other side.

        :param rays:  a RayBundle object
        :param ground_z_cm:  when rays exit tube, how far back to ground plane / image?
        :param scope_top_z_cm:  How far from the eye (0,0,0) is the top of the 'scope?
        :param max_reflect:  how many reflections before ray is considered not hitting ground plane?
        :param record:  Save all reflections & return them, else None
        :return:  dict('image_map': H x W x 3 array of ray destinations
                       'ray_distances': H x W array of ray distances traveled, or NAN if not hitting)
                       'ray_bounce_counts': H x W array of number of reflections, or -1 if not hitting,
                       'bounce_histories': list of ray intersections, i.e. b_h[i][j] =
                           [(x,y,z)_0, (x,y,z)_1, ..., (x,y,z)_ground]
                       'hit_top':  H X W bool, which rays were inside mirror polygon, but not in view-circle
                       'missed_scope': H x W bool which rays that missed scope entirely )

        """

        result = {'image_map': np.zeros(shape=list(rays.get_shape()) + [3]),
                  'ray_distances': np.zeros(shape=list(rays.get_shape())),
                  'ray_bounce_counts': np.zeros(shape=list(rays.get_shape()), dtype=np.int64) - 1,
                  'missed_scope': np.zeros(shape=list(rays.get_shape()), dtype=bool),
                  'hit_top': np.zeros(shape=list(rays.get_shape()), dtype=bool),
                  'bounce_histories': [[[] for __ in range(rays.get_shape()[1])]
                                       for _ in range(rays.get_shape()[0])]}

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
                result['bounce_histories'][i_inds[n]][j_inds[n]].append(intersections[n])
                n += 1

        ground_center = np.array([0.0, 0.0, ground_z_cm]).reshape(1, -1)
        up_normal = np.array([0.0, 0.0, -1.0])  # towards eye

        # remember these so as to not hit same thing twice (-1 means invalid, n+1 means ground (bounce!))
        last_hits = np.zeros(np.prod(rays.get_shape()), dtype=np.int64) - 1

        # maximum ray length is constant, so use as error check
        ray_dist_to_ground = rays.get_plane_intersections(ground_center, up_normal, no_points=True)[0]
        max_dist = np.max(ray_dist_to_ground) * 1.10  # margin necessary?

        # Check rays inside view-circle
        r = self._facets.get_inscribed_rad()
        dist_to_img_plane, img_plane_intersects = rays.get_plane_intersections(np.array([0.0,
                                                                                         0.0,
                                                                                         scope_top_z_cm]),
                                                                               up_normal)
        radii = np.linalg.norm(img_plane_intersects[:, :2], axis=1)
        outside_circle = radii > r

        # Start rays...
        _, m_normals = self._facets.get_mirrors()
        n = self._facets.get_n()
        rays.set_all_active()
        hit_top = None  # deactivate these first time through
        for iteration in range(max_reflect):

            active = rays.get_active()  # which rays are active in this iteration, bitmask
            n_active = active.sum()

            logging.info("Ray tracing iteration %i / %i - %s %% rays active." % (iteration + 1,
                                                                                 max_reflect,
                                                                                 pct_str(n_active, active.size)))
            if np.sum(n_active) == 0:
                logging.info('\tNo more active rays, trace complete.')
                break
            # calculate current rays' distances to all objects
            mirror_dists, mirror_intersects = rays.get_prism_intersections(self._facets)
            ground_dists, ground_intersects = rays.get_plane_intersections(ground_center, up_normal)
            invalid_m_dists = np.isnan(mirror_dists)  # ground intersects should always be valid

            # Turn off disallowed intersections (mirror hit from previous iteration), by setting dist to inf.
            active_last_hits = last_hits[active.reshape(-1)]
            valid_active_last_hits = active_last_hits >= 0
            turn_off_idx = np.where(valid_active_last_hits)[0], active_last_hits[valid_active_last_hits]
            invalid_m_dists[turn_off_idx] = True

            # Also turn of ray/surface paris that are oriented incorrectly (non-positive distance, or too big).
            with np.errstate(divide='ignore', invalid='ignore'):  # some may now be inf
                diverging = np.logical_or(mirror_dists < 0, np.isinf(mirror_dists))
                diverging = np.logical_or(mirror_dists > max_dist, diverging)
            invalid_m_dists[diverging] = True

            # Now turn of ray/mirror intersections  that are below the bottom of the mirror.
            bottom_z = self._facets.get_height() + scope_top_z_cm
            too_low = mirror_intersects[:, 2, :] > bottom_z
            invalid_m_dists[too_low] = True
            mirror_dists[invalid_m_dists] = np.inf

            # See which rays hit the ground before any mirrors (inside).
            m_hits = np.argmin(mirror_dists, axis=1)  # ray is shortest distance from which mirror?
            closest_dists = mirror_dists[(np.arange(n_active), m_hits)]
            ground_hits = closest_dists > ground_dists.reshape(-1)

            # check for rays going over top & outside circle (only first time)
            if iteration == 0:
                # ray goes over scope if it goes over any mirror (breaks non-convexity)
                missed_scope = np.logical_and(mirror_intersects[:, 2, :] < scope_top_z_cm,
                                              0.0 < mirror_intersects[:, 2, :])
                missed_scope = np.logical_or.reduce(missed_scope, axis=1)
                hit_top = np.logical_and(np.logical_not(missed_scope), outside_circle)

                logging.info("\tIn first iteration, %i rays hit the top, %i rays miss." % (
                    np.sum(hit_top), np.sum(missed_scope)))

                # save rays that hit the top
                hit_top_idx = _double_index(active, hit_top)  # should be all active, but be sure...
                result['hit_top'][hit_top_idx] = True
                result['image_map'][hit_top_idx, :] = img_plane_intersects[hit_top, :]

                # and rays that missed
                missed_scope_idx = _double_index(active, missed_scope)
                result['missed_scope'][missed_scope_idx] = True

                result['ray_distances'][missed_scope_idx] = dist_to_img_plane[missed_scope].reshape(-1)
                accumulate_bounces(missed_scope_idx, img_plane_intersects[missed_scope, :])
                ground_hits = np.logical_or(ground_hits, missed_scope)  # add to list of rays that hit ground

            # save & mark for deactivation all rays that hit the ground.
            logging.info("\tGround has %s rays hitting it first." % (np.sum(ground_hits),))
            idx = _double_index(active, ground_hits)
            accumulate_bounces(idx, ground_intersects[ground_hits, :])  # save history of ground hits
            result['image_map'][idx] = ground_intersects[ground_hits, :]
            result['ray_distances'][idx] += ground_dists[ground_hits].reshape(-1)
            result['ray_bounce_counts'][idx] = iteration
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

                distances = rays.reflect(mirror_hits, mirror_intersects[mirror_hits, :, mirror_i],
                                         m_normals[mirror_i, :])
                result['ray_distances'][idx] += distances

            to_deactivate = ground_hits
            if iteration == 0:
                to_deactivate = np.logical_or(hit_top, to_deactivate)
            rays.deactivate(to_deactivate)

        return result

    def get_image_map(self, plot_map=False, **kwargs):
        result = self.trace(**kwargs)
        if plot_map:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
            ax[0].imshow(result['ray_distances'])
            ax[1].imshow(result['ray_bounce_counts'])
            plt.suptitle("Image map, distances in [%.3f, %.3f], bounces in [%.i, %.i]." % (
                np.min(result['ray_distances']),
                np.max(result['ray_distances']),
                np.min(result['ray_bounce_counts']),
                np.max(result['ray_bounce_counts'])))
            plt.show()

        return result['image_map'][:, :, :2], result


def _double_index(mask, sub_mask):
    """
    :param mask:  N element bool array
    :param sub_mask:  bool array, length = np.sum(mask)
    :return:  N element bool array, sum(return) = sum(mask)
    """
    full = mask.copy()
    full[full] = sub_mask
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
'''

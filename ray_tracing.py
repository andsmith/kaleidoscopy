"""
Trace rays of light from the eye through the scope to the target plane.
Runs in parallel on N cores.
"""

import time
from multiprocessing import cpu_count, Pool, Pipe
from threading import Thread, Lock
import logging
import numpy as np
from surfaces import Plane


class ScopeTracer(object):
    """
    High-level ray-tracing manager:
        1.  Initialize rays, set all to live.
        2.  Break live rays into blocks for parallel tracing.
        2.  Bounce blocks of rays in parallel using multiprocessing.Pool
        4.  Find border-pixels to anti-alias & dispatch to RayBlockTracer() to render.
    """

    def __init__(self,
                 mirrors,
                 input_shape,
                 output_shape,
                 update_callback,
                 n_cores=0):
        """

        :param mirrors:  Mirrors() object, set of mirrors
        :param input_shape:  H x W of what kaleidoscope is looking at (image or video frames)
        :param output_shape:  H x W of output image
        :param update_callback:  function(new_map) to call when results update
        :param n_cores: how many cores to use during distance calculations (1: single process, 0: n_cores - 1)
        """
        self._mirrors = mirrors
        self._n_cores = n_cores if n_cores > 0 else cpu_count() - 1  # save one for renderer

        self._pool = Pool(processes=self._n_cores) if self._n_cores > 1 else None
        self._callback = update_callback
        self._pipe = Pipe(duplex=True)

        self._input_shape = input_shape
        self._output_shape = output_shape

        # self._update_result_lock = Lock()
        self._shutdown = False

        self._bounce_index = 0  # aka, iteration count
        self._mapping = None  # flattened array mapping output pixel locations to input pixel locations (lookup)
        self._setup_rays()

        logging.info("ScopeTracer() created running with %i cores." % (self._n_cores,))

    def _setup_rays(self):

        # eye / focus
        self._z_dist = 1.0
        self._origin = np.array([0.0, 0.0, self._z_dist])
        self._fov = np.deg2rad(30.0)

        # target/image (thing rays hit) dimensions
        self._input_width = np.tan(self._fov) * self._origin * 2
        dpi = self._input_shape[1] / self._input_width
        self._input_height = self._input_shape[0] / dpi
        self._input_zoom = 1.0
        self._input_pan_xy = np.array([0.0, 0.0])

        # for calculating xy -> pixel index
        self._pixel_size = 1.0 / dpi
        self._x_min = -self._input_width / 2. - self._pixel_size / 2.0  # half pixel offset
        self._y_min = -self._input_height / 2. - self._pixel_size / 2.0

        # output dimensions (where rays would hit target if mirrors weren't there)
        # match width of input dimensions, set height to keep aspect ratio
        output_dpi = self._output_shape[1] / self._input_width
        output_height = self._output_shape[0] / output_dpi

        logging.info("Target area dimensions:  W:  %.2f,  H:  %.2f" % (self._input_width, self._input_height))

        # ray vectors
        ray_extent = {'z': self._z_dist,
                      'x': self._input_width,
                      'y': output_height}
        self._ray_origins, self._ray_directions = make_unit_rays(self._output_shape, ray_extent)

        # flatten, for dividing up for multiprocessing
        self._ray_origins = self._ray_origins.reshape((-1, 3))
        self._ray_directions = self._ray_directions.reshape((-1, 3))
        self._n_rays = np.prod(self._output_shape)

        # results go in these
        self._active = np.tile(True, self._n_rays)  # Rays still bouncing
        self._bounce_counts = np.zeros(self._n_rays, dtype=np.int64)  # how many bounces for each ray until it hit
        self._ray_distances = np.zeros(self._n_rays,
                                       dtype=np.float64)  # how far did ray travel? (should be independent of mirrors)
        self._hit_xyz = np.zeros((self._n_rays, 3), dtype=np.float64) - 1.0  # z should be zero...

    def run(self):

        while not self._shutdown and np.sum(self._active) > 0:
            logging.info("Ray-tracer main loop starting bounce number %i." % (self._bounce_index,))

            if self._iterate():
                logging.info("Ray-tracing iterate() returned no active rays.")
                break
            logging.info("Iteration %i complete, mapping updated." % (self._bounce_index,))

        logging.info("Ray-tracer main mapping completed in %i iterations." % (self._bounce_index,))

    def _iterate(self):

        active = np.where(self._active)[0]
        n_active = active.size
        if n_active == 0:
            return True

        result = _bounce(self._mirrors, self._ray_origins[active, :], self._ray_directions[active, :])

        # START WORK HERE:  need to figure out active/subset hit mask indices, etc.
        target_hit_inds = active[result['ground_hit_mask']]
        remaining_active_inds = active[np.logical_not(result['ground_hit_mask'])]

        self._hit_xyz[target_hit_inds, :] = result['hit_xyz_locations']
        self._ray_origins[remaining_active_inds] = result['new_ray_origins']
        self._ray_directions[remaining_active_inds] = result['new_ray_directions']

        self._active[target_hit_inds] = False
        logging.info("\tRay-tracing iteration %i ending with %i / %i rays active (%.3f %%)",
                     self._bounce_index,
                     np.sum(self._active), self._active.size,
                     100 * np.mean(self._active))

        self._mapping = self._get_mapping()
        self._callback(self._mapping)

    def _get_mapping(self):
        """
        Figure out which pixels are under the hit_xyz locations
        Each pixel is a square around it's coordinate.
        :returns: array of indices: M[a] = b means the output pixel at location a should be the
            value of the input pixel at location b (using flattened, 1-dim coords)
        """
        x_max, y_max = -self._x_min, - self._y_min
        valid = (self._hit_xyz[:, 2] > -1.0) \
                & (self._hit_xyz[:, 0] > self._x_min) & (self._hit_xyz[:, 0] < x_max) \
                & (self._hit_xyz[:, 1] > self._y_min) & (self._hit_xyz[:, 1] < y_max)
        j_values = np.int64((self._hit_xyz[valid, 0] - self._x_min) / self._pixel_size)
        i_values = np.int64((self._hit_xyz[valid, 1] - self._y_min) / self._pixel_size)
        mapping = np.zeros(self._n_rays, np.int64) - 1
        mapping[valid] = i_values * self._output_shape[1] + j_values
        return mapping

    '''
    def _iterate_multiprocessing(self):
        """
        Do a ray-tracing iteration, i.e. advance rays to next surface:  Divide work into multiprocessing units &
         collect results.

        :return: True, if no remaining rays are active at the beginning of the iteration.
        """
        active = np.where(self._active)
        n_active = active[0].size
        if n_active == 0:
            return True

        # partition active rays into separate chunks for multiprocessing pool
        partitions = [np.arange(n_active)]  # all one
        if self._n_cores > 1:
            partitions = np.array_split(partitions[0], self._n_cores)

        work_units = [{'origins': self._ray_origins[(part,), :],
                       'directions': self._ray_directions[(part,), :],
                       'mirrors': self._mirrors} for part in partitions]

        if self._n_cores == 1:
            results = [_bounce(w) for w in work_units]  # single process
        else:
            results = self._pool.map(_bounce, work_units)  # multi process

        for result, part in zip(results, partitions):
            hit_subset_indices = part[result['hit_subset_indices']]
            self._ray_origins[(part,), :] = result['origins']
            self._ray_directions[(part,), :] = result['directions']
            self._hit_xyz[(hit_subset_indices,), :] = result['hit_xyz_locations']
            self._bounce_counts[(hit_subset_indices,), :] = self._bounce_index
    '''

    def shutdown(self):
        self._shutdown = True


def make_unit_rays(shape, extent):
    """
    Make rays from the origin to the z=0 plane.
        The rays spread so that they just cover a unit square, given the aspect ratio (with padding on the sides,
        or top & bottom as required.)

    (The value of z is unimportant, since the FOV angle scales opposite Z so the image always fits)

    :param shape:  height, width of grid of rays
    :param extent:  dict describing spread of rays:
        'z':  distance to origin
        'x': horizontal spread
        'y': vertical spread
    :returns: h x w x 3 array of ray origins (all equal to the input origin)
              h x w x 3 array of ray directions (unit).
    """
    # ray start coordinates
    origins = np.tile(np.array([0.0, 0.0, extent['z']]), (shape[0], shape[1], 1))
    x_span = np.linspace(-extent['x'] / 2., extent['x'] / 2., shape[1])
    y_span = np.linspace(-extent['y'] / 2., extent['y'] / 2., shape[1])

    # ray endpoint coordinates
    x, y = np.meshgrid(x_span, y_span)
    z = x * 0.0 - extent['z']
    directions = np.dstack([x, y, z])
    magnitudes = np.linalg.norm(directions, axis=2, keepdims=True)
    unit_directions = directions / magnitudes
    return origins, unit_directions


def _intersect_all_surfaces_all_rays(surfaces, ray_origins, ray_directions):
    """
    Gets the distances  of all rays to all surfaces
    :param surfaces:  list of S Surface objects
    :param ray_origins:  N x 3 array of ray starting points
    :param ray_directions:  N x 3 array of ray unit directions
    :returns: S x N distances, all rays to all surfaces,
             List of S x N x 3 , all ray intersections for each surface
    """
    intersect_list = [surf.get_ray_intersections(ray_origins, ray_directions)
                      for surf in surfaces]

    intersections = np.stack([inter_dist[0] for inter_dist in intersect_list])
    distances = np.hstack([inter_dist[1] for inter_dist in intersect_list])
    return intersections, distances


def _bounce(mirrors, origins, directions):
    """
    Advance rays until they hit the next surface
    :param info:  dict with:
        'ray_origins':  N x 3 (x,y,z)
        'ray_directions':  N x 3 (x,y,z)  unit
        'mirrors':  MirrorPrism object
    """
    import ipdb;
    ipdb.set_trace()

    # mirrors = info['mirrors']
    # origins, directions = info['origins'], info['directions']
    surfs = mirrors.get_planes()
    n = mirrors.get_n()
    surfs.append(Plane.make_z_zero_plane())  # ground at the end

    # pairwise distance of all rays to all surfaces
    intersections, distances = _intersect_all_surfaces_all_rays(surfs, origins, directions)

    valid = distances > 0  # by convention, not >=

    distances[np.logical_not(valid)] = np.Inf
    all_invalid = np.sum(np.isinf(distances), axis=1).reshape(-1) >= n
    if np.sum(all_invalid) > 0:
        raise Exception("This many rays hit nothing:  %s (all rays should hit something!)" % (np.sum(all_invalid),))

    # what rays hit each surface
    hit_inds = np.argmin(distances, axis=1)
    hit_lists = [np.where(hit_inds == i)[0] for i in range(mirrors.get_n() + 1)]

    result = {'new_ray_origins': origins * np.NaN,  # should all be non-nan (for rays not hitting ground) by the end
              'new_ray_directions': directions * np.NaN}

    for s_ind, surf in enumerate(surfs):
        hitting_subset = hit_inds == s_ind

        n_hit = np.sum(hitting_subset)

        if s_ind == len(surfs) - 1:
            logging.info("  %i rays hit the ground." % (np.sum(hitting_subset),))
            # rays hitting target get saved
            result['ground_hit_xyz_locations'] = intersections[s_ind, hitting_subset, :]
            result['ground_hit_mask'] = hitting_subset
        else:

            logging.info("  %i rays hit mirror %i." % (n_hit, s_ind))
            if n_hit == 0:
                continue
            # rays hitting mirror get reflected
            result['new_ray_origins'][hitting_subset, :] = intersections[s_ind, hitting_subset, :]
            result['new_ray_directions'][hitting_subset, :] = surf.get_ray_reflections(origins[hitting_subset, :],
                                                                                       directions[hitting_subset, :], )

    return result


'''

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

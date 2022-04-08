import time
from threading import Thread


class RayTracer(object):
    def __init__(self, mirrors, resolution, update_callback):
        self._img_shape = resolution[1], resolution[0]
        self._mirrors = mirrors
        self._callback = update_callback
        self._render_thread = Thread(target=self._render)
        self._map = None
        self._shutdown = False

    def _render(self):
        while not self._shutdown:
            time.sleep(1)

    def start(self):
        self._render_thread.start()

    def get_current_map(self):
        return self._map

    def shutdown(self):
        self._shutdown = True


'''
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
    def from_resolution_and_fov(resolution, image_plane_z, fov_deg=45.0, square=False, **kwargs):
        """
        Initialize from desired components...
        :param resolution: (h, w)
        :param image_plane_z:  how far from origin
        :param fov:  how wide is view angle of NARROWER of two dimensions
        :param kwargs: other params for RayBundle.__init__()
        :return:
        """

        fov = np.deg2rad(fov_deg / 2.0)
        half_span = image_plane_z * np.tan(fov)
        xy_aspect = float(resolution[1]) / resolution[0]
        if square:
            x_half_span = half_span
            y_half_span = half_span
        elif (xy_aspect < 1.0):
            x_half_span = half_span
            y_half_span = x_half_span / xy_aspect
        else:
            y_half_span = half_span
            x_half_span = y_half_span * xy_aspect

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

    def get_plane_intersections(self, point, normal, no_points=False):
        """
        Find intersection of ACTIVE rays with given plane.
        :param point:  a point on the plane
        :param normal:  normal vector to plane
        :param no_points:  just calculate distance, not intersection points
        :return: N distances for N active rays,
                 N x 3 intersection points for N active rays, or None if no_points is True.
        """
        point = point.reshape(-1)
        normal = normal.reshape(-1)

        origins = self._origins[self._active].reshape(-1, 3)
        directions = self._directions[self._active].reshape(-1, 3)

        with np.errstate(divide='ignore', invalid='ignore'):
            # parallel rays go to np.inf
            dists = np.dot(point - origins, normal) / np.dot(directions, normal)

        dists = dists.reshape(-1, 1)
        points = None
        if not no_points:
            points = origins + dists * directions
        return dists, points

    def deactivate(self, inactive):
        self._active[_double_index(self._active, inactive)] = False

    def reflect(self, subset, intersections, plane_normal):
        idx = _double_index(self._active, subset)
        # new origin is just intersection point
        new_origins = intersections.reshape(-1, 3)

        # new direction has component parallel to normal reversed
        dirs = self._directions[self._active][subset]
        delta = - 2.0 * np.dot(dirs, plane_normal).reshape(-1, 1) * plane_normal.reshape(1, 3)
        new_directions = self._directions[self._active][subset] + delta
        new_directions /= np.linalg.norm(new_directions, axis=1, keepdims=True)

        # get distance traveled for this update
        distances = np.linalg.norm(self._origins[idx] - new_origins, axis=1)

        self._origins[idx] = new_origins
        self._directions[idx] = new_directions
        return distances

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


def test_ray_tracing():
    geom = NGonPrism(n=4, r=np.sqrt(2.0), height=11.323, phi=np.pi / 4.)
    mirrors = MirrorTube(prism=geom, )
    out_shape = (240, 320)
    fov_deg = 45.
    ground_z_cm = 20.0
    r = mirrors.get_view_rad()
    top_z = r / np.tan(np.deg2rad(fov_deg) / 2.0)
    print("Test init with mirrors r=%.5f, top_z=%.5f" % (r, top_z))

    rays = RayBundle.from_resolution_and_fov(resolution=out_shape,
                                             image_plane_z=top_z,
                                             fov_deg=fov_deg)

    mirrors = MirrorTube(prism=geom)

    # load image
    img = Image.from_file('test_img.jpg', flip_bgr_rgb=True, px_per_cm=(50, 50))

    # ray-trace
    img_map, dists, n_bounces, bounce_hist = mirrors.get_image_map(rays=rays,
                                                                   ground_z_cm=ground_z_cm,
                                                                   scope_top_z_cm=top_z,
                                                                   max_reflect=100,
                                                                   record=True)

    plt.imshow(dists)
    plt.colorbar()
    plt.axis('equal')
    plt.show()

    # pretty = img.interpolate(img_map, method='nearest')
    pretty = img.interpolate_integer(img_map)

    plt.imshow(pretty)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_ray_tracing()

    import matplotlib.pyplot as plt

    pic = NGonPrism.get_icon(400)
    plt.imshow(pic[:, :, ::-1])
    plt.show()
'''

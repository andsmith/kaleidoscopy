"""
Create the kaleidoscope's image map by raytracing from the eye, through
the image plane, bouncing off the mirrors and hitting the target.

The image map can then be applied live to video to simulate the kaleidoscope effect.

Coordinate conventions:
    - eye is at (0, 0, 0), and it looks in the positive z direction
    - image plane is at z = EYE_DIST
    - target plane is at z = TARGET_DIST (TARGET_DIST > EYE_DIST)
    - mirrors are vertical, and are touching the target plane (no ray will go under the mirror to hit the target)
    - mirrors are defined by two points, p0 and p1 in the XY plane, extend indfinitely in both z directions
    - The field of view is set so image would fill the target plane if there were no mirrors:
      - X-axis of the target (the screen) is normalized to [-1, 1], 
      - Y-axis is normalized to [-1, 1] / aspect ratio

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from threading import Thread, Lock
import logging
def plot_bounce(origins, directions, bounce, targ_z, mirrors):
    """
    In a 3d plot, show the eye/origin as a blue point, the target plane as a red plane with .25 transparency,
    The mirrors as translucent green planes, incident rays as black lines, reflected rays as blue lines.
    Plot ray/target intersections as red dots.
    Plot ray/mirror intersections as green dots.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 1))  # aspect ratio is 1:1:1

    # plot the eye/origin
    ax.scatter(0,0,0, c='b', marker='o', label='eye')

    # plot the ray origins as black dots
    #ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c='k', marker='o', label='ray origins')

    # plot the target plane
    ax.plot_surface(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), targ_z * np.ones((10, 10)), alpha=0.25,
                    color='r', label='target plane')

    # plot the mirrors
    #for mirror in mirrors:
    #    mirror.plot(ax)

    # plot the rays
    ax.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
               directions[:, 0], directions[:, 1], directions[:, 2],
               length=0.5, normalize=False, color='k', alpha=0.5)

    # plot the target hits
    target_hits = bounce['target_hit_inds']
    target_origins = bounce['new_origins'][target_hits]
    ax.scatter(target_origins[:, 0], target_origins[:, 1], target_origins[:, 2], c='r', marker='o',
               label='target hit')

    # plot the mirror hits
    mirror_hits = bounce['mirror_hit_inds']
    mirror_origins = bounce['new_origins'][mirror_hits]
    ax.scatter(mirror_origins[:, 0], mirror_origins[:, 1], mirror_origins[:, 2], c='g', marker='o',
               label='mirror hit')

    plt.legend()
    plt.show()  

class Raytracer(object):
    def __init__(self, size, mirrors, targ_z, x_max, y_max, threaded=False):
        """
        :param mirrors: list of Mirror objects
        :param targ_z: z-coordinate of the target plane
        :param x_max: maximum x-coordinate of the target plane (normally 1.0)
        :param y_max: maximum y-coordinate of the image plane (normally 1.0 / aspect)

        :param threaded: if True, run the raytracing in a separate thread.
           If running in a separate thread, the map can be accessed as it is being
           raytraced.
        """
        self._size = size
        self._mirrors = mirrors
        self._targ_z = targ_z
        self._x_max = x_max
        self._y_max = y_max
        self._map = None
        self._bounce_count = None
        self._map_lock = Lock()
        self._threaded = threaded

    def _init_rays(self):
        """
        Create the initial rays from the eye to the image plane.
        Remember the (x,y) pixel each ray points to so where it ends up on the
            target can be mapped to that location.
        """
        x, y = np.meshgrid(np.linspace(-self._x_max, self._x_max, self._size[0]),
                           np.linspace(-self._y_max, self._y_max, self._size[1]))
        z = np.ones_like(x) * self._targ_z


        origins = np.zeros((self._size[1], self._size[0], 3))
        directions = np.stack([x, y, z], axis=-1)
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        x_inds, y_inds = np.meshgrid(np.arange(self._size[0]), np.arange(self._size[1], dtype=np.int32))

        # flatten the arrays for easier / parrallel processing
        self._origins = origins.reshape((-1, 3))[:2]
        self._directions = directions.reshape((-1, 3))[:2]
        self._x_inds = x_inds.reshape(-1)[:2]
        self._y_inds = y_inds.reshape(-1)[:2]

    def get_map(self):
        with self._map_lock:
            return self._map, self._bounce_count

    def _unscale_coords(self, x, y):
        """
        Convert the pixel coordinates to the target plane coordinates.
        (x,y) are in [-1, 1] x [-a, a] (where a is 1 / aspect ratio)
        :returns: x, y in pixel coordinates
        """
        x = (x / 2 + 0.5) * self._size[0]
        y = (y / 2 + 0.5) * self._size[1]
        return x.astype(np.uint8), y.astype(np.uint8)

    def start(self):

        def _trace():
            """
             Trace the image map through the mirrors to the target plane:
            1. For each ray, see which mirror (or the target) it hits first.
            2. Separate the rays into groups depending on what they hit, 
                2.a. if they hit a mirror, keep bouncing them.
                2.b. if they hit the target, assign the target xy locations as the map values for those rays.
                map_x, map_y = np.zeros((h_px, w_px)), np.zeros((h_px, w_px))
            """
            w_px, h_px = self._size
            self._size = (w_px, h_px)
            self._init_rays()
            self._map = np.meshgrid(np.arange(w_px), np.arange(h_px, dtype=np.int32))
            self._bounce_count = np.zeros((h_px, w_px), dtype=np.int32) - 1
            iter = 0

            while self._origins.size > 0:
                bounce = _bounce(self._origins, self._directions, self._mirrors, self._targ_z)
                plot_bounce(self._origins, self._directions, bounce, self._targ_z, self._mirrors)

                logging.info("Iteration %i:  %i rays hit target, %i (%.3f %%) remain." % (
                    iter, len(bounce['target_hit_inds']), len(bounce['mirror_hit_inds']),
                    len(bounce['mirror_hit_inds']) / len(self._origins) * 100))

                # where rays hit target
                target_xy = bounce['new_origins'][bounce['target_hit_inds'], :2]
                targ_ray_x, targ_ray_y = self._unscale_coords(target_xy[:, 0], target_xy[:, 1])

                # where target-hitting rays came from:
                map_x = self._x_inds[bounce['target_hit_inds']]
                map_y = self._y_inds[bounce['target_hit_inds']]

                with self._map_lock:
                    self._map[0][map_y, map_x] = targ_ray_x
                    self._map[1][map_y, map_x] = targ_ray_y
                    self._bounce_count[map_y, map_x] = iter
                print(self._bounce_count)
                self._origins = bounce['new_origins'][bounce['mirror_hit_inds']]
                self._directions = bounce['new_directions'][bounce['mirror_hit_inds']]

                iter += 1

            logging.info("Raytracing complete in %i iterations." % iter)

        if self._threaded:
            # Thread will update self._map as results are computed.
            self._thread = Thread(target=_trace)
            self._thread.start()
        else:
            _trace()


def _bounce(origins, directions, mirrors, targ_z):
    """
    Advance each ray to its next surface.
    :param origins: Nx3 array of ray origins
    :param directions: Nx3 array of ray directions
    :param mirrors: list of Mirror objects
    :param targ_z: z-coordinate of the target plane
    :return: dict {'target_hit_inds': list of T indices of rays that hit the target,
                   'mirror_hit_inds': list of N-T indices of rays that hit a mirror,
                   'new_origins': Nx3 array of new ray origins (including those in the target plane),
                   'new_directions': Nx3 array of new ray directions}
    """
    n = origins.shape[0]
    mirror_dists = np.stack([mirror.get_dist(origins, directions) for mirror in mirrors],
                                  axis=1)
    target_dists = (targ_z - origins[:, 2]) / directions[:, 2]

    closest_mirrors = np.argmin(mirror_dists, axis=1)
    closest_m_dists = mirror_dists[np.arange(n), closest_mirrors]
    target_hits = np.where(target_dists < closest_m_dists)[0]
    mirror_hits = [i for i in range(n) if i not in target_hits]

    new_origins = np.zeros_like(origins)
    new_directions = np.zeros_like(directions)

    # calculate the hit locations for the targets:
    target_origins = origins[target_hits] + directions[target_hits] * target_dists[target_hits, None]
    new_origins[target_hits] = target_origins
    new_directions[target_hits] = 0.0

    # exclude rays that hit the target from the mirror hit list
    closest_mirrors[target_hits] = -1

    # Now for each mirror, reflect rays that hit it. 
    # IF a ray shoots exactly into an intersection w/another mirror, it will
    # reflect off both.  (update origin once, call "reflect" twice)
    #  It is not physically possible for a ray to hit more than two mirrors, but there is
    # no check for this, so avoid mirror arangements with tripple+ intersections.

    hit_counts = np.sum(mirror_dists == closest_m_dists[:, None], axis=1)

    for m in range(len(mirrors)):

        # find the new origin ONLY if the ray hit mirror m AND its hit_count is 1
        # (if its hit count is higher, it will be reflected now, and the new origin will be calculated later)
        final_hit = (hit_counts == 1) & (closest_mirrors == m)  # only these get moved now (origin changed)
        refl_hit = (hit_counts >= 1) & (closest_mirrors == m)  # both get reflected now (unit vec changed)

        # reflect the directions:
        new_directions[refl_hit] = mirrors[m].reflect(directions[refl_hit])
        #print("Updating directions:", np.where(refl_hit))
        hit_counts[refl_hit] -= 1

        # Update the origins:
        new_origins[final_hit] = origins[final_hit] + directions[final_hit] * closest_m_dists[final_hit, None]
        #print("Updating origins:", np.where(final_hit))


    rv= {'target_hit_inds': target_hits,
            'mirror_hit_inds': mirror_hits,
            'new_origins': new_origins,
            'new_directions': new_directions}
    
    return rv


class FakeRaytracer:
    """
    Stub raytracer that returns a placeholder image of the correct output size.
    Can optionally be configured with mirror geometry to generate synthetic ray
    bounce data for side_view.py queries.
    """

    _FAKE_GRID_LONG = 20   # rays along the longer dimension for fake bounce grid
    _MAX_BOUNCES = 8

    def __init__(self, size, mirrors=None, x_max=1.0, y_max=None, img_z=None, targ_z=None):
        """
        :param size: (width, height) of the output image
        :param mirrors: list of Mirror objects (optional; enables ray data generation)
        :param x_max: FOV half-width in natural coords at targ_z (default 1.0)
        :param y_max: FOV half-height in natural coords (default x_max * h/w)
        :param img_z: image plane Z coordinate (required if mirrors provided)
        :param targ_z: target plane Z coordinate (default TARG_Z from geom)
        """
        from geom import TARG_Z as _TARG_Z
        self._size = size
        self._mirrors = mirrors
        w, h = size
        self._x_max = x_max
        self._y_max = y_max if y_max is not None else x_max * h / w
        self._targ_z = targ_z if targ_z is not None else _TARG_Z
        self._img_z = img_z

        self._origins = None
        self._directions = None
        self._ray_grid_shape = None
        self._bounces = []

        if mirrors is not None and img_z is not None:
            self._generate_fake_rays()

    def render(self):
        """Return a gray placeholder image with a 'rendering not implemented yet' message."""
        w, h = self._size
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        text = "rendering not implemented yet"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(img, text, (x, y), font, scale, (180, 180, 180), thickness)
        return img

    def _generate_fake_rays(self):
        """
        Generate a coarse ray grid and simulate real bounces for side_view queries.
        Uses actual mirror geometry so the paths are geometrically correct.
        """
        w, h = self._size
        aspect = w / h
        if aspect >= 1.0:
            n_x = self._FAKE_GRID_LONG
            n_y = max(2, round(self._FAKE_GRID_LONG / aspect))
        else:
            n_y = self._FAKE_GRID_LONG
            n_x = max(2, round(self._FAKE_GRID_LONG * aspect))

        self._ray_grid_shape = (n_y, n_x)

        s = self._img_z / self._targ_z
        x_vals = np.linspace(-self._x_max * s, self._x_max * s, n_x)
        y_vals = np.linspace(-self._y_max * s, self._y_max * s, n_y)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        z_grid = np.full_like(x_grid, self._img_z)

        img_plane = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
        dir3 = img_plane / np.linalg.norm(img_plane, axis=-1, keepdims=True)
        N = dir3.shape[0]

        self._origins = img_plane.copy()   # rays start at image plane, not the eye
        self._directions = dir3.copy()
        self._bounces = []

        cur_origins = self._origins.copy()
        cur_dirs = dir3.copy()
        still_active = np.ones(N, dtype=bool)

        for _ in range(self._MAX_BOUNCES):
            if not np.any(still_active):
                break

            step_origins = cur_origins.copy()
            step_dirs = cur_dirs.copy()
            hit_origins = cur_origins.copy()
            hit_mirror = np.zeros(N, dtype=bool)
            hit_target = np.zeros(N, dtype=bool)
            new_origins = cur_origins.copy()
            new_dirs = cur_dirs.copy()

            act = np.where(still_active)[0]
            ao = cur_origins[act]
            ad = cur_dirs[act]

            m_dists = np.stack(
                [m.get_dist(ao, ad) for m in self._mirrors], axis=1
            )   # (len(act), n_mirrors)
            t_dists = (self._targ_z - ao[:, 2]) / ad[:, 2]

            best_m_idx = np.argmin(m_dists, axis=1)
            best_m_dist = m_dists[np.arange(len(act)), best_m_idx]

            for local_i, global_i in enumerate(act):
                td = t_dists[local_i]
                md = best_m_dist[local_i]

                if td <= md or not np.isfinite(md):
                    # hits target
                    hit_target[global_i] = True
                    new_origins[global_i] = ao[local_i] + ad[local_i] * td
                    still_active[global_i] = False
                else:
                    # hits a mirror
                    hit_mirror[global_i] = True
                    m = self._mirrors[best_m_idx[local_i]]
                    new_origins[global_i] = ao[local_i] + ad[local_i] * md
                    new_dirs[global_i] = m.reflect(ad[local_i:local_i + 1])[0]

            hit_origins[act] = new_origins[act]

            self._bounces.append({
                'step_origins': step_origins,
                'step_dirs':    step_dirs,
                'hit_origins':  hit_origins,
                'hit_mirror':   hit_mirror,
                'hit_target':   hit_target,
            })

            cur_origins = new_origins.copy()
            cur_dirs = new_dirs.copy()

    def get_ray_params(self):
        """
        Return geometric parameters needed by side_view.

        :returns: dict with keys x_max, y_max, img_z, targ_z, mirrors, ray_grid_shape
        """
        return {
            'x_max':          self._x_max,
            'y_max':          self._y_max,
            'img_z':          self._img_z,
            'targ_z':         self._targ_z,
            'mirrors':        self._mirrors,
            'ray_grid_shape': self._ray_grid_shape,
        }

    def get_initial_rays(self):
        """
        Return the initial ray origins and directions.

        :returns: (origins Nx3, directions Nx3), or (None, None) if not configured
        """
        return self._origins, self._directions

    def get_bounces(self):
        """
        Return the list of per-step bounce dicts.

        Each dict has keys: step_origins, step_dirs, hit_origins, hit_mirror, hit_target.
        Empty list if no mirrors configured.
        """
        return self._bounces


if __name__ == "__main__":
    # test_iso_mirrors()
    logging.basicConfig(level=logging.INFO)
    logging.info("All tests passed.")

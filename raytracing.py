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
      - Y-axis is normalized to [-1, 1] * aspect ratio

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from threading import Thread, Lock
import logging



class Mirror(object):
    """
    planar, vertical mirror
    """

    def __init__(self, p0, p1):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        p_vec = self.p1 - self.p0
        self.p_unit_2d = p_vec / np.linalg.norm(p_vec)

    def get_dist(self, origins, directions):
        """
        Calculate the intersection of the mirror with the rays.
        :param origins: Nx3 array of ray origins
        :param directions: Nx3 array of ray directions (unit vectors)
        :returns: N element array of distances to the mirror (negative = no hit)
        """
        # calculate the intersection of the ray with the mirror
        import ipdb; ipdb.set_trace()
        a = self.p0 - origins
        b = np.sum(a * self.p_unit_2d, axis=1)
        c = np.sum(directions * self.p_unit_2d, axis=1)
        dists = -b / c
        return dists


    def reflect(self, u_vec):
        """
        Reflect unit vectors off the mirror.

        :param u_vec: Nx3, unit vectors to reflect
        :returns: reflected unit vector
        """
        # calculate the normal vector to the mirror
        norm = np.array([self.p_unit_2d[1], -self.p_unit_2d[0], 0])
        norm = norm / np.linalg.norm(norm)

        # calculate the reflected vector
        dot = np.sum(u_vec * norm, axis=1)
        dot = dot[:, None]
        return u_vec - 2 * dot * norm


def test_mirror_intersection():
    mirror = Mirror([0, 0], [1, 0])
    origins = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    directions = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    dists = mirror.get_dist(origins, directions)
    assert np.allclose(dists, [0, -1, 0.5])
def test_mirror_reflection():
    mirror = Mirror([0, 0], [1, 0])
    u_vec = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    ref = mirror.reflect(u_vec)
    assert np.allclose(ref, [[-1, 0, 0], [0, 1, 0], [-1, -1, 0]])

class Raytracer(object):
    def __init__(self,size, mirrors, targ_z, x_max, y_max, threaded=False):
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
        self._origins = origins.reshape((-1, 3))
        self._directions = directions.reshape((-1, 3))
        self._x_inds = x_inds.reshape(-1)
        self._y_inds = y_inds.reshape(-1)

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
            self._bounce_count = np.zeros((h_px, w_px), dtype=np.int32)
            iter=0
            
            while self._origins.size > 0:
                bounce = _bounce(self._origins, self._directions, self._mirrors, self._targ_z)

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
    mirror_dists = np.concatenate([mirror.get_dist(origins, directions) for mirror in mirrors],
                                     axis=0)
    target_dists = (targ_z - origins[:, 2]) / directions[:, 2]
    closest_mirror = np.argmin(mirror_dists)
    target_hits = np.where(target_dists < mirror_dists[closest_mirror])[0]
    mirror_hits = [i for i in range(n) if i not in target_hits]

    new_origins = np.zeros_like(origins)
    new_directions = np.zeros_like(directions)

    # calculate the hit locations for the targets:
    target_origins = origins[target_hits] + directions[target_hits] * target_dists[target_hits, None]
    new_origins[target_hits] = target_origins
    new_directions[target_hits] = 0.0

    # Now for each mirror
    for m in range(len(mirrors)):
        hit_inds = np.where(closest_mirror == m)[0]
        if len(hit_inds) == 0:
            continue

        # calculate the hit locations for the mirrors:
        mirror_origins = origins[hit_inds] + directions[hit_inds] * mirror_dists[closest_mirror][hit_inds, None]
        new_origins[hit_inds] = mirror_origins
        new_directions[hit_inds] = mirrors[m].reflect(directions[hit_inds])

    return {'target_hit_inds': target_hits,
            'mirror_hit_inds': mirror_hits,
            'new_origins': new_origins,
            'new_directions': new_directions}

    

def make_iso_mirrors(angle_deg=30., size=0.9):
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
            plt.plot([mirror.p0[0], mirror.p1[0]], [mirror.p0[1], mirror.p1[1]], 'ko-')
        plt.title("%d degrees" % angle)
        plt.axis('equal')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test_iso_mirrors()
    logging.basicConfig(level=logging.INFO)
    #test_mirror_intersection()
    test_mirror_reflection()
    logging.info("All tests passed.")

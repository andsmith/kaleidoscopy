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
from geom import rotate_2d


class Mirror(object):
    """
    planar, vertical mirror, defined by 2 points in the XY plane.
    Intersections are constrained to the finite XY segment [p0, p1].
    The mirror is assumed to be perfectly reflective, and has no thickness.
    """

    def __init__(self, p0_xy, p1_xy):
        """
        :param p0_xy: (x, y) coordinates of one point on the mirror
        :param p1_xy: (x, y) coordinates of another point on the mirror (not the same as p0)
        """
        
        self.p0 = np.array([p0_xy[0], p0_xy[1], 0])
        self.p1 = np.array([p1_xy[0], p1_xy[1], 0])
        self._init_geom()

    def _init_geom(self):
        # normal, for reflecting rays:
        p_vec = self.p1 - self.p0
        self._seg_len_2d = np.linalg.norm(p_vec[:2])
        self.p_unit_2d = p_vec[:2] / np.linalg.norm(p_vec[:2])
        self._normal_3d = np.cross(np.array([self.p_unit_2d[0], self.p_unit_2d[1], 0]),
                                   np.array([0, 0, 1]))

        # Coefficients for calculating intersections with rays:
        self._C = -(self.p0[1]-self.p1[1])*self.p0[0] - (self.p1[0]-self.p0[0])*self.p0[1]
        self._Cx = self.p0[1] - self.p1[1]
        self._Cy = self.p1[0] - self.p0[0]
        
    def get_dist(self, origins, directions):
        """
        Calculate the intersection of the mirror with the rays.
        Intersections are only valid when they hit the finite XY segment [p0, p1].

        Since Mirror objects are vertical planes, let the two points be:
          P0 = (P0x, P0y, 0) and P1 = (P1x, P1y, 0)

        Then the intersection distance with ray R  with position (Rx, Ry, Rz) and direction
        unit vector U = (Ux, Uy, Uz) is given by:

                Cx * Rx + Cy * Ry + C
        d = - ---------------------------   (note the negative sign)
                  Cx * Ux + Cy * Uy

        for the constants nRx, nRy, dUx, dUy, and C calculated from the mirror points.


        :param origins: Nx3 array of ray origins
        :param directions: Nx3 array of ray directions (unit vectors)
        :returns: N element array of distances to the mirror (np.inf = no hit)
        """
        Rx = origins[:, 0]
        Ry = origins[:, 1]
        Ux = directions[:, 0]
        Uy = directions[:, 1]

        num = -(self._Cx * Rx + self._Cy * Ry + self._C) 
        denom = (self._Cx * Ux + self._Cy * Uy)

        # if the denominator is zero, the ray is parallel to the mirror, set 
        # the distance manually to Inf instead of dividing:
        parallel = np.isclose(denom, 0)
        hits = np.logical_not(parallel)
        d = np.zeros_like(num)
        d[parallel] = np.inf
        d[hits] = num[hits] / denom[hits]

        # Mask out negative distances (rays going away from the mirror)
        d[d < 0] = np.inf

        # Restrict hits to the finite segment endpoints in XY.
        # Project intersection points onto the segment axis and require
        # 0 <= projection_length <= segment_length.
        ix = Rx + d * Ux
        iy = Ry + d * Uy
        rel_x = ix - self.p0[0]
        rel_y = iy - self.p0[1]
        proj_len = rel_x * self.p_unit_2d[0] + rel_y * self.p_unit_2d[1]
        eps = 1e-9
        on_segment = (proj_len >= -eps) & (proj_len <= self._seg_len_2d + eps)
        d[~on_segment] = np.inf

        return d
    
    def reflect(self, u_vec):
        """
        Reflect unit vectors off the mirror.

        Find the component of u_vec perpendicular to the mirror and reverse it.
       
        :param u_vec: Nx3, unit vectors to reflect
        :returns: reflected unit vector
        """
        u_vec = np.array(u_vec)
        u_vec = u_vec / np.linalg.norm(u_vec, axis=-1, keepdims=True)
        u_dot_n = np.sum(u_vec * self._normal_3d, axis=-1, keepdims=True)
        return u_vec - 2 * u_dot_n * self._normal_3d

def test_mirror_intersection(plot=False):
    """
    Test the mirror intersection function by comparing to the law of sines for a simple case where the mirror is at a 45 degree angle and the rays are in the XY plane.
    """
    
    mirror = Mirror([2, 0], [0, 2])
    mirror_angle = np.pi/4
    origin = np.zeros(3)
    directions = np.array([[1, 1, 0], [2, 1, 0], [1, 2, 0]],dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    upper_angles = np.pi - (angles + mirror_angle)  
    origins = np.tile(origin, (directions.shape[0], 1))


    # law of sines to get true distances:
    true_dists = 2 * np.sin(mirror_angle)/np.sin(upper_angles)  # will break if directions have z component nonzero...

    dists = mirror.get_dist(origins, directions)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot([mirror.p0[0], mirror.p1[0]], [mirror.p0[1], mirror.p1[1]], 'ko-', label='mirror')
        ax.plot(origins[:, 0], origins[:, 1], 'ro', label='origins')
        for d in range(len(directions)):
            ax.plot([origins[d, 0], origins[d, 0] + directions[d, 0]],
                    [origins[d, 1], origins[d, 1] + directions[d, 1], ], 'b-', label='direction %i' % d)
        ax.legend()
        ax.axis('equal')
        
    assert np.allclose(dists, true_dists)


def test_mirror_reflection(plot=False):
    
    mirror = Mirror([2, 0], [0, 2])
    mirror_angle = np.pi/4
    origin = np.zeros(3)
    directions = np.array([[1, 1, 0], [2, 1.2, 0], [1, 2, 0]],dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins= np.tile(origin, (directions.shape[0], 1))

    distances = mirror.get_dist(origins, directions)
    
    ref_dirs = mirror.reflect(directions)

    if plot:
        
        intersections = origins + directions * distances[:, None]
        fig, ax = plt.subplots()
        ax.plot([mirror.p0[0], mirror.p1[0]], [mirror.p0[1], mirror.p1[1]], 'ko-', label='mirror')
        ax.plot(origins[:, 0], origins[:, 1], 'ro', label='origins')
        for d in range(len(directions)):
            ax.plot([origins[d, 0], intersections[d, 0]],
                    [origins[d, 1], intersections[d, 1]], 'b:', label='direction %i' % d)
            ax.plot([origins[d, 0], origins[d, 0] + directions[d, 0]],
                    [origins[d, 1], origins[d, 1] + directions[d, 1], ], 'b-', label='direction %i' % d)
            ax.plot([intersections[d, 0], intersections[d, 0] + ref_dirs[d, 0]],
                    [intersections[d, 1], intersections[d, 1] + ref_dirs[d, 1]], 'g-', label='reflected %i' % d)
        ax.legend()
        ax.axis('equal')
        
        

    # check dot product of angles w/reflections is sine of twice the mirror angle
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    ref_angles = np.arctan2(ref_dirs[:, 1], ref_dirs[:, 0])
    

    dots = np.sum(directions * ref_dirs, axis=-1)
    sines = -np.sin(2 * angles)

    print(dots)
    print(sines)
    assert np.allclose(dots, sines)


def make_iso_mirrors(angle_deg=30., size=0.9, rotation_rad=0):
    """
    Make a set of mirrors in an isosceles triangle pointing up, centered around the origin.
    :param angle_deg: angle of the unique angle in the triangle
    :param size: size of the triangle (maximum distance from the center to a corner)
    :param rotation_rad: rotation of the triangle in radians (counterclockwise)
    """
    p0 = np.array([-1, 0])
    p1 = np.array([1, 0])
    p2 = np.array([0, 1/np.tan(np.radians(angle_deg/2))])

    points = np.array([p0, p1, p2])
    points = points - np.mean(points, axis=0)  # center around the origin
    r = np.max(np.linalg.norm(points, axis=1))  # maximum distance from the origin
    points = points / r * size  # scale to the desired size
    if rotation_rad != 0:
        points = rotate_2d(points, rotation_rad)
    mirrors = [Mirror(points[i], points[(i+1) % 3]) for i in range(3)]
    return mirrors

def make_mirror_box(radius=0.9, rotation_rad=0):
    points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * radius
    if rotation_rad != 0:
        points = rotate_2d(points, rotation_rad)
    mirrors = [Mirror(points[i], points[(i+1) % 4]) for i in range(4)]
    return mirrors


def show_iso_mirrors():
    # plot the mirrors for a 15, 30, 45, 60, 90, and 120 degree triangle
    angles = [15, 30, 45, 60, 90, 120]
    n_plot = 1


    def _plot_mirrors(ax, m_list, title):
        for mirror in m_list:
            ax.plot([mirror.p0[0], mirror.p1[0]], [mirror.p0[1], mirror.p1[1]], 'ko-')
        ax.set_title(title)
        ax.axis('equal')
        ax.axis('off')

    fig, ax = plt.subplots(2, 4)
    fig.suptitle("Mirror configurations")
    ax = ax.flatten()
    for i, angle in enumerate(angles):
        mirrors = make_iso_mirrors(angle)
        title="%d degrees" % angle
        _plot_mirrors(ax[i], mirrors, title)
    mbox1 = make_mirror_box(.9)
    _plot_mirrors(ax[-1], mbox1, "Mirror box %.2f" % .9)
    ax[-1].set_xlim(-1.0, 1.0)
    ax[-1].set_ylim(-1.0, 1.0)
    mbox2 = make_mirror_box(.4)
    _plot_mirrors(ax[-2], mbox2, "Mirror box %.2f" % .4)
    ax[-2].set_xlim(-1.0, 1.0)
    ax[-2].set_ylim(-1.0, 1.0)
    
    plt.tight_layout()
    #plt.show()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    show_iso_mirrors()
    test_mirror_intersection(plot=True)
    test_mirror_reflection(plot=True)
    plt.show()
    logging.info("All tests passed.")

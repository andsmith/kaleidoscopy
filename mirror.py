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
        self._init_geom()

    def _init_geom(self):
        # normal, for reflecting rays:
        p_vec = self.p1 - self.p0
        self.p_unit_2d = p_vec / np.linalg.norm(p_vec)
        self._normal_3d = np.cross(self.p_unit_2d, [0, 0, 1])

        # Coefficients for calculating intersections with rays:
        self._C = -(self.p0[1]-self.p1[1])*self.p0[0] - (self.p1[0]-self.p0[0])*self.p0[1]
        self._Cx = self.p0[1] - self.p1[1]
        self._Cy = self.p1[0] - self.p0[0]
        
    def get_dist(self, origins, directions):
        """
        Calculate the intersection of the mirror with the rays.
        Do not consider mirror endpoints (i.e. extend mirrors indefinitely).

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
        :returns: N element array of distances to the mirror (negative = no hit)
        """
        Rx = origins[:, 0]
        Ry = origins[:, 1]
        Rz = origins[:, 2]
        Ux = directions[:, 0]
        Uy = directions[:, 1]
        Uz = directions[:, 2]

        d = -(self._Cx * Rx + self._Cy * Ry + self._C) / (self._Cx * Ux + self._Cy * Uy)
        # d[Uz == 0] = -1

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


def show_iso_mirrors():
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
    #plt.show()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    show_iso_mirrors()
    test_mirror_intersection(plot=True)
    test_mirror_reflection(plot=True)
    plt.show()
    logging.info("All tests passed.")

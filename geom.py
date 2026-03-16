import numpy as np
import logging 

# Distance from viewer's eye to the target image plane:
TARG_Z = 1.0

# Output grid of pixels in the image plane must have its closest pixel
# at least this far from all mirrors:
CLOSEST_PIXEL_PX = 3.0

# Closest mirrors can be to optical axis in natural coords (Euclidean XY distance):
CLOSEST_MIRROR = 0.05
import numpy as np

COLORS = {'black': (0, 0, 0),
          'white': (255, 255, 255), 
          'red': (0, 0, 255), 
          'green': (0, 255, 0), 
          'blue': (255, 0, 0), 
          'cyan': (255, 255, 0),
          'yellow': (0, 255, 255), 
          'gray': (128, 128, 128), 
          'orange': (0, 165, 255),
          'light_blue': (255, 200, 100),
          'neon_green': (57, 255, 20),
          'magenta': (255, 0, 255),
          'dark_navy': (33, 0, 0)}

def rotate_2d(points, angle_rad):
    """
    Rotate a set of 2D points by a given angle in radians.
    :param points: numpy array of shape (N, 2) representing N 2D points.
    :param angle_rad: rotation angle in radians.
    :return: numpy array of shape (N, 2) representing the rotated points.
    """
    rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
    return points @ rot_mat.T

def lineseg_dist(p, a, b):
    """
    from https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line
    Calculates the shortest distance from point p to line segment [a, b].
    p, a, and b should be numpy arrays representing 2D or 3D points.
    """
    # Normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))
    
    # Signed parallel distance components (project p-a onto d and p-b onto d)
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    
    # Clamped parallel distance (0 if projection is between a and b, else endpoint distance is used)
    h = np.maximum.reduce([s, t, 0])
    
    # Perpendicular distance component
    c = np.cross(p - a, d)
    
    return np.hypot(h, np.linalg.norm(c))

def test_seg_pt_dist():
    # Example Usage (2D):
    p = np.array([0, 0])          # The point
    a = np.array([1, 1])          # Segment endpoint A
    b = np.array([2, 2])          # Segment endpoint B

    distance = lineseg_dist(p, a, b)
    print(f"Distance from point to segment: {distance}")

    # Example Usage (3D):
    p_3d = np.array([0, 0, 0])
    a_3d = np.array([1, 1, 1])
    b_3d = np.array([2, 2, 2])

    distance_3d = lineseg_dist(p_3d, a_3d, b_3d)
    print(f"Distance from point to segment (3D): {distance_3d}")


def make_test_check(size,sq_size=16, n_colors=2, randomize=False):
    """
    Make a test checkered pattern image of the given size.
    """
    c_list = [COLORS['white'], COLORS['black'], COLORS['red'], COLORS['green'], COLORS['blue'], COLORS['cyan'], COLORS['magenta'], COLORS['yellow']]
    if n_colors > len(c_list):
        logging.warning(f"Requested {n_colors} colors, but only {len(c_list)} are available. Using {len(c_list)} colors.")
        n_colors = len(c_list)
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(0, size[1], sq_size):
        for x in range(0, size[0], sq_size):
            if not randomize:
                if (x // sq_size + y // sq_size) % n_colors == 0:
                    img[y:y+sq_size, x:x+sq_size] = c_list[(x // sq_size + y // sq_size) % n_colors]
                else:
                    img[y:y+sq_size, x:x+sq_size] = c_list[(x // sq_size + y // sq_size) % n_colors]
            else:
                img[y:y+sq_size, x:x+sq_size] = c_list[np.random.randint(0, n_colors)]
    return img



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_seg_pt_dist()
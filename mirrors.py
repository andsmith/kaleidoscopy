"""
Model of mirror geometry.  Mirrors are perpendicular to image plane and form a convex polygon when viewed
 from the top  (i.e. are arranged in right prisms).
"""

from abc import ABC, abstractmethod
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import matplotlib.pyplot as plt
import logging
# from threading import Lock
import cv2
from scipy.optimize import minimize
from pynput import keyboard
import faulthandler  # div/0

from gui_utils.mouse import MouseKeyboardState, ButtonStates, MouseButtons
from gui_utils.text_annotation import get_best_font_scale
from surfaces import Plane
from layout import LAYOUT


def mouse_motion_to_ui_input(pos1, pos2):
    vdistance = (pos2[1] - pos1[1]) / LAYOUT['mouse_controls']['v_adjust_divisor']
    hdistance = (pos2[1] - pos1[1]) / LAYOUT['mouse_controls']['h_adjust_divisor']
    return hdistance, vdistance


def clamped_adjust(value, increment, bounds):
    """
    Adjust a parameter within an interval.
    :param value:  numeric value to adjust
    :param increment:  value to add
    :param bounds:  pair [lower, higher]
    :returns: value + increment  unless out of bounds, then the relevant bound is returned
    """
    value += increment
    if value < bounds[0]:
        return bounds[0]
    elif value > bounds[1]:
        return bounds[1]
    return value


class MirrorPrism(ABC):
    """
    Abstract class to handle geometry.
    """

    SHAPING_INSTRUCTIONS = ["Shape Mirror Geometry:  Hit SPACE-key when done...",
                            " SHIFT + Left-click + Drag up-and-down:  FOV"]
    FOV_BOUNDS = [np.deg2rad(30.0),  # shape should fill unit square at this FOV
                  np.deg2rad(179.0)]  # not sure what this will do

    def __init__(self):
        self._fov_rad = None
        self._shaping_finish_event = None
        self._shaping_window_name = None
        self._mouse_state = None
        self._mouse_pos_orig = None
        self._mask = None  # mask to keep only portion of image inside mirrors, for video during shaping
        self._mouse_state_mgr = MouseKeyboardState()  # for listening to shift (etc.) keys

    @abstractmethod
    def get_unit_shape(self, **kwargs):
        """
        Get shape of mirrors, scaled to fit into unit square, given current FOV and pitch/yaw,
        and custom params.
        """
        pass

    def get_surfaces(self):
        """
        Get Surface() objects (corresponding to all mirrors) from current params

        :returns:  Plane() objects from vertices of mirrors.
        """
        surfaces = []
        vertices = self.get_unit_shape()
        n_planes = len(vertices)
        vertices = np.array(vertices + [vertices[-1]])
        for i in range(n_planes):
            p1 = vertices[i, :]
            p2 = vertices[i + 1, :]
            xy_intersection = (p1 + p2) / 2.0  # midpoint

            # for normal, use cross-product between sides of rectangle (top/bottom don't matter)
            top_p1 = np.array([p1[0], p1[1], 0.0])
            top_p2 = np.array([p2[0], p2[1], 0.0])
            bottom_p1 = np.array([p1[0], p1[1], 1.0])
            norm_vec = np.cross(top_p2 - top_p1, bottom_p1 - top_p1)
            norm_vec /= np.linalg.norm(norm_vec)
            origin = np.zeros((1, 2), dtype=np.float64)
            ray_to_origin = origin - xy_intersection
            if np.dot(norm_vec, ray_to_origin) < 0:  # pointed away from origin
                norm_vec = -norm_vec
            surfaces.append(Plane(xyz_intersect=(xy_intersection[0], xy_intersection[1], 0.0),
                                  normal=norm_vec))
        return surfaces

    def start_shaping(self, window_name, finished_event):
        """
        User begins to set shape of mirror arrangement.
        """
        self._shaping_window_name = window_name
        self._shaping_finish_event = finished_event
        print("Start shaping:", self._shaping_finish_event, id(self))
        return self.SHAPING_INSTRUCTIONS, self.handle_mouse_adjust

    def _done_shaping(self):
        print("Stopping shaping:", self._shaping_finish_event, id(self))
        self._shaping_finish_event.set()
        logging.info("Finished event set...")
        self._shaping_finish_event = None
        cv2.setMouseCallback(self._shaping_window_name, lambda *args: None)
        # self._set_geometry()

    def handle_mouse_adjust(self, *args, **kwargs):
        """
        CV2 callback for mouse events
          if "SHIFT" is down,  adjusts the aperture size,
          else  the other params are adjusted by the sub-class.
        """
        new_mk_state = self._mouse_state_mgr.update_state(*args, **kwargs)

        if new_mk_state['mouse_buttons'][MouseButtons.LEFT] is not None and \
                new_mk_state['mod_keys']['shift']:
            # adjusting FOV
            if self._mouse_pos_orig is None:  # first time
                self._mouse_pos_orig = new_mk_state['mouse_position']
            else:
                _, adjustment = mouse_motion_to_ui_input(self._mouse_pos_orig, new_mk_state['mouse_position'])
                self._fov_rad = clamped_adjust(self._fov_rad, adjustment, MirrorPrism.FOV_BOUNDS)
                logging.info("Adjusting FOV to %.2f deg." % (np.rad2deg(self._fov_rad),))

        if new_mk_state['button_change'] == 'l-up':
            self._mouse_pos_orig = None

        if not new_mk_state['mod_keys']['shift']:
            # shape sub-classes
            self._mouse_adjust(new_mk_state['mouse_position'],
                               new_mk_state['motion'],
                               new_mk_state['button_change'],
                               new_mk_state['mouse_buttons'],
                               new_mk_state['mod_keys'])

    @abstractmethod
    def _mouse_adjust(self, pos, d_pos, d_button, button_state, keyboard_state):
        """
        Adjust shape-specific mirror params using mouse
        """
        pass

    def handle_keyboard_adjust(self, k):
        """
        Used for shaping mirror params, overridable, but needs to call _done_shaping when the right key is hit, etc.
        """
        k = k & 0xFF
        if k == 0:
            print("ENTER")
        if k == ord(' '):
            logging.info("Done shaping.")
            self._done_shaping()

    def _make_mask(self, res):
        """
        To show the shape of the mirrors while adjusting it, a video is shown with the portions exterior to the
        mirrors blacked out by this mask.

        :param res:  H x W of mask.
        :returns: H x W boolean mask, y-axis is scaled to 1.0, x-axis scaled by aspect ratio
        """
        points = self.get_unit_shape()
        points_scaled = points * res[0]
        x_padding = (res[1] - res[0]) / 2.
        points_scaled[:, 0] += x_padding
        mask = np.zeros(res[::-1], dtype=np.uint8)
        coords =np.int32(points)
        cv2.fillPoly(mask, [coords], 1, cv2.LINE_8)
        return mask[::-1, :]

    def get_masked_image(self, img):
        """
        Generate a bitmap
        """
        if self._mask is None:
            self._mask = self._make_mask(img.shape[:2])

        masked = img * np.expand_dims(self._mask, 2)
        return masked

    '''
    def _set_corners(self, corners, height, inscribed_radius=None):
        """
        NOTE:  Corners are shifted so inscribed circle center has x,y=0,0, if inscribed_radius is None.

        Init with list of 2d coordinates (i.e. closed polygon loop), representing view from the top.
        :param corners:  Nx2 array, or N-element list of (x, y) pairs, clockwise oriened corners of a N-sided polygon.
        :param height: height of prism sides
        :param inscribed_circle:  ((x, y), r):  must fit inside corners, (not checked),
            calculated if None, breaks for nonconvex

        """
        self._height = height
        bottom = height
        top = 0
        if not isinstance(corners, np.ndarray):
            corners = np.array(corners)
        self._n = corners.shape[0]

        if inscribed_radius is None:
            center, inscribed_radius = max_inscribed_circle(corners)
            corners -= np.array(center).reshape(1, 2)
        self._rad = inscribed_radius

        self._top_left = np.hstack((corners, np.ones(self._n).reshape(-1, 1) * top))
        self._bottom_left = np.hstack((corners, np.ones(self._n).reshape(-1, 1) * bottom))

        # cycle & wrap
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

    def get_inscribed_rad(self):
        return self._rad

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
    '''

    @classmethod
    @abstractmethod
    def get_name(cls):
        pass

    @classmethod
    @abstractmethod
    def get_icon_vertices(cls):
        pass

    @classmethod
    def get_icon(cls, size):
        shape = cls.get_icon_vertices()
        name = cls.get_name()
        return cls._draw_icon(size, shape, name)

    @staticmethod
    def _draw_icon(size, shape, name):
        icon_colors = LAYOUT['icons']['colors']
        fonts = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]
        font_name = fonts[0]
        thickness = int(size / 200) if size > 200 else 1
        font_thickness = thickness

        img = np.zeros((size, size, 3)).astype(np.uint8)
        img[:, :, :] = icon_colors['background']
        point_start = tuple(np.int64(size * np.array(shape['points'][0])))

        for i in range(1, len(shape['points'])):
            point = tuple(np.int64(size * np.array(shape['points'][i])))

            if not shape['final_line_dashed'] or i < len(shape['points']) - 1:
                cv2.line(img, point_start, point, icon_colors['foreground'],
                         thickness, cv2.LINE_AA)
            else:
                draw_dashed_line(img, point_start, point, num=4,
                                 color=icon_colors['foreground'],
                                 thickness=thickness, linetype=cv2.LINE_AA)
            point_start = point

        text_pos = size * np.array(shape['text_center_bottom'])
        text_x_range = tuple(np.int64(np.array([0.1, 0.90]) * size))
        font_scale = get_best_font_scale(name, font_name, font_thickness, text_x_range[1] - text_x_range[0])

        (text_width, text_height), _ = cv2.getTextSize(name, font_name, font_scale, font_thickness)

        text_anchor = (int(text_pos[0] - text_width / 2.),
                       int(text_pos[1]))

        img = cv2.putText(img, name, text_anchor, font_name, font_scale,
                          icon_colors['foreground'], font_thickness)
        img[text_anchor[1], text_anchor[0], :] = [0, 255, 0]

        return img

'''
def draw_dashed_line(img, start, end, num, color, thickness, linetype=cv2.LINE_AA):
    x = np.linspace(start[0], end[0], num * 2).astype(np.int64)
    y = np.linspace(start[1], end[1], num * 2).astype(np.int64)
    for seg in range(0, num):
        p_start = (x[seg * 2], y[seg * 2])
        p_end = (x[seg * 2 + 1], y[seg * 2 + 1])
        cv2.line(img, p_start, p_end, color, thickness, linetype)


def _test_draw_dashed_line():
    img = np.zeros((100, 100, 3)).astype(np.uint8)
    draw_dashed_line(img, (10, 10), (90, 80), 5, (255, 255, 255), thickness=2)
    plt.imshow(img)
    plt.show()


def max_inscribed_circle(corners):
    """
    Approximate max (x, y, r) such that all points inside circle are within polygon.
    Not valid for convex polygons (?)
    Algorithm:
        Sample points on polygon (N per side)
        Optimize over x,y to maximize the minimum distance distance to sample points
    :param corners: N x 2, list of x, y, corners, counterclockwise
    :return: (x,y), r of max inscribed circle
    """
    n_sample_points_per_line = 50
    samples = []
    interp = np.linspace(0.0, 1.0, n_sample_points_per_line).reshape(1, 1, -1)
    corners_shifted = np.expand_dims(np.vstack([corners[1:, :], corners[0, :]]), 2)

    for i in range(corners.shape[0]):
        corner_samples = interp * np.expand_dims(corners, 2) + (1.0 - interp) * corners_shifted
        samples.extend([corner_samples[:, :, i] for i in range(n_sample_points_per_line)])
    samples = np.vstack(samples)

    def error_fn(xy):
        margin = 0.001
        dists = np.linalg.norm(samples - xy.reshape(1, -1), axis=1)
        err = -np.min(dists)
        return err + margin

    bbox = np.vstack([np.min(corners, axis=0),
                      np.max(corners, axis=0)])
    bbox = [bbox[:, 0].tolist(), bbox[:, 1].tolist()]
    x_init = np.mean(corners, axis=0)
    # solution = minimize(error_fn, x_init, method='Nelder-Mead')
    solution = minimize(error_fn, x_init, method='Powell', bounds=bbox)
    pos = solution.x
    r = -error_fn(pos)
    return pos, r


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

'''
if __name__ == "__main__":
    faulthandler.enable()
    logging.basicConfig(level=logging.INFO)
    # test_ray_tracing()

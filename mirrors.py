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
from gui_utils.mouse import MouseKeyboardState, ButtonStates, MouseButtons
from gui_utils.text_annotation import get_best_font_scale
import faulthandler  # div/0
from surfaces import Plane


class MirrorPrism(ABC):
    """
    Abstract class to handle geometry.
    """
    ICON_COLORS = {'background': (254, 250, 245),
                   'foreground': (10, 10, 15)}

    ICON_LAYOUT = {'fig_top_y': .1,
                   'fig_bottom_y': .70,
                   'text_bottom_y': 0.90}

    SHAPING_INSTRUCTIONS = ["Shape Mirror Geometry:  Hit SPACE-key when done...",
                            " SHIFT + Left-click + Drag up-and-down:  Aperture size"]
    _MIN_APERTURE_SCALE = 0.001

    def __init__(self):
        self._aperture_scale = 0.95
        self._image_shape = None
        self._mask = None
        self._shaping_finish_event = None
        self._shaping_window_name = None
        self._mouse_state = MouseKeyboardState()
        self._mouse_pos_orig = None
        self._base_aperture_scale = None

    @abstractmethod
    def get_rel_shape(self, **kwargs):
        """
        Get shape of mirrors, scaled to fit into unit square.
        """
        pass

    def get_surfaces(self):
        """
        Get Surface() objects (corresponding to all mirrors) from current params

        Default behav. is to transform vertices from get_rel_shape() into Plane() objects.
        """
        origin = np.zeros((1, 2), dtype=np.float64)
        surfaces = []
        vertices = self.get_rel_shape()
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
            norm_vec = np.cross(top_p2-top_p1, bottom_p1 - top_p1)
            norm_vec /= np.linalg.norm(norm_vec)
            ray_to_origin = origin - xy_intersection
            if np.dot(norm_vec, ray_to_origin) <0:  # pointed away from origin
                norm_vec =-norm_vec
            surfaces.append(Plane(xyz_intersect=(xy_intersection[0], xy_intersection[1], 0.0),
                                  normal = norm_vec))
        return surfaces


    def start_shaping(self, window_name, finished_event):
        """
        User sets shape of mirror arrangement.
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
        mouse_keyboard_state = self._mouse_state.update_state(*args, **kwargs)

        if mouse_keyboard_state['mouse_buttons'][MouseButtons.LEFT] is not None and \
                mouse_keyboard_state['mod_keys']['shift']:
            if self._mouse_pos_orig is None:  # first time
                self._mouse_pos_orig = mouse_keyboard_state['mouse_position']
                self._base_aperture_scale = self._aperture_scale
            else:
                distance = (self._mouse_pos_orig[1] - mouse_keyboard_state['mouse_position'][1]) / 400.0
                self._aperture_scale = self._base_aperture_scale + distance
                if self._aperture_scale < self._MIN_APERTURE_SCALE:
                    self._aperture_scale = self._MIN_APERTURE_SCALE
                if self._aperture_scale > 1.0:
                    self._aperture_scale = 1.0
                self._mask = None

        if mouse_keyboard_state['button_change'] == 'l-up':
            self._base_aperture_scale = None
            self._mouse_pos_orig = None

        if not mouse_keyboard_state['mod_keys']['shift']:
            # shape sub-classes
            self._mouse_adjust(mouse_keyboard_state['mouse_position'],
                               mouse_keyboard_state['motion'],
                               mouse_keyboard_state['button_change'],
                               mouse_keyboard_state['mouse_buttons'],
                               mouse_keyboard_state['mod_keys'])

    @abstractmethod
    def _mouse_adjust(self, pos, d_pos, d_button, button_state, keyboard_state):
        """
        Adjust shape-specific mirror params using mouse
        NOTE:  right-mouse click is used by this parent class.
        """
        pass

    def get_inscribed_rectangle(self, xy_resolution, margin=0.05):
        """
        Get largest rectangle fitting inside shape, leaving margin space.
        :param xy_resolution:  width,height of rectangle
        :returns x half-width, y-half width of rectangle
        """
        if xy_resolution[1] > xy_resolution[0]:
            raise Exception("Portrait aspect ratios not implemented.")
        verts = self.get_rel_shape()
        xy_aspect = float(xy_resolution[0]) / float(xy_resolution[1])
        img_scale = 1.0 / float(xy_resolution[1])
        half_box_scale = img_scale * 2  # start 4 pixels wide
        img = np.zeros((xy_resolution[1], xy_resolution[0], 3), dtype=np.uint8) + 1
        masked = self.get_masked_image(img)
        half_box_size = np.array([half_box_scale / xy_aspect, half_box_scale])
        img_center = (np.array(xy_resolution) / 2.0).astype(np.int64)
        final_half_box_size = None
        while half_box_size[0] < xy_resolution[0] / 2 and half_box_size[1] < xy_resolution[1] / 2:
            half_box_scale += img_scale
            half_box_size = np.array([half_box_scale / xy_aspect, half_box_scale])
            half_box_px = half_box_size / img_scale

            b = np.int64(half_box_px)
            print(half_box_size)
            print(b)
            edges = [masked[img_center[1] - b[1]: img_center[1] + b[1], img_center[0] - b[0], 0].reshape(-1),
                     masked[img_center[1] - b[1]: img_center[1] + b[1], img_center[0] + b[0], 0].reshape(-1),
                     masked[img_center[1] - b[1], img_center[0] - b[0]: img_center[0] + b[0], 0].reshape(-1),
                     masked[img_center[1] + b[1], img_center[0] - b[0]: img_center[0] + b[0], 0].reshape(-1), ]
            if np.sum(np.hstack(edges)) > 0:
                half_box_scale -= img_scale
                half_box_scale *= (1.0 - margin)
                final_half_box_size = np.array([half_box_scale / xy_aspect, half_box_scale])
                break
        if final_half_box_size is None:
            raise Exception("Couldn't fit rectangle in shape:  %s" % (verts,))
        return final_half_box_size

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
        :param res:  (w, h) of image  (i.e. reverse from numpy)
        """
        points = self.get_rel_shape()
        side_length = np.min(res[:2])

        x_offset = np.max((0, res[0] - side_length)) / 2
        y_offset = np.max((0, res[1] - side_length)) / 2
        mask = np.zeros(res[::-1], dtype=np.uint8)
        x = points[:, 0] * side_length + x_offset
        y = points[:, 1] * side_length + y_offset
        coords = np.array([(int(xc), int(yc)) for xc, yc in zip(x, y)])
        cv2.fillPoly(mask, [coords], 1, cv2.LINE_8)
        return mask[::-1, :]

    def get_masked_image(self, img):
        """
        Generate a bitmap
        """
        if self._image_shape is None or (img.shape != self._image_shape) or self._mask is None:
            # don't make a new one unnecessarily
            self._image_shape = img.shape
            self._mask = self._make_mask((self._image_shape[1], self._image_shape[0]))

        maked = img * np.expand_dims(self._mask, 2)
        return maked

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
        fonts = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]
        font_name = fonts[0]
        thickness = int(size / 200) if size > 200 else 1
        font_thickness = thickness

        img = np.zeros((size, size, 3)).astype(np.uint8)
        img[:, :, :] = MirrorPrism.ICON_COLORS['background']
        point_start = tuple(np.int64(size * np.array(shape['points'][0])))

        for i in range(1, len(shape['points'])):
            point = tuple(np.int64(size * np.array(shape['points'][i])))

            if not shape['final_line_dashed'] or i < len(shape['points']) - 1:
                cv2.line(img, point_start, point, MirrorPrism.ICON_COLORS['foreground'],
                         thickness, cv2.LINE_AA)
            else:
                draw_dashed_line(img, point_start, point, num=4,
                                 color=MirrorPrism.ICON_COLORS['foreground'],
                                 thickness=thickness, linetype=cv2.LINE_AA)
            point_start = point

        text_pos = size * np.array(shape['text_center_bottom'])
        text_x_range = tuple(np.int64(np.array([0.1, 0.90]) * size))
        font_scale = get_best_font_scale(name, font_name, font_thickness, text_x_range[1] - text_x_range[0])

        (text_width, text_height), _ = cv2.getTextSize(name, font_name, font_scale, font_thickness)

        text_anchor = (int(text_pos[0] - text_width / 2.),
                       int(text_pos[1]))

        img = cv2.putText(img, name, text_anchor, font_name, font_scale,
                          MirrorPrism.ICON_COLORS['foreground'], font_thickness)
        img[text_anchor[1], text_anchor[0], :] = [0, 255, 0]

        return img


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


if __name__ == "__main__":
    import ipdb;

    ipdb.set_trace()
    faulthandler.enable()
    logging.basicConfig(level=logging.INFO)
    # test_ray_tracing()

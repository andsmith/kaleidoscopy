import numpy as np
from mirrors import MirrorPrism
from mirror_utils import transform_points
from surfaces import Cylinder
from layout import LAYOUT

N_SAMPLES = 1000  # for drawing mask during shaping


class CirclePrism(MirrorPrism):
    def get_unscaled_shape(self, **kwargs):
        self._r = 0.5  # constant, redundant w/aperture scale
        center = np.array((0.5, 0.5)).reshape(1, 2)
        t = np.linspace(0.0, np.pi * 2.0, N_SAMPLES)
        points = np.stack([np.cos(t) * self._r,
                           np.sin(t) * self._r], axis=1) + center
        print(points.shape)

        return points

    def __init__(self):
        """
        """
        self._n = np.inf
        super(CirclePrism, self).__init__()

    SHAPING_INSTRUCTIONS = MirrorPrism.SHAPING_INSTRUCTIONS

    def _mouse_adjust(self, pos, d_pos, d_button, button_state, keyboard_state):
        pass  # no params!

    def get_surfaces(self):
        """
        Get Surface() objects (corresponding to all mirrors) from current params
        """
        return [Cylinder(np.array([0.0, 0.0]), self._r * self._aperture_scale)]

    @staticmethod
    def is_planar():
        """
        Mirrors are planar?
        (override for non-planar mirrors)
        """
        return False

    @classmethod
    def get_name(cls):
        return "  circle  "

    @classmethod
    def get_icon_vertices(cls):
        top_y = LAYOUT['icons']['fig_top_y']
        bottom_y = LAYOUT['icons']['fig_bottom_y']
        text_y = LAYOUT['icons']['text_bottom_y']
        radius = (top_y - bottom_y) / 2.
        center_y = (top_y + bottom_y) / 2
        center_x = 0.5
        t = np.linspace(0., np.pi * 2., 1000)

        points = np.stack([np.cos(t) * radius, np.sin(t) * radius], axis=1) + \
                 np.array((center_x, center_y)).reshape(1, 2)

        return {'points': points,
                'text_center_bottom': [0.5, text_y],
                'final_line_dashed': False}


if __name__ == "__main__":
    x = CirclePrism()
    print(x)

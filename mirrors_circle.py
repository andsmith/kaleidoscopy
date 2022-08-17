import numpy as np
from mirrors import MirrorPrism
from mirror_utils import transform_points
from surfaces import Cylinder
from layout import LAYOUT

class CirclePrism(MirrorPrism):
    def __init__(self):
        """
        """
        super(CirclePrism, self).__init__()

    SHAPING_INSTRUCTIONS = MirrorPrism.SHAPING_INSTRUCTIONS

    def get_rel_shape(self, n_approx=1000, **kwargs):
        """
        Get coordinates of vertices of current shape,  fit into the unit square.

        # since elliptical, will be approximate
        """
        t = np.linspace(0.0, np.pi * 2.0, n_approx)
        r = self.get_radius()
        center = np.array([0.5, 0.4]).reshape(1, 2)
        points = np.hstack([(np.cos(t) * r).reshape(-1, 1),
                            (np.sin(t) * r).reshape(-1, 1)]) + center
        points = transform_points(unit_points=points, scale=self._aperture_scale, center=np.array([0.5, 0.5]))

        return points

    def get_radius(self):
        return 0.5 * self._aperture_scale

    def _mouse_adjust(self, pos, d_pos, d_button, button_state, keyboard_state):
        pass  # no params!

    def get_surfaces(self):
        """
        Get Surface() objects (corresponding to all mirrors) from current params
        """
        return [Cylinder(np.array([0.0, 0.0]), self.get_radius())]

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

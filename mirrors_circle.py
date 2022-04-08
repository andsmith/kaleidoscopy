import numpy as np
from mirrors import MirrorPrism
from mirror_utils import transform_points


class CirclePrism(MirrorPrism):
    def __init__(self):
        """
        """
        super(CirclePrism, self).__init__()

    SHAPING_INSTRUCTIONS = MirrorPrism.SHAPING_INSTRUCTIONS

    def get_rel_shape(self, scale=1.0, n_approx=1000, **kwargs):
        """
        Get coordinates of vertices of current shape,  fit into the unit square.

        # since elliptical, will be approximate
        """
        t = np.linspace(0.0, np.pi * 2.0, n_approx)
        r = 0.5 * scale
        center = np.array([0.5, 0.4]).reshape(1, 2)
        points = np.hstack([(np.cos(t) * r).reshape(-1,1),
                            (np.sin(t) * r).reshape(-1,1)]) + center
        points = transform_points(unit_points=points, scale=scale, center = np.array([0.5,0.5]))

        return points

    def _mouse_adjust(self, pos, d_pos, d_button, button_state,keyboard_state):
        return

    @classmethod
    def get_name(cls):
        return "  circle  "

    @classmethod
    def get_icon_vertices(cls):
        top_y = super().ICON_LAYOUT['fig_top_y']
        bottom_y = super().ICON_LAYOUT['fig_bottom_y']
        diameter = top_y - bottom_y
        points = CirclePrism().get_rel_shape(scale=diameter, n_approx=1000)

        text_y = super().ICON_LAYOUT['text_bottom_y']
        return {'points': points,
                'text_center_bottom': [0.5, text_y],
                'final_line_dashed': False}

if __name__=="__main__":
    x = CirclePrism()
    print(x)

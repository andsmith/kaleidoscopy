import numpy as np
from mirrors import MirrorPrism
from mirror_utils import transform_points
from gui_utils.mouse import MouseButtons, ButtonStates
from layout import LAYOUT


class IsoscelesPrism(MirrorPrism):

    def __init__(self):
        """
        default is 30 degree wedge.
        """
        self._theta = np.deg2rad(15.0)
        super(IsoscelesPrism, self).__init__()

    SHAPING_INSTRUCTIONS = MirrorPrism.SHAPING_INSTRUCTIONS + \
                           [' Left-click + Drag up-and-down:  Vertex angle"']

    def get_unscaled_shape(self, **kwargs):
        """
        Get "raw" 2d vertices of mirror corners, unscaled by FOV, just defined by shape customization params.
        (i.e. should be largest inscribed in unit square)
        """
        pass

        top = (0.5, 1.0)
        bottom_left = (0.5 - np.sin(self._theta / 2.0), 1.0 - np.cos(self._theta / 2.0))
        bottom_right = (0.5 + np.sin(self._theta / 2.0), 1.0 - np.cos(self._theta / 2.0))
        points = np.array([top, bottom_left, bottom_right])

        return points

    def _mouse_adjust(self, pos, d_pos, d_button, button_state, keyboard_state):

        if button_state[MouseButtons.LEFT] == ButtonStates.DOWN:
            if d_pos is not None:
                change = d_pos[1] / 400.0
                if change != 0:
                    self._theta += np.deg2rad(3 * np.sign(change))
                    self._mask = None
                if self._theta < np.deg2rad(1):
                    self._theta = np.deg2rad(1)
                if self._theta >= np.deg2rad(179):
                    self._theta = np.deg2rad(178)

    @classmethod
    def get_name(cls):
        return "iso. triangle"

    @classmethod
    def get_icon_vertices(cls):
        top_y = LAYOUT['icons']['fig_top_y']
        bottom_y = LAYOUT['icons']['fig_bottom_y']
        text_y = LAYOUT['icons']['text_bottom_y']
        points = [(0.63, bottom_y),
                  (0.37, bottom_y),
                  (0.5, top_y), ]

        final_point_a = np.array(points[0]) * 0.5 + np.array(points[2]) * 0.5
        final_point_b = np.array(points[0]) * 0.8 + np.array(points[2]) * 0.2
        points.extend([tuple(final_point_a), tuple(final_point_b)])
        return {'points': points,
                'text_center_bottom': [0.5, text_y],
                'final_line_dashed': True}


if __name__ == "__main__":
    x = IsoscelesPrism()
    print(x)

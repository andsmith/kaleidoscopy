import numpy as np
from mirrors import MirrorPrism
from mirror_utils import transform_points
from gui_utils.mouse import MouseButtons, ButtonStates, ModKeys

class IsoscelesPrism(MirrorPrism):
    def __init__(self):
        """
        default is 30 degree wedge.
        """
        self._theta = np.deg2rad(15.0)
        super(IsoscelesPrism, self).__init__()

    SHAPING_INSTRUCTIONS = MirrorPrism.SHAPING_INSTRUCTIONS + \
                           [' Left-click + Drag up-and-down:  Vertex angle"']

    def get_rel_shape(self, scale=1.0, **kwargs):
        """
        Get coordinates of vertices of current shape,  fit into the unit square.
        """
        top = (0.5, 1.0)
        bottom_left = (0.5 - np.sin(self._theta / 2.0), 1.0 - np.cos(self._theta / 2.0))
        bottom_right = (0.5 + np.sin(self._theta / 2.0), 1.0 - np.cos(self._theta / 2.0))
        points = np.array([top, bottom_left, bottom_right])

        points = transform_points(unit_points=points, scale=scale, center = np.array([0.5,0.5]))

        return points

    def set_corners(self, theta_deg, h_cm, **kwargs):
        theta = np.deg2rad(theta_deg)
        corners = [np.array([-np.sin(theta), -h_cm / 2]),
                   np.array([0, h_cm / 2]),
                   np.array([np.sin(theta), -h_cm / 2]), ]
        corners = np.array(corners + corners[0])
        super(IsoscelesPrism, self)._set_corners(corners=corners, **kwargs)

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
        top_y = super().ICON_LAYOUT['fig_top_y']
        bottom_y = super().ICON_LAYOUT['fig_bottom_y']
        text_y = super().ICON_LAYOUT['text_bottom_y']
        points = [(0.63, bottom_y),
                  (0.37, bottom_y),
                  (0.5, top_y), ]

        final_point_a = np.array(points[0]) * 0.5 + np.array(points[2]) * 0.5
        final_point_b = np.array(points[0]) * 0.8 + np.array(points[2]) * 0.2
        points.extend([tuple(final_point_a), tuple(final_point_b)])
        return {'points': points,
                'text_center_bottom': [0.5, text_y],
                'final_line_dashed': True}

if __name__=="__main__":
    x = IsoscelesPrism()
    print(x)

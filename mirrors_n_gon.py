import numpy as np
from mirrors import MirrorPrism
from mirror_utils import transform_points
from gui_utils.mouse import MouseButtons, ButtonStates
from layout import LAYOUT


class NGonPrism(MirrorPrism):
    _MAX_N = 15

    def __init__(self, ):
        self._n = 6
        self._phi = 0.0
        self._button_first_pressed_pos = None
        self._base_n = 0
        self._base_phi = 0
        super(NGonPrism, self).__init__()

    @classmethod
    def get_name(cls):
        return " n-gon "

    SHAPING_INSTRUCTIONS = MirrorPrism.SHAPING_INSTRUCTIONS + \
        ['  left-click + drag up/down:  +/- N',
         '  left-click + drag left/right:  +/- angle', ]

    def get_unscaled_shape(self, **kwargs):
        """
        Get coordinates of vertices of current shape, fit into the unit square.
        """
        r = 0.5
        center = np.array([0.5, 0.5]).reshape(1, 2)
        theta = np.linspace(self._phi, np.pi * 2 + self._phi, self._n, endpoint=False)
        points = np.array([(np.cos(t), np.sin(t)) for t in theta]) * r + center
        return points[::-1,:]  # clockwise

    def _mouse_adjust(self, pos, d_pos, d_button, button_state, keyboard_state):

        if d_button == 'l-up':
            self._base_n = self._n
            self._base_phi = self._phi

        if button_state[MouseButtons.LEFT] == ButtonStates.DOWN:
            if self._button_first_pressed_pos is None:
                self._button_first_pressed_pos = pos
                self._base_n = self._n
                self._base_phi = self._phi
            else:
                # change N
                change = self._button_first_pressed_pos[1] - pos[1]
                new_sides = int(change / 25.)
                new_n = self._base_n + new_sides

                if 2 < new_n < self._MAX_N:
                    if new_n != self._n:
                        self._n = new_n
                        self._mask = None

                # change phi
                change = self._button_first_pressed_pos[0] - pos[0]
                if change != 0:
                    self._phi = self._base_phi + np.deg2rad(change / 2.0)
                    self._mask = None
        else:
            self._button_first_pressed_pos = None

    @classmethod
    def get_icon_vertices(cls):

        top_y = LAYOUT['icons']['fig_top_y']
        bottom_y = LAYOUT['icons']['fig_bottom_y']
        text_y = LAYOUT['icons']['text_bottom_y']

        img_center_y = (top_y + bottom_y) / 2.0
        img_center_x = 0.5

        radius = (bottom_y - top_y) / 2.0

        theta = np.pi / 3.0
        points = []
        for i in range(6):
            points.append((radius * np.cos(theta) + img_center_x,
                           radius * np.sin(theta) + img_center_y))
            theta += np.pi / 3.0
        final_point_b = np.array(points[0]) * 0.6 + np.array(points[5]) * 0.4
        points.append(tuple(final_point_b))
        return {'points': points,
                'text_center_bottom': [0.5, text_y],
                'final_line_dashed': True}


if __name__ == "__main__":
    t = NGonPrism()
    img = np.zeros((640, 480, 3), dtype=np.uint8) + 128
    mask = t.get_masked_image(img)
    plt.imshow(mask);
    plt.show()

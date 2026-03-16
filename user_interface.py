"""
UILayer draws the UI over the current output frame, inset by UI_MARGIN_PX on all sides.
It will handle all keypresses and manage the app's state (change it as directed by the user).

From inactive mode, space enters menu mode.  If currently rendering, this continues in the background.

Menu mode:

    Show an icon for each of the available preset mirror configs, tabs through them, and selects one with space.
    An additional option, the custom editor, is always the last option in this list.
    If one of the non-custom options is selected, the app starts rendering, otherwise manual config mode starts.
    
    The menu divides the frame into at least 4 columns for all the options, populates cells in rows then cols.
    Each cell is a square padded on all four sides by COL_MARGIN_PX, and the content is centered in the cell. 
    Draw the square (-1 to +1 for XY in natural coords) for each icon in gray.
        
    In addition, the lines and control points of each shape will be drawn in the icon (as in manual config mode).
    
    The selected option will have its square drawn in SELECTED_COLOR and twice as thick.

Manual config mode:

    Starting with an equliateral triangle if app currenty has no mirrors (else the current mirror configuration).
    Line segments are drawn for each mirror in MIRROR_COLOR.
    Endpoints are drawn as filled circles in the same color or size END_PT_RAD.
    
    Control points are drawn as a circle around a dot at each (shared) endpoint, both in CTRL_COLOR if not selected.
    
    Mouse behavior:
       When the mouse cursor is within MOUSEOVER_DIST pixels of a control point, both parts of it are highlighted by drawing them
       in MOUSEOVER_COLOR.  IF the user then clicks, the control point is "selected" (drawn in SELECTED_COLOR) and can be dragged.
       
       Whenever the closest control point is mouseovered/selected, the second closest control point's inner dot should be also drawn that
       color, indicating the segment is also being highlighted/selected. 


Inactive mode:


"""

from geom import COLORS, make_test_check
import cv2
import numpy as np
from enum import IntEnum
from mirror_configs import PresetFactory
import logging
from mirror import Mirror
from mirror_tube import MirrorTube

class UIModes(IntEnum):
    INACTIVE = 0
    MENU = 1
    EDITING_MIRRORS = 2


UI_MARGIN_PX = 30
COL_MARGIN_PX = 10

UI_BKG = COLORS['dark_navy']
UI_OPACITY = 200

MIRROR_THIC = 1
CTRL_PT_RAD = (6, 2)
MIRROR_COLOR = COLORS['light_blue']
CTRL_COLOR = COLORS['white']
END_PT_RAD = 4

MOUSEOVER_DIST = 20
MOUSEOVER_COLOR = COLORS['orange']
SELECTED_COLOR = COLORS['neon_green']


class UILayer(object):
    def __init__(self, app):
        self.app = app
        self.mode = UIModes.MENU
        self._custom_mirrors = None  # create this MirrorTube object
        self.selected_menu_idx = 0
        self._option_names = PresetFactory.PRESET_NAMES + ["CUSTOM"]

    def draw_ui_layer(self, frame_out, draw_debug_info=True):
        """
        Draw the UI layer over the given frame_out, which is the current output of the renderer/mapper.
        This should be called every frame, even if the mode is INACTIVE, since it handles keypresses and mode changes.
        :param frame_out: the current output frame from the renderer/mapper, which we will draw the UI over.
        :param draw_debug_info: if True, draw additional debug info (like the current mode) on the frame:
            lines separating menu cells
        """

        if self.mode == UIModes.INACTIVE:
            return frame_out
        elif self.mode == UIModes.MENU:
            return self._draw_menu(frame_out)
        elif self.mode == UIModes.EDITING_MIRRORS:
            return self._draw_mirror_editor(frame_out)
        else:
            raise ValueError(f"Invalid UI mode: {self.mode}")

    def _get_mirrors_to_customize(self):
        if self.app.mirrors is not None:
            return self.app.mirrors
        else:
            return PresetFactory.make_preset('equilateral triangle', r=.4)

    def _get_selected_preset(self):
        if self.selected_menu_idx >= len(self._option_names) - 1:
            raise ValueError("Selected menu index is out of range for presets.")
        return PresetFactory.make_preset(self._option_names[self.selected_menu_idx])

    def handle_keypress(self, key):
        
        if key==ord('q'):
            self.app.shutdown()
            return False
        
        
        if self.mode == UIModes.INACTIVE:
            if key == ord(' '):
                # PICK A CONFIG FROM THE MENU
                self.mode = UIModes.MENU
                self.selected_menu_idx = 0
            if key == ord('c'):
                # EDIT CURRENT CONFIG IN MANUAL MODE
                self.mode = UIModes.EDITING_MIRRORS

        elif self.mode == UIModes.MENU:
            if key == ord(' '):
                if self.selected_menu_idx == len(self._option_names) - 1:
                    self.mode = UIModes.EDITING_MIRRORS
                    self.mirrors = self._get_mirrors_to_customize()
                else:
                    self.app.set_mirrors_and_restart(self._get_selected_preset())
                    self.mode = UIModes.INACTIVE
            if key == 27:  # esc - resume rendering/mapping
                if self.app.mirrors is not None:
                    self.mode = UIModes.INACTIVE
                else:
                    logging.info("No mirror config selected yet, staying in menu mode.")
            if key == 9:  # tab - cycle through menu options
                self.selected_menu_idx = (self.selected_menu_idx + 1) % len(self._option_names)

        elif self.mode == UIModes.EDITING_MIRRORS:
            if key == ord(' '):
                self.app.set_mirrors_and_restart(self._custom_mirrors)
                self.mode = UIModes.INACTIVE

            if key == 27:  # esc - go back to menu if nothing to do
                if self.app.mirrors is not None:
                    self.mode = UIModes.MENU
                    self.selected_menu_idx = 0
                else:
                    self.mode = UIModes.INACTIVE
                    
        return True
    
    
    def _get_icon_mirrors(self, kind):
        """
         (for custom shape use an octagon with small amounts of noise added to its verticies.)
        """
        if kind == "CUSTOM":
            n=8
            
            mt = MirrorTube.make_reg_n_gon(n, radius=0.4)
            # Extract the mirrors, move them, make a new tube
            points = np.array([m.p0 for m in mt.mirrors] ) + np.random.normal(scale=0.05, size=(n, 3))
            mirrors = [Mirror(points[i], points[i+1]) for i in range(n-1)]
            mirrors.append(Mirror(points[-1], points[0]))
            return MirrorTube(mirrors)
        else:
            return PresetFactory.make_preset(kind, r=0.4)
    
    def _draw_icon(self, frame, icon_name, bbox, is_selected=False):
        """
        Draw the icon for the given preset name in the given bounding box on the frame.

        1. compute margin within bbox, only draw in here.  Get the mirrors, scale them to this box.
        2. Draw a dark colored square for the whole box.
        3. Draw the lines and control points.

        The bbox is {'x': (x_min, x_max), 'y': (y_min, y_max)} in pixel coordinates.
        :param: frame: the frame to draw on
        :param: icon_name: the name of the preset to draw the icon for
        :param: bbox: the bounding box to draw the icon in, dict with keys 'x' and 'y' mapping to (min, max) pixel coords, may not be square, use
            largest centered square within it for drawing the icon.
        :param: is_selected: if True, draw the border in SELECTED_COLOR and twice as thick.
        """
        x_min, x_max = bbox['x']
        y_min, y_max = bbox['y']

        # Step 1: Largest centered square within bbox
        sq_size = min(x_max - x_min, y_max - y_min)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        sq_x0 = cx - sq_size // 2
        sq_x1 = sq_x0 + sq_size
        sq_y0 = cy - sq_size // 2
        sq_y1 = sq_y0 + sq_size

        # Step 2: Inset draw area by COL_MARGIN_PX
        draw_x0 = sq_x0 + COL_MARGIN_PX
        draw_x1 = sq_x1 - COL_MARGIN_PX
        draw_y0 = sq_y0 + COL_MARGIN_PX
        draw_y1 = sq_y1 - COL_MARGIN_PX

        # Step 3: Fill draw area with background color
        cv2.rectangle(frame, (draw_x0, draw_y0), (draw_x1, draw_y1), UI_BKG, -1)

        # Step 4: Draw border
        border_color = SELECTED_COLOR if is_selected else COLORS['gray']
        border_thickness = 2 if is_selected else 1
        cv2.rectangle(frame, (draw_x0, draw_y0), (draw_x1, draw_y1), border_color, border_thickness)

        # Step 5: Get mirrors
        tube = self._get_icon_mirrors(icon_name)

        # Step 6: Coordinate transform helper ([-1,+1] -> pixel, y-flipped)
        def to_px(pt):
            px = int(draw_x0 + (pt[0] + 1) / 2 * (draw_x1 - draw_x0))
            py = int(draw_y1 - (pt[1] + 1) / 2 * (draw_y1 - draw_y0))
            return (px, py)

        # Step 7: Draw mirror lines
        for m in tube.mirrors:
            cv2.line(frame, to_px(m.p0), to_px(m.p1), MIRROR_COLOR, MIRROR_THIC)

        # Step 8: Draw control points at each mirror's p0
        for m in tube.mirrors:
            center = to_px(m.p0)
            cv2.circle(frame, center, CTRL_PT_RAD[0], CTRL_COLOR, 1)
            cv2.circle(frame, center, CTRL_PT_RAD[1], CTRL_COLOR, -1)

    
    def _draw_menu(self, frame_out):
        h, w = frame_out.shape[:2]
        n_cols = 4
        n_options = len(self._option_names)
        n_rows = (n_options + n_cols - 1) // n_cols

        cell_w = (w - 2 * UI_MARGIN_PX) // n_cols
        cell_h = (h - 2 * UI_MARGIN_PX) // n_rows

        for idx, name in enumerate(self._option_names):
            row = idx // n_cols
            col = idx % n_cols
            x0 = UI_MARGIN_PX + col * cell_w
            x1 = x0 + cell_w
            y0 = UI_MARGIN_PX + row * cell_h
            y1 = y0 + cell_h
            bbox = {'x': (x0, x1), 'y': (y0, y1)}
            self._draw_icon(frame_out, name, bbox, is_selected=(idx == self.selected_menu_idx))

        return frame_out

    def _draw_mirror_editor(self, frame_out):
        pass
    
    def draw_layer(self, frame_out):
        if self.mode == UIModes.INACTIVE:
            return frame_out
        elif self.mode == UIModes.MENU:
            return self._draw_menu(frame_out)
        elif self.mode == UIModes.EDITING_MIRRORS:
            return self._draw_mirror_editor(frame_out)
        else:
            raise ValueError(f"Invalid UI mode: {self.mode}")


def test_ui_layer():
    """
    Test all behaviors of the full app, rendering on a static image instead of the mapper's output.
    Run at a 30fps to simulate the mapper's output.
    """
    class FakeApp:
        mirrors = None
        shutdown = False
        def set_mirrors_and_restart(self, new_mirrors):
            logging.info("FakeApp: set_mirrors_and_restart called with new_mirrors: %s", new_mirrors)
            self.mirrors = new_mirrors
        def shutdown(self):
            logging.info("FakeApp: shutdown called.")
            self.shutdown = True

    img = make_test_check((800, 600), sq_size=50, n_colors=7, randomize=True)
    ui = UILayer(app=FakeApp())
    while True:
        frame = ui.draw_layer(img.copy())
        cv2.imshow("UI Layer Test", frame)
        key = cv2.waitKey(30)
        if not ui.handle_keypress(key):
            break
        
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ui_layer()

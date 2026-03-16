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

from geom import COLORS, make_test_check, pt_in_bbox, lineseg_dist, CLOSEST_MIRROR
import matplotlib.path as mpath
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

MIRROR_THIC = 2
CTRL_PT_RAD = (8, 4)
MIRROR_COLOR = COLORS['light_blue']
CTRL_COLOR = COLORS['white']
END_PT_RAD = 7

MOUSEOVER_DIST = 20
MOUSEOVER_COLOR = COLORS['orange']
SELECTED_COLOR = COLORS['neon_green']


class UILayer(object):
    def __init__(self, app, window_name=None):
        self.app = app
        self.mode = UIModes.MENU
        self._custom_mirrors = None
        self.selected_menu_idx = 0
        self._option_names = PresetFactory.PRESET_NAMES + ["CUSTOM"]
        self.window_name = window_name
        self._menu_cells = []
        self._icon_radius = 0.4

        # Mouse state for manual mirror editing
        self.mouse_pos = (0, 0)
        self.selected_ctrl_pt = None  # vertex index into _custom_mirrors.mirrors
        self.mouse_dragging = False
        self._translating = False
        self._translate_anchor = None   # normalized (x,y) at drag start
        self._translate_base_pts = None # vertex positions at drag start
        self._mouse_in_interior = False
        self._hover_idx = None
        self._second_idx = None
        self._mouse_cb_registered = False
        

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
                self._activate_selected_menu_item()
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
    
    
    def _activate_selected_menu_item(self):
        if self.selected_menu_idx == len(self._option_names) - 1:
            self.mode = UIModes.EDITING_MIRRORS
            self._custom_mirrors = self._get_mirrors_to_customize()
        else:
            self.app.set_mirrors_and_restart(self._get_selected_preset())
            self.mode = UIModes.INACTIVE

    def _get_icon_mirrors(self, kind):
        """
         (for custom shape use an octagon with small amounts of noise added to its verticies.)
        """
        r = self._icon_radius
        if kind == "CUSTOM":
            n = 5
            mt = MirrorTube.make_reg_n_gon(n, radius=r)
            rng = np.random.default_rng(432)
            noise_scale = r * 0.25
            points = np.array([m.p0 for m in mt.mirrors]) + rng.normal(scale=noise_scale, size=(n, 3))
            mirrors = [Mirror(points[i], points[i+1]) for i in range(n-1)]
            mirrors.append(Mirror(points[-1], points[0]))
            return MirrorTube(mirrors)
        else:
            return PresetFactory.make_preset(kind, r=r)
    
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
        is_custom = (icon_name == "CUSTOM")
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
            if is_custom:
                cv2.circle(frame, center, CTRL_PT_RAD[0], CTRL_COLOR, 1)
            cv2.circle(frame, center, CTRL_PT_RAD[1], CTRL_COLOR, -1)

        # Optical axis crosshair
        ox, oy = to_px([0, 0, 0])
        arm = 3
        cv2.line(frame, (ox - arm, oy), (ox + arm, oy), COLORS['white'], 1)
        cv2.line(frame, (ox, oy - arm), (ox, oy + arm), COLORS['white'], 1)

    
    def _draw_menu(self, frame_out):
        h, w = frame_out.shape[:2]
        n_cols = 4
        n_options = len(self._option_names)
        n_rows = (n_options + n_cols - 1) // n_cols

        cell_w = (w - 2 * UI_MARGIN_PX) // n_cols
        cell_h = (h - 2 * UI_MARGIN_PX) // n_rows

        self._menu_cells = []
        for idx, name in enumerate(self._option_names):
            row = idx // n_cols
            col = idx % n_cols
            x0 = UI_MARGIN_PX + col * cell_w
            x1 = x0 + cell_w
            y0 = UI_MARGIN_PX + row * cell_h
            y1 = y0 + cell_h
            bbox = {'x': (x0, x1), 'y': (y0, y1)}
            self._menu_cells.append(bbox)
            self._draw_icon(frame_out, name, bbox, is_selected=(idx == self.selected_menu_idx))

        return frame_out

    def _px_to_norm(self, px, py):
        x = (px - self._draw_x0) / (self._draw_x1 - self._draw_x0) * 2 - 1
        y = (self._draw_y1 - py) / (self._draw_y1 - self._draw_y0) * 2 - 1
        return np.array([x, y])

    def _cell_at(self, x, y):
        """Return the index of the menu cell containing (x, y), or None."""
        for idx, bbox in enumerate(self._menu_cells):
            x0, x1 = bbox['x']
            y0, y1 = bbox['y']
            if x0 <= x < x1 and y0 <= y < y1:
                return idx
        return None

    def _mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

        if self.mode == UIModes.MENU:
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
                idx = self._cell_at(x, y)
                if idx is not None:
                    self.selected_menu_idx = idx
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self._activate_selected_menu_item()
            elif event == cv2.EVENT_MOUSEWHEEL:
                step = 0.05 if flags > 0 else -0.05
                new_r = float(np.clip(self._icon_radius + step, 0.05, 0.85))
                old_r = self._icon_radius
                self._icon_radius = new_r
                try:
                    for name in self._option_names:
                        self._get_icon_mirrors(name)
                except ValueError:
                    self._icon_radius = old_r  # hit the valid lower bound, revert
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._hover_idx is not None:
                self.selected_ctrl_pt = self._hover_idx
                self.mouse_dragging = True
            elif self._mouse_in_interior:
                self._translating = True
                self._translate_anchor = self._px_to_norm(x, y)
                self._translate_base_pts = [m.p0[:2].copy() for m in self._custom_mirrors.mirrors]
        elif event == cv2.EVENT_MBUTTONDOWN:
            if self._hover_idx is not None:
                self._delete_vertex(self._hover_idx)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._hover_idx is not None:
                self._insert_vertex(self._hover_idx, self._second_idx)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_dragging and self.selected_ctrl_pt is not None:
                self._move_vertex(self.selected_ctrl_pt, x, y)
            elif self._translating:
                self._translate_all(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_dragging = False
            self.selected_ctrl_pt = None
            self._translating = False
            self._translate_anchor = None
            self._translate_base_pts = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._scale_mirrors(1.1 if flags > 0 else 1 / 1.1)

    def _move_vertex(self, idx, px, py):
        """Move vertex idx to pixel position (px, py), rebuilding the MirrorTube."""
        n = len(self._custom_mirrors.mirrors)
        pts = [m.p0[:2].copy() for m in self._custom_mirrors.mirrors]
        pts[idx] = self._px_to_norm(px, py)
        try:
            mirrors = [Mirror(pts[i], pts[(i + 1) % n]) for i in range(n)]
            self._custom_mirrors = MirrorTube(mirrors)
        except ValueError:
            pass  # revert silently if validation fails

    def _translate_all(self, px, py):
        """Translate all vertices by the delta from the drag anchor."""
        delta = self._px_to_norm(px, py) - self._translate_anchor
        n = len(self._translate_base_pts)
        new_pts = [p + delta for p in self._translate_base_pts]
        try:
            mirrors = [Mirror(new_pts[i], new_pts[(i + 1) % n]) for i in range(n)]
            self._custom_mirrors = MirrorTube(mirrors)
        except ValueError:
            pass

    def _scale_mirrors(self, factor):
        """Scale all vertices uniformly about the origin; revert if validation fails."""
        pts = [m.p0[:2] * factor for m in self._custom_mirrors.mirrors]
        n = len(pts)
        old = self._custom_mirrors
        try:
            self._custom_mirrors = MirrorTube([Mirror(pts[i], pts[(i + 1) % n]) for i in range(n)])
        except ValueError:
            self._custom_mirrors = old

    def _delete_vertex(self, idx):
        """Remove vertex idx, merging its two adjacent mirrors into one. Requires >= 3 remaining."""
        pts = [m.p0[:2].copy() for m in self._custom_mirrors.mirrors]
        if len(pts) <= 3:
            return
        pts.pop(idx)
        n = len(pts)
        try:
            mirrors = [Mirror(pts[i], pts[(i + 1) % n]) for i in range(n)]
            self._custom_mirrors = MirrorTube(mirrors)
        except ValueError:
            pass

    def _insert_vertex(self, primary_idx, secondary_idx):
        """Insert a vertex at the midpoint of the edge between primary and secondary."""
        pts = [m.p0[:2].copy() for m in self._custom_mirrors.mirrors]
        n = len(pts)
        # Determine which vertex is 'from' (the one whose mirror points to the other)
        if secondary_idx == (primary_idx + 1) % n:
            from_idx = primary_idx
        else:
            from_idx = secondary_idx
        midpoint = (pts[from_idx] + pts[(from_idx + 1) % n]) / 2
        pts.insert(from_idx + 1, midpoint)
        n = len(pts)
        try:
            mirrors = [Mirror(pts[i], pts[(i + 1) % n]) for i in range(n)]
            self._custom_mirrors = MirrorTube(mirrors)
        except ValueError:
            pass

    def _draw_mirror_editor(self, frame_out):
        """
        Find the largest center square in the frame (minus at least UI_MARGIN_PX on all sides).
        Draw the lines/points scaled to this square, centered in it etc.
        """
        if self._custom_mirrors is None:
            self._custom_mirrors = self._get_mirrors_to_customize()

        h, w = frame_out.shape[:2]
        sq_size = min(w, h) - 2 * UI_MARGIN_PX
        cx, cy = w // 2, h // 2
        draw_x0 = cx - sq_size // 2
        draw_x1 = draw_x0 + sq_size
        draw_y0 = cy - sq_size // 2
        draw_y1 = draw_y0 + sq_size

        # Store for use in mouse callback
        self._draw_x0, self._draw_x1 = draw_x0, draw_x1
        self._draw_y0, self._draw_y1 = draw_y0, draw_y1

        cv2.rectangle(frame_out, (draw_x0, draw_y0), (draw_x1, draw_y1), UI_BKG, -1)
        cv2.rectangle(frame_out, (draw_x0, draw_y0), (draw_x1, draw_y1), COLORS['gray'], 1)

        def to_px(pt):
            px = int(draw_x0 + (pt[0] + 1) / 2 * (draw_x1 - draw_x0))
            py = int(draw_y1 - (pt[1] + 1) / 2 * (draw_y1 - draw_y0))
            return (px, py)

        mirrors = self._custom_mirrors.mirrors
        ctrl_pts_px = [to_px(m.p0) for m in mirrors]

        # Find closest vertex to mouse, then pick the angularly closer neighbor as secondary
        mx, my = self.mouse_pos
        dists = [np.hypot(p[0] - mx, p[1] - my) for p in ctrl_pts_px]
        closest_idx = int(np.argmin(dists))
        n = len(ctrl_pts_px)
        prev_idx = (closest_idx - 1) % n
        next_idx = (closest_idx + 1) % n
        cx_px, cy_px = ctrl_pts_px[closest_idx]
        angle_to_mouse = np.arctan2(my - cy_px, mx - cx_px)
        angle_to_prev = np.arctan2(ctrl_pts_px[prev_idx][1] - cy_px, ctrl_pts_px[prev_idx][0] - cx_px)
        angle_to_next = np.arctan2(ctrl_pts_px[next_idx][1] - cy_px, ctrl_pts_px[next_idx][0] - cx_px)
        diff_prev = abs(np.arctan2(np.sin(angle_to_mouse - angle_to_prev), np.cos(angle_to_mouse - angle_to_prev)))
        diff_next = abs(np.arctan2(np.sin(angle_to_mouse - angle_to_next), np.cos(angle_to_mouse - angle_to_next)))
        second_idx = prev_idx if diff_prev < diff_next else next_idx
        self._second_idx = second_idx

        mouseover_idx = closest_idx if dists[closest_idx] < MOUSEOVER_DIST else None
        self._hover_idx = mouseover_idx

        poly_verts = np.array([m.p0[:2] for m in mirrors])
        norm_mouse = self._px_to_norm(mx, my)
        self._mouse_in_interior = (mouseover_idx is None and
                                   mpath.Path(poly_verts).contains_point(norm_mouse))

        def ring_color(i):
            if self._translating or self.selected_ctrl_pt == i:
                return SELECTED_COLOR
            if self._mouse_in_interior or mouseover_idx == i:
                return MOUSEOVER_COLOR
            return CTRL_COLOR

        def dot_color(i):
            if self._translating or self.selected_ctrl_pt == i:
                return SELECTED_COLOR
            if self._mouse_in_interior or mouseover_idx == i:
                return MOUSEOVER_COLOR
            # second closest inner dot mirrors the active color
            if i == second_idx:
                if mouseover_idx is not None:
                    return MOUSEOVER_COLOR
            return CTRL_COLOR

        for m in mirrors:
            cv2.line(frame_out, to_px(m.p0), to_px(m.p1), MIRROR_COLOR, MIRROR_THIC)

        for i, pt_px in enumerate(ctrl_pts_px):
            cv2.circle(frame_out, pt_px, CTRL_PT_RAD[0], ring_color(i), 1)
            cv2.circle(frame_out, pt_px, CTRL_PT_RAD[1], dot_color(i), -1)

        # Optical axis cross
        origin_px = to_px([0, 0, 0])
        origin_inside = mpath.Path(poly_verts).contains_point((0, 0))
        not_too_close = all(lineseg_dist(m.p0, m.p1, (0, 0, 0)) >= CLOSEST_MIRROR for m in mirrors)
        cross_color = COLORS['white'] if (origin_inside and not_too_close) else COLORS['red']
        arm = 5
        cv2.line(frame_out, (origin_px[0] - arm, origin_px[1]), (origin_px[0] + arm, origin_px[1]), cross_color, 1)
        cv2.line(frame_out, (origin_px[0], origin_px[1] - arm), (origin_px[0], origin_px[1] + arm), cross_color, 1)

        return frame_out


    
    def draw_layer(self, frame_out):
        if not self._mouse_cb_registered and self.window_name is not None:
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            self._mouse_cb_registered = True

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
    window_name = "UI Layer Test"
    cv2.namedWindow(window_name)
    ui = UILayer(app=FakeApp(), window_name=window_name)
    while True:
        frame = ui.draw_layer(img.copy())
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30)
        if not ui.handle_keypress(key):
            break
        
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ui_layer()

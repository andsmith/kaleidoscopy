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

from geom import COLORS, BKG, make_test_check, pt_in_bbox, lineseg_dist, CLOSEST_MIRROR
import matplotlib.path as mpath
import cv2
import numpy as np
from enum import IntEnum
from mirror_configs import PresetFactory
import logging
from mirror import Mirror
from mirror_tube import MirrorTube
from init_rays import compute_fov

class UIModes(IntEnum):
    INACTIVE = 0
    MENU = 1
    EDITING_MIRRORS = 2
    LIVE_EDITING = 3


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

TITLE_FONT = cv2.FONT_HERSHEY_DUPLEX
TITLE_MAX_FONT_SCALE = 1.75
TITLE_INDENT_PX = 20

HELP_FONT_SCALES = {'title': 1.25, 'subtitle': .6, 'section': .9, 'text': .7}

_HELP_SECTIONS = [
    ("hotkeys", [
        ("h / H",      "show / hide this help"),
        ("q / ESC",    "quit"),
        ("f",          "toggle fullscreen"),
        ("v",          "toggle control-point overlay (live edit)"),
        ("-  /  =",    "scale mirrors down / up"),
    ]),
    ("menu", [
        ("SPACE",      "open menu / confirm selection"),
        ("TAB",        "cycle menu options (in menu)"),
        ("x",          "close menu without resetting raytracer"),
    ]),
    ("effects", [
        ("0",          "reset pan / zoom and disable all effects"),
        ("1",          "cycle fade effect"),
        ("2",          "cycle stained-glass effect"),
        ("w",          "cycle stain threshold"),
    ]),
    ("--debug enabled keys", [
        ("ENTER",      "advance one raytrace step"),
        ("b",          "toggle bounce / hit draw mode"),
        ("s",          "start (vector) draw mode"),
        ("t",          "trails draw mode"),
        ("r",          "reset zoom"),
        ("scroll",     "zoom in / out"),
        ("d",          "close debug window"),
    ]),
]

class UILayer(object):
    def __init__(self, app, window_name=None):
        self.app = app
        self.mode = UIModes.MENU
        self._custom_mirrors = None
        self.selected_menu_idx = 0
        self._option_names = PresetFactory.PRESET_NAMES + ["CUSTOM", "LIVE EDIT"]
        self.window_name = window_name
        self._menu_cells = []
        self._icon_radii = {name: 0.4 for name in PresetFactory.PRESET_NAMES + ["CUSTOM", "LIVE EDIT"]}

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

        # View pan/zoom state (INACTIVE mode)
        self._view_pan = [0.0, 0.0]   # source-pixel offset of viewport center
        self._view_zoom = 1.0
        self._pan_drag_last = None    # (x, y) of previous drag position

        # Art layer indices
        self._fade_idx = 0         # 0=off, 1-5 = decay factor
        self._stain_idx = 0        # 0=off, 1-5 = leading distance
        self._stain_thresh_idx = 0 # threshold scale factor index
        self._fullscreen = False
        self._show_ctrl_pts = True  # toggled by 'v' in live-edit mode
        self._drag_kind = None      # None|'vertex'|'translate'|'pan' while LMB is held


    @property
    def view_transform(self):
        """Return (pan_x, pan_y, zoom) for use in view-space coordinate transform."""
        return (self._view_pan[0], self._view_pan[1], self._view_zoom)

    @property
    def fade_idx(self):
        return self._fade_idx

    @property
    def stain_idx(self):
        return self._stain_idx

    @property
    def stain_thresh_idx(self):
        return self._stain_thresh_idx

    def reset_view(self):
        self._view_pan = [0.0, 0.0]
        self._view_zoom = 1.0
        self._pan_drag_last = None

    def _reset_view_and_effects(self):
        self.reset_view()
        self._fade_idx = 0
        self._stain_idx = 0
        self._stain_thresh_idx = 0

    def _is_editing_mode(self):
        return self.mode in (UIModes.EDITING_MIRRORS, UIModes.LIVE_EDITING)

    @staticmethod
    def _make_edit_tube(mirrors):
        """Construct an editable mirror tube without center/min-distance hard constraints."""
        return MirrorTube(
            mirrors,
            require_center_containment=False,
            require_min_mirror_dist=False,
        )

    def _commit_live_if_active(self):
        """After each shape-changing edit, restart the raytracer (live mode only)."""
        if self.mode == UIModes.LIVE_EDITING and self._custom_mirrors is not None:
            self.app.set_mirrors_and_restart(self._custom_mirrors, preserve_old_map=True)

    def _restart_edit(self):
        """Restart raytracer with current custom mirrors from any editing mode."""
        if self._custom_mirrors is None:
            return
        preserve = (self.mode == UIModes.LIVE_EDITING)
        self.app.set_mirrors_and_restart(self._custom_mirrors, preserve_old_map=preserve)

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
            return self._draw_mirror_editor(frame_out, overlay=False)
        elif self.mode == UIModes.LIVE_EDITING:
            return self._draw_mirror_editor(frame_out, overlay=True)
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
        name = self._option_names[self.selected_menu_idx]
        return PresetFactory.make_preset(name, r=self._icon_radii[name])

    def handle_keypress(self, key):
        
        if key == ord('q'):
            self.app.shutdown()
            return False

        if key == ord('f') and self.window_name is not None:
            self._fullscreen = not self._fullscreen
            prop = cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)
        
        
        if self.mode == UIModes.INACTIVE:
            if key == ord(' '):
                # PICK A CONFIG FROM THE MENU
                self.mode = UIModes.MENU
                self.selected_menu_idx = 0
            if key == ord('0'):
                self._reset_view_and_effects()
            if key == ord('1'):
                self._fade_idx = (self._fade_idx + 1) % 6
            if key == ord('2'):
                self._stain_idx = (self._stain_idx + 1) % 6
            if key == ord('w'):
                self._stain_thresh_idx = (self._stain_thresh_idx + 1) % 4

        elif self.mode == UIModes.LIVE_EDITING:
            if key == ord(' '):
                self.mode = UIModes.MENU   # pick a different config
                self.selected_menu_idx = 0
            elif key == 27:
                self.mode = UIModes.INACTIVE
            if key == ord('0'):
                self._reset_view_and_effects()
            if key == ord('1'):
                self._fade_idx = (self._fade_idx + 1) % 6
            if key == ord('2'):
                self._stain_idx = (self._stain_idx + 1) % 6
            if key == ord('w'):
                self._stain_thresh_idx = (self._stain_thresh_idx + 1) % 4
            if key == ord('v'):
                self._show_ctrl_pts = not self._show_ctrl_pts
            if key == ord('-'):
                self._scale_mirrors(0.9)
                self._restart_edit()
            if key == ord('='):
                self._scale_mirrors(1.1)
                self._restart_edit()

        elif self.mode == UIModes.MENU:
            if key == ord(' '):
                self._activate_selected_menu_item()
            if key == ord('x'):  # close menu and resume rendering/mapping
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
            if key == ord('-'):
                self._scale_mirrors(0.9)
                self._restart_edit()
            if key == ord('='):
                self._scale_mirrors(1.1)
                self._restart_edit()
                    
        return True
    
    
    def _activate_selected_menu_item(self):
        n = len(self._option_names)
        if self.selected_menu_idx == n - 1:          # LIVE EDIT
            self._custom_mirrors = self._get_mirrors_to_customize()
            self.app.set_mirrors_and_restart(self._custom_mirrors, preserve_old_map=True)
            self.mode = UIModes.LIVE_EDITING
        elif self.selected_menu_idx == n - 2:         # CUSTOM
            self.mode = UIModes.EDITING_MIRRORS
            self._custom_mirrors = self._get_mirrors_to_customize()
        else:
            self.app.set_mirrors_and_restart(self._get_selected_preset())
            self.mode = UIModes.INACTIVE

    def _get_icon_mirrors(self, kind, r=None):
        """
         (for custom shape use an n-gon with small amounts of noise added to its verticies.)
        """
        if r is None:
            r = self._icon_radii[kind]
        if kind in ("CUSTOM", "LIVE EDIT"):
            seed = {'CUSTOM': 432, 'LIVE EDIT': 42}[kind]
            n = {'CUSTOM': 5, 'LIVE EDIT': 8}[kind]
            mt = MirrorTube.make_reg_n_gon(n, radius=r)
            rng = np.random.default_rng(seed)
            noise_scale = r * 0.2
            points = np.array([m.p0 for m in mt.mirrors]) + rng.normal(scale=noise_scale, size=(n, 3))
            mirrors = [Mirror(points[i], points[i+1]) for i in range(n-1)]
            mirrors.append(Mirror(points[-1], points[0]))
            return MirrorTube(mirrors)
        else:
            return PresetFactory.make_preset(kind, r=r)
    
    def _draw_icon(self, frame, icon_name, bbox, is_selected=False, label_scale=0.5, r=None):
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
        is_live_edit = (icon_name == "LIVE EDIT")
        aa = cv2.LINE_AA
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

        # Step 3 (early): Get mirrors and pixel transform — needed before background fill for cutout
        tube = self._get_icon_mirrors(icon_name, r=r)

        def to_px(pt):
            px = int(draw_x0 + (pt[0] + 1) / 2 * (draw_x1 - draw_x0))
            py = int(draw_y1 - (pt[1] + 1) / 2 * (draw_y1 - draw_y0))
            return (px, py)

        ctrl_pts_px = [to_px(m.p0) for m in tube.mirrors]

        # Step 4: Fill draw area with background color
        cv2.rectangle(frame, (draw_x0, draw_y0), (draw_x1, draw_y1), UI_BKG, -1, aa)

        # Step 4b: Draw icon title label centered at the top
        _lf = TITLE_FONT
        _lt = 1
        label_text = tube.name if tube.name is not None else icon_name
        (lw, lh), _ = cv2.getTextSize(label_text, _lf, label_scale, _lt)
        lx = draw_x0 + (draw_x1 - draw_x0 - lw) // 2
        ly = draw_y0 + lh + TITLE_INDENT_PX
        label_color = SELECTED_COLOR if is_selected else COLORS['gray']
        cv2.putText(frame, label_text, (lx, ly), _lf, label_scale, label_color, _lt, cv2.LINE_AA)

        # Step 5: For LIVE EDIT, composite source image through mirror-tube interior (cutout effect)
        if is_live_edit:
            src = getattr(self.app, '_last_input_frame', None)
            if src is None:
                src = getattr(self.app, '_bkg', None)
            if src is not None:
                dw = draw_x1 - draw_x0
                dh = draw_y1 - draw_y0
                sh, sw = src.shape[:2]
                scale = min(dw / sw, dh / sh)
                nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
                scaled = cv2.resize(src, (nw, nh))
                lx0 = (dw - nw) // 2
                ly0 = (dh - nh) // 2

                # Build polygon mask in local (draw-region) coordinates
                mask_local = np.zeros((dh, dw), dtype=np.uint8)
                pts_local = np.array([(p[0] - draw_x0, p[1] - draw_y0) for p in ctrl_pts_px],
                                     dtype=np.int32)
                cv2.fillPoly(mask_local, [pts_local], 255)

                # Place scaled image centered on a canvas the size of the draw region
                src_canvas = np.full((dh, dw, 3), UI_BKG, dtype=np.uint8)
                src_canvas[ly0:ly0 + nh, lx0:lx0 + nw] = scaled

                # Composite: copy source canvas into frame wherever polygon mask is set
                region = frame[draw_y0:draw_y1, draw_x0:draw_x1]
                region[mask_local > 0] = src_canvas[mask_local > 0]

        # Step 6: Draw border
        border_color = SELECTED_COLOR if is_selected else COLORS['gray']
        border_thickness = 2 if is_selected else 1
        cv2.rectangle(frame, (draw_x0, draw_y0), (draw_x1, draw_y1), border_color, border_thickness, aa)

        # Step 7: Draw mirror lines
        for m in tube.mirrors:
            cv2.line(frame, to_px(m.p0), to_px(m.p1), MIRROR_COLOR, MIRROR_THIC, aa)

        # Step 8: Draw control points at each mirror's p0
        for i, center in enumerate(ctrl_pts_px):
            if is_custom or is_live_edit:
                cv2.circle(frame, center, CTRL_PT_RAD[0], CTRL_COLOR, 1, aa)
            cv2.circle(frame, center, CTRL_PT_RAD[1], CTRL_COLOR, -1, aa)

        # Optical axis crosshair
        ox, oy = to_px([0, 0, 0])
        arm = 3
        cv2.line(frame, (ox - arm, oy), (ox + arm, oy), COLORS['white'], 1, aa)
        cv2.line(frame, (ox, oy - arm), (ox, oy + arm), COLORS['white'], 1, aa)

    
    def _draw_menu(self, frame_out):
        h, w = frame_out.shape[:2]
        n_cols = 4
        n_options = len(self._option_names)
        n_rows = (n_options + n_cols - 1) // n_cols

        cell_w = (w - 2 * UI_MARGIN_PX) // n_cols
        cell_h = (h - 2 * UI_MARGIN_PX) // n_rows

        # Compute a uniform font scale: largest scale where the longest label fits the draw width.
        _label_font = TITLE_FONT
        _label_thickness = 1
        sq_size = min(cell_w, cell_h)
        available_w = sq_size - 2 * COL_MARGIN_PX - 2 * TITLE_INDENT_PX
        actual_labels = [self._get_icon_mirrors(n).name or n for n in self._option_names]
        longest = max(actual_labels,
                      key=lambda s: cv2.getTextSize(s, _label_font, 1.0, _label_thickness)[0][0])
        (tw_ref, _), _ = cv2.getTextSize(longest, _label_font, 1.0, _label_thickness)
        label_scale = min(available_w / max(tw_ref, 1), TITLE_MAX_FONT_SCALE)

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
            self._draw_icon(frame_out, name, bbox,
                            is_selected=(idx == self.selected_menu_idx),
                            label_scale=label_scale,
                            r=self._icon_radii[name])

        return frame_out

    def _px_to_norm(self, px, py):
        w, h = self.app.out_size
        x_max, y_max = compute_fov(w, h)
        x = px / w * (2.0 * x_max) - x_max
        y = y_max * (1.0 - py / h * 2.0)
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

        if self.mode == UIModes.INACTIVE:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._pan_drag_last = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self._pan_drag_last is not None:
                dx = x - self._pan_drag_last[0]
                dy = y - self._pan_drag_last[1]
                self._view_pan[0] -= dx / self._view_zoom
                self._view_pan[1] -= dy / self._view_zoom
                self._pan_drag_last = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self._pan_drag_last = None
            elif event == cv2.EVENT_MOUSEWHEEL:
                old_zoom = self._view_zoom
                factor = 1.15 if flags > 0 else 1.0 / 1.15
                new_zoom = float(np.clip(old_zoom * factor, 0.1, 20.0))
                # Adjust pan so the source point under the cursor stays fixed.
                # The cursor is at output pixel (x, y); output center is (out_w/2, out_h/2).
                # Offset from center: (dx_scr, dy_scr)
                # In source space before zoom: pan + dx_scr * old_zoom
                # We want: pan' + dx_scr * new_zoom = pan + dx_scr * old_zoom
                # => pan' = pan + dx_scr * (old_zoom - new_zoom)
                out_w, out_h = self.app.out_size
                dx_scr = x - out_w / 2.0
                dy_scr = y - out_h / 2.0
                self._view_pan[0] += dx_scr * (old_zoom - new_zoom)
                self._view_pan[1] += dy_scr * (old_zoom - new_zoom)
                self._view_zoom = new_zoom
            return

        if self.mode == UIModes.MENU:
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
                idx = self._cell_at(x, y)
                if idx is not None:
                    self.selected_menu_idx = idx
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self._activate_selected_menu_item()
            elif event == cv2.EVENT_MOUSEWHEEL:
                idx = self._cell_at(x, y)
                if idx is not None:
                    name = self._option_names[idx]
                    step = 0.05 if flags > 0 else -0.05
                    old_r = self._icon_radii[name]
                    new_r = float(np.clip(old_r + step, 0.05, 0.85))
                    self._icon_radii[name] = new_r
                    try:
                        self._get_icon_mirrors(name, r=new_r)
                    except ValueError:
                        self._icon_radii[name] = old_r  # revert if shape becomes invalid
            return

        if not self._is_editing_mode():
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._hover_idx is not None:
                self.selected_ctrl_pt = self._hover_idx
                self.mouse_dragging = True
                self._drag_kind = 'vertex'
            elif self._mouse_in_interior:
                self._translating = True
                self._translate_anchor = self._px_to_norm(x, y)
                self._translate_base_pts = [m.p0[:2].copy() for m in self._custom_mirrors.mirrors]
                self._drag_kind = 'translate'
            elif self.mode == UIModes.LIVE_EDITING:
                # Not over a vertex or interior — treat as pan start
                self._pan_drag_last = (x, y)
                self._drag_kind = 'pan'
        elif event == cv2.EVENT_MBUTTONDOWN:
            if self._hover_idx is not None:
                self._delete_vertex(self._hover_idx)
                self._commit_live_if_active()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._hover_idx is not None:
                self._insert_vertex(self._hover_idx, self._second_idx)
                self._commit_live_if_active()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drag_kind == 'vertex' and self.selected_ctrl_pt is not None:
                self._move_vertex(self.selected_ctrl_pt, x, y)
            elif self._drag_kind == 'translate' and self._translating:
                self._translate_all(x, y)
            elif self._drag_kind == 'pan' and self.mode == UIModes.LIVE_EDITING and self._pan_drag_last is not None:
                dx = x - self._pan_drag_last[0]
                dy = y - self._pan_drag_last[1]
                self._view_pan[0] -= dx / self._view_zoom
                self._view_pan[1] -= dy / self._view_zoom
                self._pan_drag_last = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            was_dragging = self.mouse_dragging
            was_translating = self._translating
            self.mouse_dragging = False
            self.selected_ctrl_pt = None
            self._translating = False
            self._translate_anchor = None
            self._translate_base_pts = None
            self._pan_drag_last = None
            self._drag_kind = None
            if was_dragging or was_translating:
                self._commit_live_if_active()
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.mode == UIModes.LIVE_EDITING and not self._mouse_in_interior:
                # Outside the mirror tube interior: zoom like INACTIVE mode
                old_zoom = self._view_zoom
                factor = 1.15 if flags > 0 else 1.0 / 1.15
                new_zoom = float(np.clip(old_zoom * factor, 0.1, 20.0))
                out_w, out_h = self.app.out_size
                self._view_pan[0] += (x - out_w / 2.0) * (old_zoom - new_zoom)
                self._view_pan[1] += (y - out_h / 2.0) * (old_zoom - new_zoom)
                self._view_zoom = new_zoom
            else:
                # Over the interior (or EDITING_MIRRORS): scale the mirror tube
                self._scale_mirrors(1.1 if flags > 0 else 1 / 1.1)
                self._commit_live_if_active()

    def _move_vertex(self, idx, px, py):
        """Move vertex idx to pixel position (px, py), rebuilding the MirrorTube."""
        n = len(self._custom_mirrors.mirrors)
        pts = [m.p0[:2].copy() for m in self._custom_mirrors.mirrors]
        pts[idx] = self._px_to_norm(px, py)
        try:
            mirrors = [Mirror(pts[i], pts[(i + 1) % n]) for i in range(n)]
            self._custom_mirrors = self._make_edit_tube(mirrors)
        except ValueError:
            pass  # revert silently if validation fails

    def _translate_all(self, px, py):
        """Translate all vertices by the delta from the drag anchor."""
        delta = self._px_to_norm(px, py) - self._translate_anchor
        n = len(self._translate_base_pts)
        new_pts = [p + delta for p in self._translate_base_pts]
        try:
            mirrors = [Mirror(new_pts[i], new_pts[(i + 1) % n]) for i in range(n)]
            self._custom_mirrors = self._make_edit_tube(mirrors)
        except ValueError:
            pass

    def _scale_mirrors(self, factor):
        """Scale all vertices uniformly about the origin; revert if validation fails."""
        pts = [m.p0[:2] * factor for m in self._custom_mirrors.mirrors]
        n = len(pts)
        old = self._custom_mirrors
        try:
            self._custom_mirrors = self._make_edit_tube([Mirror(pts[i], pts[(i + 1) % n]) for i in range(n)])
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
            self._custom_mirrors = self._make_edit_tube(mirrors)
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
            self._custom_mirrors = self._make_edit_tube(mirrors)
        except ValueError:
            pass

    def _draw_mirror_editor(self, frame_out, overlay=False):
        """
        Draw mirror control points and lines using the same coordinate space as the rendering
        (natural coords mapped to the full output frame via x_max/y_max from compute_fov).
        When overlay=True (live editing), draw directly on the rendered image without a background.
        """
        if self._custom_mirrors is None:
            self._custom_mirrors = self._get_mirrors_to_customize()

        h, w = frame_out.shape[:2]
        x_max, y_max = compute_fov(w, h)

        # Use the full output frame for coordinate mapping so mirror positions
        # coincide exactly with the rendered kaleidoscope geometry.
        self._draw_x0, self._draw_x1 = 0, w
        self._draw_y0, self._draw_y1 = 0, h

        if not overlay:
            cv2.rectangle(frame_out, (UI_MARGIN_PX, UI_MARGIN_PX),
                          (w - UI_MARGIN_PX, h - UI_MARGIN_PX), UI_BKG, -1)
            cv2.rectangle(frame_out, (UI_MARGIN_PX, UI_MARGIN_PX),
                          (w - UI_MARGIN_PX, h - UI_MARGIN_PX), COLORS['gray'], 1)

        def to_px(pt):
            px = int((pt[0] + x_max) / (2.0 * x_max) * w)
            py = int((y_max - pt[1]) / (2.0 * y_max) * h)
            return (px, py)

        mirrors = self._custom_mirrors.mirrors
        ctrl_pts_px = [to_px(m.p0) for m in mirrors]
        mx, my = self.mouse_pos
        n = len(ctrl_pts_px)
        interaction_active = (
            self._drag_kind is not None
            or self.mouse_dragging
            or self._translating
            or self._pan_drag_last is not None
        )

        if interaction_active:
            # Keep the interaction latched while dragging/translating/panning.
            if self.mouse_dragging and self.selected_ctrl_pt is not None:
                mouseover_idx = self.selected_ctrl_pt
                closest_idx = self.selected_ctrl_pt
                prev_idx = (closest_idx - 1) % n
                next_idx = (closest_idx + 1) % n
                second_idx = self._second_idx if self._second_idx is not None else next_idx
                self._second_idx = second_idx
                self._mouse_in_interior = False
            elif self._translating:
                mouseover_idx = None
                second_idx = self._second_idx if self._second_idx is not None else 0
                self._mouse_in_interior = True
            else:
                mouseover_idx = None
                second_idx = self._second_idx if self._second_idx is not None else 0
                self._mouse_in_interior = False
            self._hover_idx = mouseover_idx
        else:
            # Find closest vertex to mouse, then pick the angularly closer neighbor as secondary
            dists = [np.hypot(p[0] - mx, p[1] - my) for p in ctrl_pts_px]
            closest_idx = int(np.argmin(dists))
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

        poly_verts = np.array([m.p0[:2] for m in mirrors])

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

        show_ui_geometry = (not overlay) or self._show_ctrl_pts
        if show_ui_geometry:
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
            return self._draw_mirror_editor(frame_out, overlay=False)
        elif self.mode == UIModes.LIVE_EDITING:
            return self._draw_mirror_editor(frame_out, overlay=True)
        else:
            raise ValueError(f"Invalid UI mode: {self.mode}")

    def draw_hotkey_help(self, frame):
        """Draw a hotkey-help overlay on frame in-place."""
        h, w = frame.shape[:2]
        inset = 4 * COL_MARGIN_PX
        bx0, by0 = inset, inset
        bx1, by1 = w - inset, h - inset
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), COLORS['off-white'], -1)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), BKG, 1)

        font = TITLE_FONT
        color = BKG
        thick = 1
        aa = cv2.LINE_AA
        tx0 = bx0 + TITLE_INDENT_PX
        key_x = tx0 + TITLE_INDENT_PX
        title_text = "Kaleidoscopy!"
        subtitle_text = "(2026) github:andsmith/kaleidoscopy"
        tc = HELP_FONT_SCALES['text']
        sc = HELP_FONT_SCALES['section']

        (title_size, _) = cv2.getTextSize(title_text, font, HELP_FONT_SCALES['title'], thick)
        (subtitle_size, _) = cv2.getTextSize(subtitle_text, font, HELP_FONT_SCALES['subtitle'], thick)
        max_section_w = 0
        max_key_w = 0
        max_desc_w = 0
        for section_name, entries in _HELP_SECTIONS:
            (section_size, _) = cv2.getTextSize(section_name, font, sc, thick)
            max_section_w = max(max_section_w, section_size[0])
            for key_str, desc_str in entries:
                (key_size, _) = cv2.getTextSize(key_str, font, tc, thick)
                (desc_size, _) = cv2.getTextSize(desc_str, font, tc, thick)
                max_key_w = max(max_key_w, key_size[0])
                max_desc_w = max(max_desc_w, desc_size[0])

        desc_x = key_x + max_key_w + TITLE_INDENT_PX
        left_text_right = max(
            tx0 + title_size[0],
            tx0 + subtitle_size[0],
            tx0 + max_section_w,
            key_x + max_key_w,
            desc_x + max_desc_w,
        )
        divider_x = min(bx1 - TITLE_INDENT_PX, left_text_right + TITLE_INDENT_PX)
        left_inner_x1 = divider_x - TITLE_INDENT_PX
        y = by0 + TITLE_INDENT_PX

        def _put_centered(text, scale, gap_after=4):
            nonlocal y
            (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
            x = tx0 + (left_inner_x1 - tx0 - tw) // 2
            cv2.putText(frame, text, (x, y + th), font, scale, color, thick, aa)
            y += th + bl + gap_after

        _put_centered(title_text, HELP_FONT_SCALES['title'], gap_after=6)
        _put_centered(subtitle_text, HELP_FONT_SCALES['subtitle'], gap_after=20)

        def _section_height(section_name, entries):
            (_, sh), sbl = cv2.getTextSize(section_name, font, sc, thick)
            height = sh + sbl + 2 + 8
            for key_str, desc_str in entries:
                (_, kh), kbl = cv2.getTextSize(key_str, font, tc, thick)
                height += kh + kbl + 4
            return height

        section_heights = [_section_height(section_name, entries) for section_name, entries in _HELP_SECTIONS]
        available_section_height = max(0, (by1 - TITLE_INDENT_PX) - y)
        total_section_height = sum(section_heights)
        n_section_gaps = max(0, len(_HELP_SECTIONS) - 1)
        extra_gap = 0.0 if n_section_gaps == 0 else max(0.0, (available_section_height - total_section_height) / n_section_gaps)

        for section_idx, (section_name, entries) in enumerate(_HELP_SECTIONS):
            section_size, sbl = cv2.getTextSize(section_name, font, sc, thick)
            sw = section_size[0]
            sh = section_size[1]
            cv2.putText(frame, section_name, (tx0, y + sh), font, sc, color, thick, aa)
            y += sh + sbl + 2
            cv2.line(frame, (tx0, y), (tx0 + sw, y), color, 1, aa)
            y += 8

            for key_str, desc_str in entries:
                (_, kh), kbl = cv2.getTextSize(key_str, font, tc, thick)
                cv2.putText(frame, key_str, (key_x, y + kh), font, tc, color, thick, aa)
                cv2.putText(frame, desc_str, (desc_x, y + kh), font, tc, color, thick, aa)
                y += kh + kbl + 4
            if section_idx < len(_HELP_SECTIONS) - 1:
                y += int(round(extra_gap))


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

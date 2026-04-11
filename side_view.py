"""
For debugging, this shows orthographic projections of the mirrors/rays:
 it shows 3 views:
   * an XZ and YZ view of the world (2 views)
    - Eye at the top and target plane at the bottom. Dotted lines drawn from eye to target plane, indicating field of view
    angles for the 2 side views, and a dotted bounding box to indicate in the top-down view.
    - All lines representing mirrors will be drawn faintly, the most extreme on the left and right will be drawn
      darker.
   * A top-down view with the rays drawn as lines instead of the rendered map, etc.
   * dotted lines are drawn at the boundaries of the output on all views (in the target plane) (should be at the +/-
     1.0 in natural coords)


A subset of rays from the actual raytracing are rendered in all 3 views(i.e. the same subset of rays is shown in all 3 views).
The subset of rays is chosen to be evenly distributed across the raytracing output, with the following rules:
    * All 4 corner rays are always included in the subset.
    * The horizontal and vertical sampling rates are equal so that the rays are evenly distributed across the
        side-view plots/images.
    * n_rays parameter controls this, defining the number of rays along the longer dimension of the ray
      subset grid, (i.e. if n_rays=10 and the aspect ratio is 2:1 then the subset grid will be 10 x 5 rays).

This module will query the raytracer for the information it needs (i.e. can ask for past states).

Rays can be drawn in the following modes:
    * "start":  a small dot is drawn at the ray's origin and a constant length line points in its direction (like a vector field)
    * "hit":  A small red dot is drawn at each ray's origin, and a green dot is drawn at it's hit point.
       a thin line is drawn between the origin and hit point.
    * "bounce": Like "hit" but also shows the directions of the rays after each bounce (or a heavier dot if hitting
       the target plane), as in the "start" mode.   (i.e. a combination of the "start" and "hit" modes, but with the
       ray directions after each bounce shown as the vector field).


The entire image will be divided into 4 sections, horizontally in half and vertically into an upper/lower portions that
are 1/3rd and 2/3rds the height respectively:
    * The smaller left (top) portion will be for text (status--see below)
    * The smaller right (top) portion will be for the top-down view.
    * The larger left (bottom) portion will show the XZ view.
    * The larger right (bottom) will show the  YZ view.

There will be a MARGIN_PX sized margin between each section.  The entire image should be initialized off-white,
each plot will be on a geom::BKG colored background (i.e. its entire bounding box).
"""

from geom import COLORS, BKG
import cv2
import numpy as np
import logging
import time

EYE_COLOR = COLORS['light_blue']

# Layout
MARGIN_PX = 12
VERT_SPLIT_REL = 0.5

# Colors
FOV_LINE_COLOR         = COLORS['gray']
TARGET_LINE_COLOR      = COLORS['red']
IMAGE_PLANE_COLOR      = COLORS['yellow']
IMAGE_FOOTPRINT_COLOR  = COLORS['gray']   # image-border footprint in the top-down panel
MIRROR_FAINT_ALPHA = 0.35
MIRROR_DARK_ALPHA  = 0.85

# Ray drawing
DOT_RADIUS           = 3
VECTOR_STEP          = 0.06   # natural-coord length for direction arrows (deprecated in favor of UNIT_VEC_LEN_PX)
UNIT_VEC_LEN_PX      = 30     # fixed pixel length for direction arrows, independent of zoom
N_RAYS_DEFAULT       = 10
ARROW_TIP_RATIO      = 0.25    # arrowhead length as fraction of shaft length
ARROW_TIP_ANGLE_DEG  = 30.0   # full opening angle of arrowhead barbs


# ---------------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------------

def _compute_panels(img_w, img_h, margin):
    """Return {name: {'x0','y0','x1','y1'}} bounding boxes for each panel."""
    mx = img_w // 2
    my = int(img_h * VERT_SPLIT_REL)
    return {
        'topdown': {'x0': margin,      'y0': margin,      'x1': mx - margin,      'y1': my - margin},
        'yz':      {'x0': mx + margin, 'y0': margin,      'x1': img_w - margin,   'y1': my - margin},
        'xz':      {'x0': margin,      'y0': my + margin, 'x1': mx - margin,      'y1': img_h - margin},
        'bounce':  {'x0': mx + margin, 'y0': my + margin, 'x1': img_w - margin,   'y1': img_h - margin},
    }


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Zoom state
# ---------------------------------------------------------------------------

class ZoomState:
    """Persistent zoom for one panel: scale factor + center in natural coords."""

    ZOOM_FACTOR = 1.25    # zoom change per scroll click
    MIN_SCALE   = 0.02
    MAX_SCALE   = 500.0

    def __init__(self):
        self.scale    = 1.0
        self.center_x = None   # None → use default panel center
        self.center_y = None

    def reset(self):
        self.scale    = 1.0
        self.center_x = None
        self.center_y = None


class PanelTransform:
    """Maps a 2D natural-coordinate rectangle into pixel coords within a panel.

    Supports per-panel zoom (ZoomState).  All natural-coord quantities are
    expressed relative to the *default* (unzoomed) range; zoom simply shifts
    and scales that range while keeping the zoom center fixed.
    """

    def __init__(self, nat_x_min, nat_x_max, nat_y_min, nat_y_max, panel, flip_y,
                 zoom=None, keep_aspect=False):
        """
        :param nat_x_min/max: natural coord range along horizontal panel axis (unzoomed)
        :param nat_y_min/max: natural coord range along vertical panel axis (unzoomed)
        :param panel: dict with x0, y0, x1, y1
        :param flip_y: if True, nat_y_max → panel top; False → nat_y_min → top
        :param zoom: ZoomState (optional); None means no zoom
        :param keep_aspect: if True, expand whichever axis needs it so that
               pixels-per-natural-unit is identical in X and Y.  This preserves
               the world's aspect ratio regardless of panel shape or zoom level.
        """
        # Default (unzoomed) center and half-ranges – kept for zoom math
        self._def_cx = (nat_x_min + nat_x_max) / 2.0
        self._def_cy = (nat_y_min + nat_y_max) / 2.0
        self._def_hx = (nat_x_max - nat_x_min) / 2.0
        self._def_hy = (nat_y_max - nat_y_min) / 2.0

        # Apply zoom to get the effective visible natural range
        if zoom is not None and zoom.scale != 1.0:
            cx = zoom.center_x if zoom.center_x is not None else self._def_cx
            cy = zoom.center_y if zoom.center_y is not None else self._def_cy
            hx = self._def_hx / zoom.scale
            hy = self._def_hy / zoom.scale
        else:
            cx, cy = self._def_cx, self._def_cy
            hx, hy = self._def_hx, self._def_hy

        self._px0 = panel['x0']
        self._py0 = panel['y0']
        self._px1 = panel['x1']
        self._py1 = panel['y1']
        self._flip_y = flip_y

        # Aspect-ratio correction: choose a single pixels-per-unit so the
        # world fits in the panel without distortion, expanding the shown nat
        # range on whichever axis has excess panel space.
        if keep_aspect:
            pw = self._px1 - self._px0
            ph = self._py1 - self._py0
            ppu = min(pw / (2.0 * hx), ph / (2.0 * hy))  # pixels per nat unit
            hx = pw / (2.0 * ppu)
            hy = ph / (2.0 * ppu)

        self._nxmin = cx - hx
        self._nxmax = cx + hx
        self._nymin = cy - hy
        self._nymax = cy + hy

    def to_px(self, nx, ny):
        """Map natural coords (nx, ny) to pixel (px, py).

        Returns raw (possibly out-of-panel) coordinates — callers must check
        ``in_panel`` / ``clip_line`` before drawing to avoid spill.
        """
        tx = (nx - self._nxmin) / (self._nxmax - self._nxmin)
        ty = (ny - self._nymin) / (self._nymax - self._nymin)
        px = int(self._px0 + tx * (self._px1 - self._px0))
        if self._flip_y:
            py = int(self._py1 - ty * (self._py1 - self._py0))
        else:
            py = int(self._py0 + ty * (self._py1 - self._py0))
        return px, py

    def in_panel(self, px, py):
        """Return True if pixel (px, py) is inside this panel's bounding box."""
        return self._px0 <= px <= self._px1 and self._py0 <= py <= self._py1

    def clip_line(self, pt1, pt2):
        """Liang-Barsky clip of segment pt1→pt2 to this panel's rectangle.

        Returns ((x1,y1),(x2,y2)) of the visible portion, or None if entirely
        outside the panel.
        """
        x1, y1 = float(pt1[0]), float(pt1[1])
        x2, y2 = float(pt2[0]), float(pt2[1])
        dx, dy = x2 - x1, y2 - y1
        p = [-dx, dx, -dy, dy]
        q = [x1 - self._px0, self._px1 - x1, y1 - self._py0, self._py1 - y1]
        u1, u2 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if abs(pi) < 1e-10:
                if qi < 0:
                    return None          # parallel and outside
            elif pi < 0:
                u1 = max(u1, qi / pi)
            else:
                u2 = min(u2, qi / pi)
        if u1 > u2:
            return None
        return ((int(x1 + u1 * dx), int(y1 + u1 * dy)),
                (int(x1 + u2 * dx), int(y1 + u2 * dy)))

    def from_px(self, px, py):
        """Map pixel (px, py) to natural coords (nx, ny)  [inverse of to_px]."""
        tx = (px - self._px0) / (self._px1 - self._px0)
        ty = (py - self._py0) / (self._py1 - self._py0)
        nx = self._nxmin + tx * (self._nxmax - self._nxmin)
        if self._flip_y:
            ny = self._nymax - ty * (self._nymax - self._nymin)
        else:
            ny = self._nymin + ty * (self._nymax - self._nymin)
        return nx, ny


# ---------------------------------------------------------------------------
# Side-view state  (zoom + mode + n_rays – persistent across renders)
# ---------------------------------------------------------------------------

class SideViewState:
    """
    Persistent per-session state for the side view: mode, ray count, and
    per-panel zoom.  Attach to a cv2 window via ``cv2.setMouseCallback``.

    Usage::

        state = SideViewState()
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, state.mouse_callback)

        while True:
            img = render_side_view(rt, cur_size, state=state)
            cv2.imshow(win, img)
            ...
    """

    def __init__(self, ray_mode='bounce', show_trails=False, show_start=False, n_rays=N_RAYS_DEFAULT):
        self.ray_mode    = ray_mode     # 'bounce', 'hit', or 'off'
        self.show_trails = show_trails  # draw full path history
        self.show_start  = show_start   # draw initial ray vectors
        self.n_rays = n_rays
        self._zooms = {k: ZoomState() for k in ('topdown', 'xz', 'yz')}
        # Set by render_side_view so the mouse callback has current geometry
        self._last_panels     = None
        self._last_transforms = None
        # Drag-to-pan state
        self._drag_panel      = None   # panel key being dragged
        self._drag_start_px   = None   # (x, y) pixel where drag began
        self._drag_tf         = None   # PanelTransform snapshot at drag start
        self._drag_orig_cx    = None   # zoom.center_x at drag start
        self._drag_orig_cy    = None   # zoom.center_y at drag start

    def reset_zoom(self, panel_key=None):
        """Reset zoom for one panel (or all if panel_key is None)."""
        keys = [panel_key] if panel_key else list(self._zooms)
        for k in keys:
            self._zooms[k].reset()

    def update_last_render(self, panels, transforms):
        """Called by render_side_view to cache panel geometry for hit-testing."""
        self._last_panels     = panels
        self._last_transforms = transforms

    def mouse_callback(self, event, x, y, flags, param):
        """cv2 mouse callback.  Register with ``cv2.setMouseCallback``."""
        if self._last_panels is None:
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            for key in ('topdown', 'xz', 'yz'):
                r = self._last_panels[key]
                if r['x0'] <= x < r['x1'] and r['y0'] <= y < r['y1']:
                    tf = self._last_transforms.get(key)
                    if tf is None:
                        return
                    nat_x, nat_y = tf.from_px(x, y)
                    zoom = self._zooms[key]
                    factor = (ZoomState.ZOOM_FACTOR if flags > 0
                              else 1.0 / ZoomState.ZOOM_FACTOR)
                    old_scale = zoom.scale
                    new_scale = float(np.clip(old_scale * factor,
                                             ZoomState.MIN_SCALE, ZoomState.MAX_SCALE))
                    cx = zoom.center_x if zoom.center_x is not None else tf._def_cx
                    cy = zoom.center_y if zoom.center_y is not None else tf._def_cy
                    zoom.center_x = nat_x - (nat_x - cx) * old_scale / new_scale
                    zoom.center_y = nat_y - (nat_y - cy) * old_scale / new_scale
                    zoom.scale    = new_scale
                    break

        elif event == cv2.EVENT_LBUTTONDOWN:
            for key in ('topdown', 'xz', 'yz'):
                r = self._last_panels[key]
                if r['x0'] <= x < r['x1'] and r['y0'] <= y < r['y1']:
                    tf = self._last_transforms.get(key)
                    if tf is None:
                        return
                    zoom = self._zooms[key]
                    self._drag_panel    = key
                    self._drag_start_px = (x, y)
                    self._drag_tf       = tf
                    self._drag_orig_cx  = zoom.center_x if zoom.center_x is not None else tf._def_cx
                    self._drag_orig_cy  = zoom.center_y if zoom.center_y is not None else tf._def_cy
                    break

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self._drag_panel is None:
                return
            tf   = self._drag_tf
            zoom = self._zooms[self._drag_panel]
            # Compute the natural-coord delta by inverting the start transform
            nat_start = tf.from_px(*self._drag_start_px)
            nat_curr  = tf.from_px(x, y)
            zoom.center_x = self._drag_orig_cx + (nat_start[0] - nat_curr[0])
            zoom.center_y = self._drag_orig_cy + (nat_start[1] - nat_curr[1])

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_panel    = None
            self._drag_start_px = None
            self._drag_tf       = None
            self._drag_orig_cx  = None
            self._drag_orig_cy  = None


def _build_transforms(params, panels, pad=0.12, zooms=None):
    x_max  = params['x_max']
    y_max  = params['y_max']
    targ_z = params['targ_z']
    xr = x_max  * (1 + pad)
    yr = y_max  * (1 + pad)
    zr = targ_z * (1 + pad * 0.5)
    zs = zooms or {}
    return {
        # XY top-down: x horizontal, y vertical, math orientation (y up)
        # keep_aspect=True so the render aspect ratio is preserved on zoom/resize
        'topdown': PanelTransform(-xr, xr, -yr, yr, panels['topdown'],
                                  flip_y=True,  zoom=zs.get('topdown'), keep_aspect=True),
        # XZ side: x horizontal, z vertical (z=0 eye at top, z=targ_z at bottom)
        'xz':      PanelTransform(-xr, xr,  0,  zr, panels['xz'],
                                  flip_y=False, zoom=zs.get('xz')),
        # YZ side (flipped): z horizontal, y vertical (y up)
        'yz':      PanelTransform(0,  zr, -yr, yr, panels['yz'],
                      flip_y=True, zoom=zs.get('yz')),
    }


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_dotted_line(img, pt1, pt2, color, dash=6, gap=4, thickness=1, clip_tf=None):
    """Draw a dotted line.  If clip_tf is a PanelTransform the segment is first
    clipped to that panel's bounding box; nothing is drawn if fully outside."""
    if clip_tf is not None:
        result = clip_tf.clip_line(pt1, pt2)
        if result is None:
            return
        pt1, pt2 = result
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    length = np.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    t = 0.0
    drawing = True
    while t < length:
        if drawing:
            t_end = min(t + dash, length)
            p0 = (int(x1 + dx * t),    int(y1 + dy * t))
            p1 = (int(x1 + dx * t_end), int(y1 + dy * t_end))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
            t = t_end
        else:
            t += gap
        drawing = not drawing


def _arrow_endpoint_px_fixed_len(origin_3d, direction_3d, axes, tf, len_px):
    """Compute arrow endpoint for fixed pixel length, independent of zoom.
    
    Projects the 3D direction to 2D screen space, normalizes it, and scales to len_px pixels.
    This ensures direction arrows remain the same visual size when zoomed.
    
    :param origin_3d: 3D origin point
    :param direction_3d: 3D direction vector
    :param axes: projection axes ('xy', 'xz', or 'yz')
    :param tf: PanelTransform for the panel
    :param len_px: desired arrow length in pixels
    :return: (x, y) endpoint in pixel coordinates
    """
    o_px = tf.to_px(*_project(origin_3d, axes))
    
    # Compute screen-space direction by projecting two points along the ray.
    # Use a large probe distance (1.0) for numerical stability, especially when zoomed out.
    # The exact distance doesn't matter—we normalize by it anyway.
    probe_pt = origin_3d + direction_3d * 1.0
    probe_px = tf.to_px(*_project(probe_pt, axes))
    
    # Get direction in pixel space
    dx = float(probe_px[0] - o_px[0])
    dy = float(probe_px[1] - o_px[1])
    dist = np.hypot(dx, dy)
    
    if dist < 1e-6:
        # Direction is essentially zero in screen space, return origin
        return o_px
    
    # Normalize and scale to fixed pixel length
    scale = len_px / dist
    end_px = (int(o_px[0] + dx * scale), int(o_px[1] + dy * scale))
    return end_px


def _draw_arrow(img, pt1, pt2, color, thickness=1,
                tip_ratio=ARROW_TIP_RATIO, tip_angle_deg=ARROW_TIP_ANGLE_DEG,
                clip_tf=None):
    """Draw an anti-aliased line from pt1 to pt2 with an arrowhead at pt2.

    :param tip_ratio:     arrowhead barb length as a fraction of shaft length
    :param tip_angle_deg: full opening angle of arrowhead barbs in degrees
    :param clip_tf:       PanelTransform for clipping; segment dropped if fully outside
    """
    if clip_tf is not None:
        result = clip_tf.clip_line(pt1, pt2)
        if result is None:
            return
        pt1, pt2 = result
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    shaft_len = np.hypot(x2 - x1, y2 - y1)
    if shaft_len < 1:
        return
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    # Unit vector pointing from pt2 back toward pt1
    back_x = (x1 - x2) / shaft_len
    back_y = (y1 - y2) / shaft_len
    barb_len  = shaft_len * tip_ratio
    half_rad  = np.radians(tip_angle_deg / 2.0)
    cos_a, sin_a = np.cos(half_rad), np.sin(half_rad)
    for sign in (+1, -1):
        # Rotate back-vector by ±half_angle
        bx = back_x * cos_a - sign * back_y * sin_a
        by = sign * back_x * sin_a + back_y * cos_a
        tip = (int(x2 + bx * barb_len), int(y2 + by * barb_len))
        cv2.line(img, (int(x2), int(y2)), tip, color, thickness, cv2.LINE_AA)


def _alpha_color(base_color, alpha):
    return tuple(int(c * alpha) for c in base_color)


# ---------------------------------------------------------------------------
# Mirror drawing
# ---------------------------------------------------------------------------

def _extreme_mirror_indices(mirrors, axis):
    """
    Return (min_idx, max_idx) for the mirrors whose endpoint has the smallest /
    largest coordinate on `axis` (0=x, 1=y).
    """
    vals = [min(m.p0[axis], m.p1[axis]) for m in mirrors]
    maxvals = [max(m.p0[axis], m.p1[axis]) for m in mirrors]
    return int(np.argmin(vals)), int(np.argmax(maxvals))


def _draw_mirror_lines_xy(img, mirrors, tf):
    """Draw mirror segments in the XY top-down panel."""
    color = _alpha_color(COLORS['white'], MIRROR_DARK_ALPHA)
    thickness = 2
    for m in mirrors:
        p0 = tf.to_px(m.p0[0], m.p0[1])
        p1 = tf.to_px(m.p1[0], m.p1[1])
        seg = tf.clip_line(p0, p1)
        if seg:
            cv2.line(img, seg[0], seg[1], color, thickness, cv2.LINE_AA)


def _draw_mirror_lines_side(img, mirrors, tf, axis, line_orientation='vertical'):
    """
    Draw mirror projections in a side panel (XZ or YZ).
    Mirrors are vertical planes; depending on panel orientation they render as
    either vertical or horizontal lines. `axis` is 0 for XZ (use x),
    1 for YZ (use y).
    """
    # Identify extreme endpoints for the given axis
    all_coords = [m.p0[axis] for m in mirrors] + [m.p1[axis] for m in mirrors]
    c_min, c_max = min(all_coords), max(all_coords)

    # Get targ_z from the transform's y-natural range (nat_y_max ≈ targ_z * (1+pad))
    # We draw from z=0 to the full panel height; use an explicitly large z value.
    # Use the transform to map z=0 and z far enough:
    seen = set()
    for m in mirrors:
        for coord in [m.p0[axis], m.p1[axis]]:
            coord = round(coord, 9)
            if coord in seen:
                continue
            seen.add(coord)
            is_extreme = (np.isclose(coord, c_min) or np.isclose(coord, c_max))
            color = _alpha_color(COLORS['white'], MIRROR_DARK_ALPHA if is_extreme else MIRROR_FAINT_ALPHA)
            thickness = 2 if is_extreme else 1
            if line_orientation == 'vertical':
                # Constant horizontal coord, spanning the panel's vertical axis.
                pt_a = tf.to_px(coord, 0.0)
                pt_b = tf.to_px(coord, 1e9)
            else:
                # Constant vertical coord, spanning the panel's horizontal axis.
                pt_a = tf.to_px(0.0, coord)
                pt_b = tf.to_px(1e9, coord)
            seg = tf.clip_line(pt_a, pt_b)
            if seg:
                cv2.line(img, seg[0], seg[1], color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# FOV boundary lines
# ---------------------------------------------------------------------------

def _draw_fov_xz(img, x_max, img_z, targ_z, tf):
    """FOV boundary lines and plane markers in the XZ panel."""
    eye = tf.to_px(0, 0)
    s = img_z / targ_z

    _draw_dotted_line(img, eye, tf.to_px(-x_max, targ_z), FOV_LINE_COLOR, clip_tf=tf)
    _draw_dotted_line(img, eye, tf.to_px( x_max, targ_z), FOV_LINE_COLOR, clip_tf=tf)

    _draw_dotted_line(img, tf.to_px(-x_max * s, img_z), tf.to_px(x_max * s, img_z),
                      IMAGE_PLANE_COLOR, thickness=2, clip_tf=tf)

    _draw_dotted_line(img, tf.to_px(-x_max, targ_z), tf.to_px(x_max, targ_z),
                      TARGET_LINE_COLOR, clip_tf=tf)


def _draw_fov_yz(img, y_max, img_z, targ_z, tf):
    """FOV boundary lines and plane markers in the YZ panel."""
    eye = tf.to_px(0, 0)
    s = img_z / targ_z

    _draw_dotted_line(img, eye, tf.to_px(targ_z, -y_max), FOV_LINE_COLOR, clip_tf=tf)
    _draw_dotted_line(img, eye, tf.to_px(targ_z,  y_max), FOV_LINE_COLOR, clip_tf=tf)

    _draw_dotted_line(img, tf.to_px(img_z, -y_max * s), tf.to_px(img_z, y_max * s),
                      IMAGE_PLANE_COLOR, thickness=2, clip_tf=tf)

    _draw_dotted_line(img, tf.to_px(targ_z, -y_max), tf.to_px(targ_z, y_max),
                      TARGET_LINE_COLOR, clip_tf=tf)


def _draw_fov_box_topdown(img, x_max, y_max, tf):
    """Dotted rectangle showing the target-plane boundary in the XY top-down panel."""
    corners = [(-x_max, -y_max), (x_max, -y_max), (x_max, y_max), (-x_max, y_max)]
    px_corners = [tf.to_px(x, y) for x, y in corners]
    for i in range(4):
        _draw_dotted_line(img, px_corners[i], px_corners[(i + 1) % 4],
                          TARGET_LINE_COLOR, clip_tf=tf)


def _draw_image_footprint_topdown(img, x_max, y_max, tf):
    """Dotted rectangle for target image boundaries in the XY top-down panel.

    This is the aspect-ratio-constrained target-image region inside the full
    FOV box: an inscribed box with half-extent min(x_max, y_max).  It touches
    the red FOV boundary on either top/bottom or left/right, depending on
    output aspect ratio.
    """
    h = min(x_max, y_max)
    ix, iy = h, h
    corners = [(-ix, -iy), (ix, -iy), (ix, iy), (-ix, iy)]
    px_corners = [tf.to_px(x, y) for x, y in corners]
    for i in range(4):
        _draw_dotted_line(img, px_corners[i], px_corners[(i + 1) % 4],
                          IMAGE_FOOTPRINT_COLOR, clip_tf=tf)


def _draw_eye(img, transforms):
    """Draw the eye position (z=0) as a dot in XZ and YZ side panels."""
    for key in ('xz', 'yz'):
        tf = transforms[key]
        pt = tf.to_px(0, 0)
        if tf.in_panel(*pt):
            cv2.circle(img, pt, DOT_RADIUS + 3, EYE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(img, pt, DOT_RADIUS + 5, EYE_COLOR, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Ray subset selection
# ---------------------------------------------------------------------------

def select_ray_subset(ray_grid_shape, n_rays):
    """
    Select indices into the flattened ray array for visualization.

    :param ray_grid_shape: (n_rows, n_cols)
    :param n_rays: number of rays along the longer dimension
    :returns: 1D int array of flat indices
    """
    n_rows, n_cols = ray_grid_shape
    if n_cols >= n_rows:
        n_x = n_rays
        n_y = max(2, round(n_rays * n_rows / n_cols))
    else:
        n_y = n_rays
        n_x = max(2, round(n_rays * n_cols / n_rows))

    row_inds = np.round(np.linspace(0, n_rows - 1, n_y)).astype(int)
    col_inds = np.round(np.linspace(0, n_cols - 1, n_x)).astype(int)
    row_inds = np.unique(np.concatenate([[0, n_rows - 1], row_inds]))
    col_inds = np.unique(np.concatenate([[0, n_cols - 1], col_inds]))

    grid_r, grid_c = np.meshgrid(row_inds, col_inds, indexing='ij')
    flat = (grid_r * n_cols + grid_c).flatten()
    return np.unique(flat)


# ---------------------------------------------------------------------------
# Ray drawing
# ---------------------------------------------------------------------------

def _project(pt3, axes):
    """Project a 3D point to 2D natural coords for the given panel axes."""
    if axes == 'xy':
        return pt3[0], pt3[1]
    elif axes == 'xz':
        return pt3[0], pt3[2]
    elif axes == 'yz':
        return pt3[2], pt3[1]


_PANEL_AXES = {'topdown': 'xy', 'xz': 'xz', 'yz': 'yz'}


def _draw_rays(img, mode, subset_inds, origins, init_dirs, bounces, transforms):
    """Draw the ray subset in all three panels according to the given mode.

    Modes:
      'start'  – dot + arrow at each ray's current position/direction.
                 Uses the last bounce's step_origins/step_dirs, or the
                 initial rays when no bounces exist yet.
      'hit'    – segment from step_origins → hit_origins for the latest
                 bounce, with endpoint dots.  Nothing drawn if no bounces.
      'bounce' – like 'hit' plus a direction arrow at hit_origins showing
                 each mirror-hit ray's post-bounce direction (hit_dirs).
      'trails' – all bounce segments faintly; last bounce segment bright.

    All primitives are clipped so zoomed views never spill into adjacent panels.
    """
    # When bounce records are compact (Raytracer stores only display_inds rows),
    # map each global ray index to its local position in the stored arrays.
    if bounces and 'display_inds' in bounces[-1]:
        _disp = bounces[-1]['display_inds']
        _g2l = {int(g): l for l, g in enumerate(_disp)}
        def _li(idx): return _g2l[idx]
    else:
        def _li(idx): return idx

    for panel_name, axes in _PANEL_AXES.items():
        tf = transforms[panel_name]

        for idx in subset_inds:

            if mode == 'start':
                if bounces:
                    li = _li(idx)
                    cur_o = bounces[-1]['step_origins'][li]
                    cur_d = bounces[-1]['step_dirs'][li]
                else:
                    cur_o = origins[idx]
                    cur_d = init_dirs[idx]
                o_px = tf.to_px(*_project(cur_o, axes))
                if tf.in_panel(*o_px):
                    cv2.circle(img, o_px, DOT_RADIUS - 1, EYE_COLOR, -1, cv2.LINE_AA)
                e_px = _arrow_endpoint_px_fixed_len(cur_o, cur_d, axes, tf, UNIT_VEC_LEN_PX)
                _draw_arrow(img, o_px, e_px, COLORS['cyan'], 1, clip_tf=tf)

            elif mode == 'hit':
                if not bounces:
                    continue
                step = bounces[-1]
                li = _li(idx)
                o_px = tf.to_px(*_project(step['step_origins'][li], axes))
                h_px = tf.to_px(*_project(step['hit_origins'][li], axes))
                if tf.in_panel(*o_px):
                    cv2.circle(img, o_px, DOT_RADIUS, COLORS['red'], -1, cv2.LINE_AA)
                seg = tf.clip_line(o_px, h_px)
                if seg:
                    cv2.line(img, seg[0], seg[1], COLORS['gray'], 1, cv2.LINE_AA)
                if tf.in_panel(*h_px):
                    color = COLORS['blue'] if step['hit_mirror'][li] else COLORS['green']
                    r = DOT_RADIUS + 2 if step['hit_target'][li] else DOT_RADIUS
                    cv2.circle(img, h_px, r, color, -1, cv2.LINE_AA)

            elif mode == 'bounce':
                if not bounces:
                    continue
                step = bounces[-1]
                li = _li(idx)
                o_px = tf.to_px(*_project(step['step_origins'][li], axes))
                h_px = tf.to_px(*_project(step['hit_origins'][li], axes))
                if tf.in_panel(*o_px):
                    cv2.circle(img, o_px, DOT_RADIUS, COLORS['red'], -1, cv2.LINE_AA)
                seg = tf.clip_line(o_px, h_px)
                if seg:
                    cv2.line(img, seg[0], seg[1], COLORS['gray'], 1, cv2.LINE_AA)
                if step['hit_mirror'][li]:
                    if tf.in_panel(*h_px):
                        cv2.circle(img, h_px, DOT_RADIUS, COLORS['blue'], -1, cv2.LINE_AA)
                    hit_d = step['hit_dirs'][li]
                    e_px = _arrow_endpoint_px_fixed_len(step['hit_origins'][li], hit_d, axes, tf, UNIT_VEC_LEN_PX)
                    _draw_arrow(img, h_px, e_px, COLORS['cyan'], 1, clip_tf=tf)
                elif step['hit_target'][li]:
                    if tf.in_panel(*h_px):
                        cv2.circle(img, h_px, DOT_RADIUS + 2, COLORS['green'], -1, cv2.LINE_AA)

            elif mode == 'trails':
                if not bounces:
                    continue
                for k, step in enumerate(bounces):
                    li = _li(idx)
                    is_last = (k == len(bounces) - 1)
                    alpha     = 0.40
                    thickness = 1
                    color = tuple(int(c * alpha) for c in COLORS['white'])
                    so_px = tf.to_px(*_project(step['step_origins'][li], axes))
                    ho_px = tf.to_px(*_project(step['hit_origins'][li], axes))
                    seg = tf.clip_line(so_px, ho_px)
                    if seg:
                        cv2.line(img, seg[0], seg[1], color, thickness, cv2.LINE_AA)
                    if is_last:
                        if step['hit_mirror'][li] and tf.in_panel(*ho_px):
                            cv2.circle(img, ho_px, DOT_RADIUS, COLORS['green'], -1, cv2.LINE_AA)
                        elif step['hit_target'][li] and tf.in_panel(*ho_px):
                            cv2.circle(img, ho_px, DOT_RADIUS + 2, COLORS['red'], -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Text panel
# ---------------------------------------------------------------------------

def _draw_text_panel(img, panel, params, mode, n_rays, t_elapsed=None):
    r = panel
    cv2.rectangle(img, (r['x0'], r['y0']), (r['x1'], r['y1']), COLORS['off-white'], -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    color = COLORS['black']
    thickness = 1
    line_h = 18
    x0 = r['x0'] + 8
    y0 = r['y0'] + 18

    n_mirrors = len(params['mirrors']) if params['mirrors'] else 0
    img_z     = params['img_z']
    lines = [
        "kaleidoscopy / side_view",
        f"mode:    {mode}",
        f"mirrors: {n_mirrors}",
        f"x_max:   {params['x_max']:.4f}",
        f"y_max:   {params['y_max']:.4f}",
        f"img_z:   {img_z:.6f}" if img_z is not None else "img_z:   N/A",
        f"targ_z:  {params['targ_z']:.3f}",
        f"n_rays:  {n_rays}",
    ]
    if t_elapsed is not None:
        lines.append(f"render:  {t_elapsed * 1000:.1f} ms")

    for line in lines:
        if y0 >= r['y1'] - 4:
            break
        cv2.putText(img, line, (x0, y0), font, scale, color, thickness)
        y0 += line_h


# ---------------------------------------------------------------------------
# Bounce map panel
# ---------------------------------------------------------------------------

def _draw_bounce_map(img, panel, bounce_count):
    """
    Render the bounce-count heatmap into `panel` (the top-left area).

    bounce_count[y, x] == -1  →  medium gray  (ray still active / not yet resolved)
    bounce_count[y, x] >= 0   →  viridis color scaled to [0, max_bounces]
                                  (0 bounces = darkest purple; max = bright yellow)
    bounce_count is None       →  uniform dark background with a label
    """
    px0, py0 = panel['x0'], panel['y0']
    px1, py1 = panel['x1'], panel['y1']
    pw, ph = px1 - px0, py1 - py0
    if pw <= 0 or ph <= 0:
        return

    cv2.rectangle(img, (px0, py0), (px1, py1), BKG, -1)

    if bounce_count is None:
        _draw_panel_label(img, panel, "bounce map  (N/A)")
        return

    valid = bounce_count >= 0
    if not np.any(valid):
        _draw_panel_label(img, panel, "bounce map  (pending...)")
        return

    max_b = int(bounce_count[valid].max())

    # Normalise to uint8: 0 bounces → 0 (dark/purple), max_b → 255 (bright/yellow)
    if max_b == 0:
        norm = np.zeros(bounce_count.shape, dtype=np.uint8)
    else:
        norm = np.clip(bounce_count.astype(np.float32) / max_b * 255,
                       0, 255).astype(np.uint8)

    # Apply viridis colormap (outputs BGR)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

    # Override still-active pixels with a neutral gray
    colored[~valid] = (100, 100, 100)

    # Resize to fill the panel (nearest-neighbour keeps discrete color bands crisp)
    panel_img = cv2.resize(colored, (pw, ph), interpolation=cv2.INTER_NEAREST)
    img[py0:py1, px0:px1] = panel_img

    _draw_panel_label(img, panel, f"bounce map  (max={max_b})")


# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------

def _draw_panel_label(img, panel, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (panel['x0'] + 4, panel['y0'] + 14),
                font, 0.4, COLORS['gray'], 1)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_side_view(raytracer, img_size, state=None, mode='bounce', n_rays=N_RAYS_DEFAULT):
    """
    Render the 4-panel side-view debug image.

    :param raytracer: object supporting get_ray_params, get_initial_rays,
                      get_bounces, and get_map
    :param img_size:  (width, height) of the output debug image
    :param state:     SideViewState (optional).  When provided, ``mode`` and
                      ``n_rays`` are taken from the state, and zoom + last-render
                      geometry are stored in it for the mouse callback.
    :param mode:      'start', 'hit', 'bounce', or 'trails'  (ignored when state is given)
    :param n_rays:    rays along longer grid dimension (ignored when state is given)
    :returns: (h, w, 3) uint8 numpy array
    """
    if state is not None:
        ray_mode    = state.ray_mode
        show_trails = state.show_trails
        show_start  = state.show_start
        n_rays      = state.n_rays
    else:
        # Legacy mode string → new independent flags
        ray_mode    = mode if mode in ('bounce', 'hit') else 'off'
        show_trails = (mode == 'trails')
        show_start  = (mode == 'start')

    t0 = time.time()
    img_w, img_h = img_size

    img = np.full((img_h, img_w, 3), COLORS['off-white'], dtype=np.uint8)

    panels = _compute_panels(img_w, img_h, MARGIN_PX)

    for key in ('topdown', 'xz', 'yz'):
        r = panels[key]
        cv2.rectangle(img, (r['x0'], r['y0']), (r['x1'], r['y1']), BKG, -1)

    params  = raytracer.get_ray_params()
    origins, init_dirs = raytracer.get_initial_rays()
    bounces = raytracer.get_bounces()

    t_elapsed = None

    if params['mirrors'] and origins is not None:
        zooms = state._zooms if state is not None else None
        tfs   = _build_transforms(params, panels, zooms=zooms)

        if state is not None:
            state.update_last_render(panels, tfs)

        x_max  = params['x_max']
        y_max  = params['y_max']
        img_z  = params['img_z']
        targ_z = params['targ_z']
        mirrors = params['mirrors']

        _draw_mirror_lines_xy(img, mirrors, tfs['topdown'])
        _draw_mirror_lines_side(img, mirrors, tfs['xz'], axis=0, line_orientation='vertical')
        _draw_mirror_lines_side(img, mirrors, tfs['yz'], axis=1, line_orientation='horizontal')

        _draw_fov_xz(img, x_max, img_z, targ_z, tfs['xz'])
        _draw_fov_yz(img, y_max, img_z, targ_z, tfs['yz'])
        _draw_fov_box_topdown(img, x_max, y_max, tfs['topdown'])
        _draw_image_footprint_topdown(img, x_max, y_max, tfs['topdown'])

        _draw_eye(img, tfs)

        if params['ray_grid_shape']:
            if bounces and 'display_inds' in bounces[-1]:
                # Compact bounce records: indices and arrays are already subsampled.
                subset = bounces[-1]['display_inds']
            else:
                subset = select_ray_subset(params['ray_grid_shape'], n_rays)
            display_mask = params.get('display_mask')
            if display_mask is not None:
                subset = subset[display_mask[subset]]
            if show_trails:
                _draw_rays(img, 'trails', subset, origins, init_dirs, bounces, tfs)
            if show_start:
                _draw_rays(img, 'start', subset, origins, init_dirs, bounces, tfs)
            if ray_mode != 'off':
                _draw_rays(img, ray_mode, subset, origins, init_dirs, bounces, tfs)

        t_elapsed = time.time() - t0

    for label, key in [('XY top-down', 'topdown'), ('XZ side', 'xz'), ('YZ side', 'yz')]:
        _draw_panel_label(img, panels[key], label)

    _, bounce_count = raytracer.get_map()
    _draw_bounce_map(img, panels['bounce'], bounce_count)

    return img


# ---------------------------------------------------------------------------
# Test / entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from mirror_configs import PresetFactory
    import init_rays
    from raytracing import FakeRaytracer
    from geom import TARG_Z

    OUT_SIZE = (1280, 720)
    PRESET   = 'equilateral triangle'

    logging.info("Creating mirror tube: %s", PRESET)
    tube = PresetFactory.make_preset(PRESET, r=0.4)
    w, h = OUT_SIZE
    x_max, y_max = init_rays.compute_fov(w, h)
    img_z = init_rays.find_img_z(w, h, tube.mirrors, x_max, y_max)
    logging.info("img_z=%.6f  x_max=%.4f  y_max=%.4f", img_z, x_max, y_max)

    rt = FakeRaytracer(OUT_SIZE, mirrors=tube.mirrors,
                       x_max=x_max, y_max=y_max, img_z=img_z, targ_z=TARG_Z)

    state  = SideViewState()
    window = "side_view  [b=bounce/hit/off  t=trails  s=start  r=reset zoom  scroll=zoom  q=quit]"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, *OUT_SIZE)
    cv2.setMouseCallback(window, state.mouse_callback)

    cur_size = OUT_SIZE
    while True:
        img = render_side_view(rt, cur_size, state=state)
        cv2.imshow(window, img)
        k = cv2.waitKey(30)
        if k in (ord('q'), 27):
            break
        elif k == ord('b'):
            modes = ['bounce', 'hit', 'off']
            state.ray_mode = modes[(modes.index(state.ray_mode) + 1) % 3]
        elif k == ord('s'):
            state.show_start = not state.show_start
        elif k == ord('t'):
            state.show_trails = not state.show_trails
        elif k == ord('r'):
            state.reset_zoom()

        # Detect window resize and re-render at the new size next frame
        rect = cv2.getWindowImageRect(window)
        if rect[2] > 10 and rect[3] > 10:
            new_size = (rect[2], rect[3])
            if new_size != cur_size:
                logging.info("Window resized to %dx%d", *new_size)
                cur_size = new_size

    cv2.destroyAllWindows()

"""
Start the Kalleidoscope app:
1.  First pick a mirror configuration with TAB, then hit SPACE to run.
2.  This starts the renderer/mapper (will display partial results as they are computed).
3.  Hit space again to bring up the UI layer and pick another mirror config.
4.  Right click to manually adjust the mirror configuration.

Input source:
   * Run with no argument to use the webcam as the input.
   * Run with an image as the input argument to focus the kalleidoscope on that image. (pan and zoom in next version)

"""

from raytracing import FakeRaytracer, Raytracer
from init_rays import compute_fov, find_img_z
from side_view import SideViewState, render_side_view
import numpy as np
import cv2
import logging
import time
import sys
import argparse
from scipy.ndimage import maximum_filter, minimum_filter
from geom import TARG_Z, COLORS, BKG
from user_interface import UIModes, UILayer

_FADE_FACTORS = [1.0, 0.99, 0.95, 0.90, 0.75, 0.50]
_STAIN_DISTANCES = [0, 1, 3]
_STAIN_THRESHOLDS = [1.0, 2.0, 3.0, 5.0]  # multiples of neighborhood diagonal

WINDOW_NAME = "Kaleidoscopy!"
SIDE_VIEW_WINDOW = "Kaleidoscopy - Side View  [ENTER=step  b=bounce/hit  s/t=mode  r=reset zoom  scroll=zoom  d=close]"
GLASS_BORDER_COLOR = np.array((50, 50, 50), dtype=np.uint8)  # dark gray for stained-glass leading edges

class ScopeApp(object):
    def __init__(self, output_size, input_size=None, input_img=None, debug=False, interpolate=False,
                 camera_index=0, camera_res=None, n_cpu=1):
        """
        :param output_size: the size of the output video stream (width, height)
        :param input_size: the size of the input video stream (width, height), ignored if input_img is provided
        :param input_img: if provided, this image will be used instead of the webcam feed. Should be a numpy array.
        :param debug: if True, shows side-view window, steps with Enter, uses non-threaded raytracer.
        :param interpolate: if True, use bilinear interpolation in the map lookup (cv2.remap); default is nearest-neighbour.
        :param camera_index: camera device index passed to cv2.VideoCapture (default 0).
        :param camera_res: (width, height) to request from the camera; defaults to output_size.
        :param n_cpu: number of CPU cores to use for raytracing (default 1 = single-core).
        """
        self._debug = debug
        self._interpolate = interpolate
        self._bkg = input_img
        self.out_size = output_size
        self._camera_index = camera_index
        self._camera_res = camera_res
        self._n_cpu = n_cpu
        self.in_size = input_size if input_img is None else (input_img.shape[1], input_img.shape[0])
        self._frame_out = np.zeros((self.out_size[1], self.out_size[0], 3), dtype=np.uint8)
        self._f_no = 0
        self._running = False
        self._hotkey_help_showing = False
        self._mirrors = None

        self._init_input()
        self._last_input_frame = self._bkg   # updated each loop; used by UI icon
        self._fake_raytracer = FakeRaytracer(output_size)
        self._raytracer = None
        self._raytrace_done = True
        self._init_ui()

        # Art layer state
        self._fade_layer = None    # (h, w) float32 multiplier per pixel
        self._stain_mask = None    # (h, w) bool — True = leading (black)
        self._last_raytrace_step = -1  # detect raytracer progress for layer invalidation
        self._last_fade_idx = -1   # detect index change to force rebuild
        self._last_stain_idx = -1
        self._last_stain_thresh_idx = -1

        self._side_view_open = False  # opened in start() if debug mode
        self._side_view_state = SideViewState()
        self._side_view_size = output_size

    @property
    def mirrors(self):
        return self._mirrors

    def shutdown(self):
        self._running = False

    def set_mirrors_and_restart(self, new_mirrors, preserve_old_map=False):
        logging.info("Setting mirrors: %s", new_mirrors)

        w_out, h_out = self.out_size
        x_max, y_max = compute_fov(w_out, h_out)
        try:
            img_z = find_img_z(w_out, h_out, new_mirrors.mirrors, x_max, y_max)
        except ValueError as e:
            logging.error("Could not compute img_z: %s", e)
            return False

        # Commit new geometry only after all dependent parameters are valid.
        self._mirrors = new_mirrors

        # Reset view and art layers for fresh config (skip view reset during live edits)
        if not preserve_old_map:
            self._ui_layer.reset_view()
        self._last_raytrace_step = -1
        self._last_fade_idx = -1
        self._last_stain_idx = -1
        self._last_stain_thresh_idx = -1

        # Recreate FakeRaytracer with geometry so side_view shows real ray paths
        self._fake_raytracer = FakeRaytracer(
            self.out_size, mirrors=new_mirrors.mirrors,
            x_max=x_max, y_max=y_max, img_z=img_z, targ_z=TARG_Z,
        )

        # Stop any running background thread before replacing the raytracer.
        if self._raytracer is not None:
            self._raytracer.stop()

        # Optionally preserve the existing float map so live edits don't flash to identity.
        initial_map = None
        if preserve_old_map and self._raytracer is not None:
            old_map, _ = self._raytracer.get_map()
            if old_map is not None:
                initial_map = old_map.copy()   # (h, w, 2) float32

        # In debug mode use non-threaded stepping; otherwise run full-speed threaded.
        self._raytracer = Raytracer(
            self.out_size, new_mirrors.mirrors, TARG_Z, x_max, y_max, img_z,
            threaded=not self._debug,
            initial_map=initial_map,
            n_workers=self._n_cpu,
        )
        if self._debug:
            self._raytracer._init_rays()   # debug: manual stepping via Enter key
        else:
            self._raytracer.start()        # non-debug: launches background thread
        self._raytrace_done = False
        logging.info("Raytracer initialized. img_z=%.6f, x_max=%.3f, y_max=%.3f", img_z, x_max, y_max)
        return True

    def _init_ui(self):
        self._ui_layer = UILayer(self, window_name=WINDOW_NAME)

    # ------------------------------------------------------------------
    # View transform (pan/zoom)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Art layers
    # ------------------------------------------------------------------

    def _rebuild_fade_layer(self, bounce_count):
        factor = _FADE_FACTORS[self._ui_layer.fade_idx]
        bc = np.maximum(bounce_count, 0).astype(np.float32)
        self._fade_layer = np.power(factor, bc)  # shape (h, w)

    def _rebuild_stain_mask(self, float_map, bounce_count):
        """
        A pixel is 'leading' (metal frame) if EITHER of the two border conditions holds
        (combined with logical OR), using the same neighborhood for both:

        1. Source-distance border: any neighbor within output-space distance d originated
           from a source point more than (threshold * diag) natural-coord units away
           (raw float_map coords, before any pan/zoom).
        2. Bounce-count border: any neighbor within output-space distance d has a different
           bounce count.
        """
        dist = _STAIN_DISTANCES[self._ui_layer.stain_idx]
        if dist == 0:
            self._stain_mask = None
            return
        size = 2 * dist + 1

        # --- Border method 1: source-distance threshold ---
        threshold_scale = _STAIN_THRESHOLDS[self._ui_layer.stain_thresh_idx]
        # Convert threshold from "pixels" to natural-coord units (one output pixel ≡ 2*x_max/w_out)
        pixel_size_nat = 2.0 * compute_fov(*self.out_size)[0] / self.out_size[0]
        diag_nat = np.sqrt(2.0) * (2 * dist) * pixel_size_nat
        threshold = threshold_scale * diag_nat

        xm = float_map[0]   # natural x coords, (h, w) float32 contiguous
        ym = float_map[1]   # natural y coords, (h, w) float32 contiguous

        max_x = maximum_filter(xm, size=size)
        min_x = minimum_filter(xm, size=size)
        max_y = maximum_filter(ym, size=size)
        min_y = minimum_filter(ym, size=size)

        x_dist = np.maximum(max_x - xm, xm - min_x)
        y_dist = np.maximum(max_y - ym, ym - min_y)
        src_dist_border = np.sqrt(x_dist ** 2 + y_dist ** 2) > threshold

        # --- Border method 2: differing bounce count ---
        bc = bounce_count.astype(np.float32)
        max_bc = maximum_filter(bc, size=size)
        min_bc = minimum_filter(bc, size=size)
        bounce_border = (max_bc > bc) | (min_bc < bc)

        self._stain_mask = src_dist_border | bounce_border

    def _apply_art_layers(self, frame_out, bounce_count, float_map):
        """Apply fade and stained-glass layers to frame_out in-place."""
        if bounce_count is None:
            return

        # Detect raytracer progress via step index
        step = self._raytracer._step_index if self._raytracer is not None else 0
        bounce_changed = step != self._last_raytrace_step
        self._last_raytrace_step = step

        fade_idx = self._ui_layer.fade_idx
        stain_idx = self._ui_layer.stain_idx
        stain_thresh_idx = self._ui_layer.stain_thresh_idx
        fade_changed = fade_idx != self._last_fade_idx
        stain_changed = stain_idx != self._last_stain_idx or stain_thresh_idx != self._last_stain_thresh_idx
        self._last_fade_idx = fade_idx
        self._last_stain_idx = stain_idx
        self._last_stain_thresh_idx = stain_thresh_idx

        # Rebuild fade layer when needed
        if fade_idx > 0 and (fade_changed or bounce_changed or self._fade_layer is None):
            self._rebuild_fade_layer(bounce_count)

        # Rebuild stain mask when needed (uses raw float_map, not view-transformed)
        if stain_idx > 0 and (stain_changed or bounce_changed or self._stain_mask is None):
            self._rebuild_stain_mask(float_map, bounce_count)

        # Apply fade
        if fade_idx > 0 and self._fade_layer is not None:
            frame_out[:] = (frame_out * self._fade_layer[..., np.newaxis]).clip(0, 255).astype(np.uint8)

        # Apply stained glass
        if stain_idx > 0 and self._stain_mask is not None:
            frame_out[self._stain_mask] = GLASS_BORDER_COLOR

    def _make_img_frame(self, img):
        """
        Fit the image centered and as large as possible into out_size, filling the rest with BKG.
        """
        out_frame = (np.zeros((self.out_size[1], self.out_size[0], 3)) + BKG).astype(np.uint8)
        in_h, in_w = img.shape[:2]
        out_w, out_h = self.out_size
        in_aspect = in_w / in_h
        out_aspect = out_w / out_h

        if in_aspect > out_aspect:
            new_w = out_w
            new_h = int(out_w / in_aspect)
        else:
            new_h = out_h
            new_w = int(out_h * in_aspect)

        resized_img = cv2.resize(img, (new_w, new_h))
        x_offset = (out_w - new_w) // 2
        y_offset = (out_h - new_h) // 2
        out_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
        return out_frame

    def _init_input(self):
        if self._bkg is not None:
            logging.info("Using %i x %i image instead of camera frames." % (self._bkg.shape[1], self._bkg.shape[0]))
            self._cam = None
            self.in_size = (self._bkg.shape[1], self._bkg.shape[0])
        else:
            logging.info("Starting camera %d...", self._camera_index)
            self._cam = cv2.VideoCapture(self._camera_index)
            cam_w, cam_h = self._camera_res or self.out_size
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
            height = int(self._cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(self._cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            logging.info("Camera resolution: %i x %i" % (width, height))
            self.in_size = (width, height)
        self._f_no = 0

    def _get_img(self):
        """Return the current input image (from camera or static image)."""
        if self._cam is None:
            return self._bkg.copy()
        ret, frame = self._cam.read()
        if not ret:
            logging.warning("Failed to read frame.")
            return None
        return frame

    def _open_side_view(self):
        cv2.namedWindow(SIDE_VIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(SIDE_VIEW_WINDOW, *self.out_size)
        cv2.setMouseCallback(SIDE_VIEW_WINDOW, self._side_view_state.mouse_callback)
        self._side_view_open = True
        self._side_view_size = self.out_size
        logging.info("Side view opened: ENTER advances one raytrace step.")

    @staticmethod
    def _is_enter_key(k):
        """Return True when the key code corresponds to Enter."""
        return k in (10, 13)

    def _close_side_view(self):
        cv2.destroyWindow(SIDE_VIEW_WINDOW)
        self._side_view_open = False

    def _handle_side_view_key(self, k):
        if k == ord('b'):
            modes = ['bounce', 'hit', 'off']
            st = self._side_view_state
            st.ray_mode = modes[(modes.index(st.ray_mode) + 1) % 3]
        elif k == ord('s'):
            self._side_view_state.show_start = not self._side_view_state.show_start
        elif k == ord('t'):
            self._side_view_state.show_trails = not self._side_view_state.show_trails
        elif k == ord('r'):
            self._side_view_state.reset_zoom()
        elif k == ord('d'):
            self._close_side_view()

    def _tick_side_view(self):
        rect = cv2.getWindowImageRect(SIDE_VIEW_WINDOW)
        if rect[2] > 10 and rect[3] > 10:
            self._side_view_size = (rect[2], rect[3])
        tracer_for_view = self._raytracer if self._raytracer is not None else self._fake_raytracer
        sv_img = render_side_view(tracer_for_view, self._side_view_size,
                                  state=self._side_view_state)
        cv2.imshow(SIDE_VIEW_WINDOW, sv_img)

    def _step_raytrace_once(self):
        """Advance one raytracing step (used when side-view is open)."""
        if self._raytracer is None or self._raytrace_done:
            logging.info("Raytracing already complete.")
            return
        has_more = self._raytracer.step(record_bounces=True)
        self._raytrace_done = not has_more
        if self._raytrace_done:
            logging.info("Raytracing complete.")

    def start(self):
        self._running = True
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *self.out_size)

        if self._debug:
            self._open_side_view()

        # FPS tracking
        _fps_interval = 5.0
        _fps_last_log = time.monotonic()
        _fps_frame_count = 0
        _fps_ui_time = 0.0
        _fps_render_time = 0.0
        _fps_idle_time = 0.0

        while self._running:
            input_frame = self._get_img()
            # if _fps_frame_count % 10 == 0:
            #     print("input frame shape:", input_frame.shape if input_frame is not None else None)
            if input_frame is None:
                time.sleep(0.1)
                continue
            self._last_input_frame = input_frame

            _t_render_start = time.monotonic()

            if self._mirrors is not None:
                float_map, bounce_count = self._raytracer.get_map() if self._raytracer else (None, None)

                if float_map is not None:
                    src_h, src_w = input_frame.shape[:2]
                    pan_x, pan_y, zoom = self._ui_layer.view_transform
                    x_map, y_map, oob_mask = self._raytracer.get_integer_map(
                        src_w, src_h, pan_x, pan_y, zoom
                    )
                    interp = cv2.INTER_LINEAR if self._interpolate else cv2.INTER_NEAREST
                    frame_out = cv2.remap(input_frame, x_map, y_map, interp)
                    if oob_mask is not None:
                        frame_out[oob_mask] = BKG
                    self._apply_art_layers(frame_out, bounce_count, float_map)
                    self._frame_out = frame_out
                else:
                    self._frame_out = self._fake_raytracer.render()
            else:
                # No mirrors yet: show raw input as background
                self._frame_out = self._make_img_frame(input_frame)

            _t_ui_start = time.monotonic()
            _fps_render_time += _t_ui_start - _t_render_start

            self._ui_layer.draw_layer(self._frame_out)
            if self._hotkey_help_showing:
                self._ui_layer.draw_hotkey_help(self._frame_out)
            
            self._f_no += 1

            _t_display_start = time.monotonic()
            _fps_ui_time += _t_display_start - _t_ui_start

            cv2.imshow(WINDOW_NAME, self._frame_out)

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break  # window was closed by the user

            if self._side_view_open:
                self._tick_side_view()

            k = cv2.waitKey(1) & 0xFF
            _fps_idle_time += time.monotonic() - _t_display_start

            if k == 27:  # ESC — quit immediately, bypassing UI state machine
                break

            if k in (ord('h'), ord('H')):
                self._hotkey_help_showing = not self._hotkey_help_showing

            if self._debug and self._side_view_open:
                if self._is_enter_key(k):
                    self._step_raytrace_once()
                    continue
                self._handle_side_view_key(k)

            if not self._ui_layer.handle_keypress(k):
                break

            _fps_frame_count += 1
            _now = time.monotonic()
            _elapsed = _now - _fps_last_log
            if _elapsed >= _fps_interval:
                _mean_fps = _fps_frame_count / _elapsed
                _ui_ms = _fps_ui_time / _fps_frame_count * 1000 if _fps_frame_count else 0
                _render_ms = _fps_render_time / _fps_frame_count * 1000 if _fps_frame_count else 0
                _idle_ms = _fps_idle_time / _fps_frame_count * 1000 if _fps_frame_count else 0
                if not self._debug:
                    _rt_active = (self._raytracer is not None
                                  and self._raytracer._thread is not None
                                  and self._raytracer._thread.is_alive())
                    logging.info(
                        "FPS %.1f | render %.1f ms | UI %.1f ms | idle %.1f ms | raytracer active: %s",
                        _mean_fps, _render_ms, _ui_ms, _idle_ms, _rt_active,
                    )
                else:
                    logging.info(
                        "FPS %.1f | render %.1f ms | UI %.1f ms | idle %.1f ms | debug (manual stepping)",
                        _mean_fps, _render_ms, _ui_ms, _idle_ms,
                    )
                w_out, h_out = self.out_size
                # logging.info("output resolution: %d x %d, %.2f MPix", w_out, h_out, w_out * h_out / 1e6)
                _fps_last_log = _now
                _fps_frame_count = 0
                _fps_ui_time = 0.0
                _fps_render_time = 0.0
                _fps_idle_time = 0.0

        if self._cam is not None:
            self._cam.release()
        cv2.destroyAllWindows()


_RESOLUTION_PRESETS = {1: (640, 480), 2: (800, 600), 3: (1024, 768), 4: (1280, 1024), 5: (1920, 1080)}
_PRESET_HELP = "1=640x480  2=800x600  3=1024x768  4=1280x1024  5=1920x1080"


def _parse_res(vals, flag, parser):
    """Parse a resolution argument: 1 int → preset lookup, 2 ints → (width, height)."""
    if len(vals) == 1:
        if vals[0] not in _RESOLUTION_PRESETS:
            parser.error("%s: preset must be 1–5 (%s)" % (flag, _PRESET_HELP))
        return _RESOLUTION_PRESETS[vals[0]]
    if len(vals) == 2:
        if any(v < 1 for v in vals):
            parser.error("%s: width and height must be positive integers" % flag)
        return (vals[0], vals[1])
    parser.error("%s: expected 1 preset number or 2 integers (width height)" % flag)


def start_scope():
    parser = argparse.ArgumentParser(
        description="Kaleidoscopy — live kaleidoscope using mirrors and a camera or image.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("image", nargs="?",
                        help="Input image path. Omit to use the webcam.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Debug mode: show side-view, step with Enter, non-threaded raytracer.")
    parser.add_argument("-i", "--interpolate", action="store_true",
                        help="Use bilinear interpolation in map lookup (default: nearest-neighbour).")
    parser.add_argument("-r", "--res", nargs="+", type=int, metavar=("PRESET", "H"),
                        help=("Output window resolution.\n"
                              "  One int  : preset number — %s\n"
                              "  Two ints : custom width height  (e.g. -r 1280 720)\n"
                              "Default: 5 (1920x1080)." % _PRESET_HELP))
    parser.add_argument("-cr", "--camera-res", nargs="+", type=int, metavar=("PRESET", "H"),
                        help=("Camera input resolution (same format as -r).\n"
                              "Defaults to the output resolution when not specified."))
    parser.add_argument("-c", "--camera", type=int, default=0, metavar="INDEX",
                        help="Camera device index (default: 0).")
    parser.add_argument("-n", "--n_cpu", type=int, default=1, metavar="N",
                        help="Number of CPU cores for raytracing (default: 1 = single-core).")
    args = parser.parse_args()

    if args.image is not None:
        img = cv2.imread(args.image)
        if img is None:
            logging.error("Could not read image: %s", args.image)
            sys.exit(1)
    else:
        img = None

    out_size = _parse_res(args.res, "-r/--res", parser) if args.res else (1920, 1080)
    cam_res = _parse_res(args.camera_res, "-cr/--camera-res", parser) if args.camera_res else None

    app = ScopeApp(out_size, input_img=img, debug=args.debug, interpolate=args.interpolate,
                   camera_index=args.camera, camera_res=cam_res, n_cpu=args.n_cpu)

    iw, ih = app.in_size
    ow, oh = app.out_size
    src = ("image '%s' (%dx%d)" % (args.image, iw, ih)) if args.image else ("camera %d (%dx%d)" % (args.camera, iw, ih))
    print("Input  : %s" % src)
    print("Output : %dx%d" % (ow, oh))
    print("CPUs   : %d" % args.n_cpu)
    print("Interp : %s" % ("bilinear" if args.interpolate else "nearest-neighbour"))
    print("Debug  : %s" % ("on" if args.debug else "off"))

    app.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_scope()

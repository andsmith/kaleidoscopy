#import pylab as plt
import numpy as np
import cv2
import logging
import time
# from util import make_bounds, TextManager, Image
from threading import Lock, Event, enumerate, get_ident, Thread
from ray_tracing import  RayTracer
from prisms import  PRISMS
from enum import Enum
from gui_utils.gui_picker import ChooseItemDialog
from gui_utils.camera import Camera, pick_camera
from gui_utils.text_annotation import StatusMessages
from error_handling import ShutdownException
from rendering import ImageMapper


class KScopeState(Enum):
    setup = 0
    running = 1
    shaping = 2

    shutdown = 100


class KScopyApp(object):
    MIRROR_TYPES = PRISMS
    _OSD_TEXT_COLOR = (255, 254, 250)
    _OSD_BKG_COLOR = (118, 100, 90)
    _OSD_ALPHA = 0.65
    _FPS_PERIOD_SEC = 2.0

    def __init__(self):
        self._in_frame = None
        self._in_frame_time = None
        self._out_frame = None
        self._out_frame_time = None
        self._last_frame_time = None
        self._icon_size = 200
        self._window_name = "Kaleidoscopy"
        self._state = KScopeState.setup
        self._app_flow_lock = Lock()
        self._waiter = Event()
        self._OSD_on = True
        self._k_map = None  # stores current input-output mapping of pixels
        self._render_stats = None
        self._raytracing_stats = None

        self._hotkeys = [{'key': ' ', 'desc': 'Help', 'func': self._toggle_osd,
                          'txt': "SPACE - toggle this on-screen display."},

                         {'key': 'q', 'desc': 'Quit', 'func': self._shutdown,
                          'txt': "Q - Quit."}]

        self._fps = {'n_in': 0,
                     'n_out': 0,
                     'n_dropped': 0,
                     'last_update_time': time.time() - self._FPS_PERIOD_SEC - 1.0,
                     'fps_in': 0,
                     'fps_out': 0,
                     'fps_drop': 0}

        self._drop_frame_lock = Lock()

        # start
        try:
            if False:  # TEMP DEBUG
                with self._app_flow_lock:
                    self._cam_ind = pick_camera()
                    self._cam = Camera(self._cam_ind, self._proc_frame, prompt_resolution=True)
            else:
                self._cam_ind = 0
                self._cam = Camera(self._cam_ind, self._proc_frame, prompt_resolution=False)

            # User chooses type of scope
            self._ray_tracer = None
            self._mirror_type = self._user_pick_shape()
            self._mirrors = self._mirror_type()
            self._cam.start()
            self._resolution = self._cam.get_resolution(wait=True)
            image_shape = (self._resolution[1], self._resolution[0], 3)
            self._status_bar = StatusMessages(image_shape, self._OSD_TEXT_COLOR,
                                              self._OSD_BKG_COLOR,
                                              bkg_alpha=self._OSD_ALPHA, spacing=10, max_font_scale=1.0)
            self._renderer = ImageMapper(self._resolution)

            # User shapes mirror parameters
            self._user_shape_mirrors()
            self._ray_tracer = RayTracer(self._mirrors, self._resolution, self._update_k_map)

            # Start raytracing and rendering
            self._start()

        except ShutdownException:
            print("App Shutdown - by exception")
            pass

    def _update_k_map(self, img_map, stats):
        self._k_map = img_map
        self._render_stats = "Ray-tracing:  %i of %i rays hit image in at most %i bounces." % (
            stats['rays_hit'], stats['n_rays'], stats['n_bounces'])

    def _toggle_osd(self):
        self._OSD_on = not self._OSD_on

    def _start(self):
        print("Starting scope...")
        self._state = KScopeState.running

        # set main help-text
        self._status_bar.clear()
        self._status_bar.add_msgs([k['txt'] for k in self._hotkeys], "Help", duration_sec=0)

        # begin
        self._ray_tracer.start()


    def _shutdown(self):
        self._state = KScopeState.shutdown
        cv2.destroyAllWindows()
        self._cam.shutdown()
        if self._ray_tracer is not None:
            self._ray_tracer.shutdown()
        print("Main._shutdown() done.")

    def _wait(self):
        logging.info("Waiting...")
        self._waiter.clear()
        while self._state != KScopeState.shutdown:
            if self._waiter.wait(timeout=0.1):
                break
        logging.info("Done waiting.")
        if self._state == KScopeState.shutdown:
            raise ShutdownException("App shut down during wait.")

    def _user_shape_mirrors(self):
        """
        User sets parameters of selected mirror arrangement.
        """
        self._state = KScopeState.shaping  # now camera frames go to shaping method of prism
        shaping_instructions, mouse_callback = self._mirrors.start_shaping(self._window_name, self._waiter)
        self._pending_mouse_callback = mouse_callback
        self._status_bar.clear()
        self._status_bar.add_msgs(shaping_instructions, "shaping")

        self._wait()  # clears wait after user accepts?
        logging.info("Done waiting for shaping")
        # clear callback
        cv2.setMouseCallback(self._window_name, lambda *args: None)

    def _user_pick_shape(self):
        """
        User selects type of mirror arrangement.
        """
        icons = [prism_type.get_icon(self._icon_size) for prism_type in KScopyApp.MIRROR_TYPES]
        if len(icons) == 4:  # Need good way to arrange these automatically...
            icons = [icons[:2], icons[2:]]
        choice_ind = ChooseItemDialog(prompt="Select mirror Geometry:").ask_icons(icons)
        return KScopyApp.MIRROR_TYPES[choice_ind]

    def _proc_frame(self, in_frame, frame_time):
        self._fps['n_in'] += 1
        if self._drop_frame_lock.acquire(blocking=False):
            self._proc_frame_helper(in_frame, frame_time)
            self._drop_frame_lock.release()
        else:
            logging.warning("DROP")
            self._fps['n_dropped'] += 1

    def _proc_frame_helper(self, in_frame, frame_time):
        """
        Frame processing callback.
        Can't throw shutdown exception, because this isn't the main thread.
        """

        self._in_frame = in_frame
        self._in_frame_time = frame_time

        # PROCESS
        if self._state == KScopeState.setup:
            return

        elif self._state == KScopeState.shaping:
            out_frame = self._mirrors.get_masked_image(in_frame)
            self._status_bar.annotate_img(out_frame)

        elif self._state == KScopeState.running:
            out_frame = self._render(in_frame)
            self._update_osd_info()
            if self._OSD_on:
                self._status_bar.annotate_img(out_frame)

        elif self._state == KScopeState.shutdown:
            return

        else:
            raise Exception("Unknown app state:  %s" % (self._state,))

        # DISPLAY
        self._out_frame = out_frame
        self._out_frame_time = time.time()
        cv2.imshow(self._window_name, self._out_frame)
        k = cv2.waitKey(1)
        self._fps['n_out'] += 1

        # KEYBOARD
        if ord('q') == k & 0xff:
            self._shutdown()  # catch-all

        if self._state == KScopeState.running:
            for key in self._hotkeys:
                if ord(key['key']) == k & 0xff:
                    key['func']()

        elif self._state == KScopeState.shaping:
            self._mirrors.handle_keyboard_adjust(k)

        # Needs to be done after window is created (alt idea: named window)
        if self._pending_mouse_callback is not None:
            cv2.setMouseCallback(self._window_name, self._pending_mouse_callback)
            self._pending_mouse_callback = None

    def _render(self, frame):
        if self._k_map is None:
            self._ray_tracer.get_current_map()
        return frame.copy()

    def _update_osd_info(self):
        """
            On-screen display components:
                * Hotkeys
                * Ray-tracing status
                * Input-FPS:  output-FPS:  dropped frames:
        """
        elapsed = time.time() - self._fps['last_update_time']

        if elapsed > self._FPS_PERIOD_SEC:
            self._fps['fps_in'] = self._fps['n_in'] / elapsed
            self._fps['fps_out'] = self._fps['n_out'] / elapsed
            self._fps['fps_drop'] = self._fps['n_dropped'] / elapsed

            self._fps['n_in'] = 0
            self._fps['n_out'] = 0
            self._fps['n_dropped'] = 0

            self._status_bar.add_msg("FPS:  in = %.3f,  out = %.3f,  drop = %i" % (
                self._fps['fps_in'], self._fps['fps_out'],
                self._fps['fps_drop']), "fps")
            self._fps['last_update_time'] = time.time()

        if self._raytracing_stats is not None:
            self._status_bar.add_msg(self._raytracing_stats, "raytracing_stats", duration_sec=5.0)

        if self._render_stats is not None:
            self._status_bar.add_msg(self._render_stats, "render_stats")
        else:
            self._status_bar.remove_msg('render_stats')


def thread_monitor():
    start = time.time()
    while start + 5.0 > time.time():
        time.sleep(1)
        for thread in enumerate():
            print("%s - %s - %s" % (get_ident(), thread.ident, thread.name))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # t1 = Thread(target=thread_monitor)
    # t1.start()
    scope = KScopyApp()
    # _test_kscope_diagrams()
    # scope.view_live(0)
    # scope.view_image(cv2.imread('test_img.jpg'))

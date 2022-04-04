import pylab as plt
import numpy as np
import cv2
import logging
import time
from util import make_bounds, TextManager, Image
from threading import Lock, Event, enumerate, get_ident, Thread
from ray_tracing import RayBundle, MirrorTube
from prisms import NGonPrism, IsoscelesPrism, RectangularPrism, CirclePrism, PRISMS
from enum import Enum
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from gui_utils.camera_settings import count_cameras, user_pick_resolution
from gui_utils.gui_picker import ChooseItemDialog
from gui_utils.camera import Camera, pick_camera
from gui_utils.text_annotation import StatusMessages
from error_handling import ShutdownException

import sys



class KScopeState(Enum):
    setup = 0
    running = 1
    shaping = 2

    shutdown = 100


class KScopyApp(object):
    MIRROR_TYPES = PRISMS
    RUNNING_MESSAGES = ["SPACE - toggle this on-screen display."
                        "Q - quit"]
    _OSD_TEXT_COLOR = (255, 254, 250)
    _OSD_BKG_COLOR = (118, 100, 90)
    _OSD_ALPHA = 0.65
    _FPS_PERIOD_SEC = 2.0

    def __init__(self):
        # state vars
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

        self._fps = {'n_in': 0,
                     'n_out': 0,
                     'n_dropped': 0,
                     'last_update_time': time.time() - self._FPS_PERIOD_SEC - 1.0,
                     'fps_in': 0,
                     'fps_out': 0,
                     'fps_drop': 0,
                     'OSD_lines': []}

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

            self._mirror_type = self._user_pick_shape()
            self._mirrors = self._mirror_type()
            self._cam.start()
            self._resolution = self._cam.get_resolution(wait=True)
            image_shape = (self._resolution[1], self._resolution[0], 3)
            self._status_bar = StatusMessages(image_shape, self._OSD_TEXT_COLOR,
                                              self._OSD_BKG_COLOR,
                                              bkg_alpha=self._OSD_ALPHA, spacing=10, max_font_scale=3.0)
            print("XXX")
            self._user_shape_mirrors()

            print("Starting scope...")
            self._state = KScopeState.running

        except ShutdownException:
            print("App Shutdown - by exception")
            pass

        print("MAIN_APP END")

    def _shutdown(self):
        self._state = KScopeState.shutdown
        cv2.destroyAllWindows()
        self._cam.shutdown()
        print("Main._shutdown() done.")

    def _wait(self):
        self._waiter.clear()
        while self._state != KScopeState.shutdown:
            self._waiter.wait(timeout=0.1)  # inelegant...

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
        self._status_bar.add_msgs(shaping_instructions, duration_sec=0)

        self._wait()

    def _user_pick_shape(self):
        """
        User selects type of mirror arrangement.
        """
        icons = [prism_type.get_icon(self._icon_size) for prism_type in KScopyApp.MIRROR_TYPES]
        if len(icons)==4:
            icons = [icons[:2],icons[2:]]
        #  Need good way to arange these...
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
            out_frame = in_frame.copy()  # render placeholder
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

        # KEYBOARD
        if k == ord('q'):
            print("q-Quit in main.")
            self._shutdown()

            print("Shutdown complete.")

            return

        elif self._state == KScopeState.running:
            self._handle_hotkeys(k)

        elif self._state == KScopeState.shaping:
            self._mirrors.handle_keyboard_adjust(k)

        # Needs to be done after window is created (alt idea: named window)
        if self._pending_mouse_callback is not None:
            cv2.setMouseCallback(self._window_name, self._pending_mouse_callback)
            self._pending_mouse_callback = None

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
            self._fps['OSD_lines'] = self.RUNNING_MESSAGES + ['  ',
                                                              "Ray-tracing:  %s" % (self._get_raytracing_status(),),
                                                              "FPS:  in = %.3f,  out = %.3f,  drop = %i" % (
                                                                  self._fps['fps_in'], self._fps['fps_out'],
                                                                  self._fps['fps_drop'])]

            self._status_bar.add_msgs(self._fps['OSD_lines'], self._FPS_PERIOD_SEC)

    def _get_raytracing_status(self):
        return "(ray-tracing status goes here)"

    def _handle_hotkeys(self, k):
        pass

def thread_monitor():
    start = time.time()
    while start + 5.0 > time.time():
        time.sleep(1)
        for thread in enumerate():
            print("%s - %s - %s" % (get_ident(), thread.ident, thread.name))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #t1 = Thread(target=thread_monitor)
    #t1.start()
    scope = KScopyApp()
    # _test_kscope_diagrams()
    # scope.view_live(0)
    # scope.view_image(cv2.imread('test_img.jpg'))

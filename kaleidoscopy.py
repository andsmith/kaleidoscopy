# import pylab as plt
import numpy as np
import cv2
import logging
import time
# from util import make_bounds, TextManager, Image
from threading import Lock, Event, enumerate, get_ident, Thread
from ray_tracing import ScopeTracer
from prisms import PRISMS
from enum import Enum
from gui_utils.gui_picker import ChooseItemDialog
from gui_utils.camera import Camera, pick_camera
from gui_utils.text_annotation import StatusMessages
from error_handling import ShutdownException
from rendering import ImageMapper
from layout import LAYOUT


class KScopeState(Enum):
    setup = 0
    shaping = 1
    ray_tracing = 2
    running = 3
    shutdown = 100


class KScopyApp(object):
    MIRROR_TYPES = PRISMS

    def __init__(self, output_shape=(300, 500)):
        self._output_shape = output_shape
        self._window_name = "Kaleidoscopy"
        self._state = KScopeState.setup
        # self._app_flow_lock = Lock()
        self._waiter = Event()
        self._OSD_on = True
        self._hotkeys = [{'key': ' ',
                          'desc': 'Help',
                          'func': self._toggle_osd,
                          'txt': "SPACE - toggle this on-screen display."},

                         {'key': 'q',
                          'desc': 'Quit',
                          'func': self._shutdown,
                          'txt': "Q - Quit."}]

        # self._drop_frame_lock = Lock()
        # try:
        # start
        if False:  ### TEMP DEBUG
            # with self._app_flow_lock:

            # User chooses camera
            self._cam_ind = pick_camera()
            self._cam = Camera(self._cam_ind, self._proc_frame, prompt_resolution=True)
        else:
            self._cam_ind = 0
            self._cam = Camera(self._cam_ind, self._proc_frame, prompt_resolution=False)

        # User chooses type of scope
        self._mirror_type = self._user_pick_shape()
        self._mirrors = self._mirror_type()
        self._cam.start()
        self._input_shape = self._cam.get_resolution(wait=True)[::-1]
        self._status_bar = StatusMessages(self._input_shape, LAYOUT['osd']['text_color'],
                                          # starts with resolution of camera
                                          LAYOUT['osd']['bkg_color'],
                                          bkg_alpha=LAYOUT['osd']['osd_alpha'], spacing=10, max_font_scale=1.0)
        self._renderer = ImageMapper(self._input_shape, self._output_shape)

        # User shapes mirror parameters
        if not self._user_shape_mirrors():
            return

        # clear callback
        cv2.setMouseCallback(self._window_name, lambda *args: None)
        self._ray_tracer =  ScopeTracer(mirrors=self._mirrors,
                                      output_shape=self._output_shape,  # "output" = produced image/mapping
                                       update_callback=self._update_k_map)  #

        # Start raytracing and rendering
        print("Starting Raytracer and Scope...")
        self._state = KScopeState.ray_tracing

        # setup main OSD (params may differ from OSD during mirror shaping)
        ###self._status_bar.set_image_shape(self._output_shape)  uncomment when outputinput is different
        self._status_bar.clear()
        self._status_bar.add_msgs([k['txt'] for k in self._hotkeys], "Help", duration_sec=0)

        # begin
        # self._ray_tracer.start()

        # except ShutdownException:
        #    print("App Shutdown - by exception!")
        #    pass

    @staticmethod
    def _user_pick_shape():
        """
        User selects type of mirror arrangement.
        """
        icons = [prism_type.get_icon(LAYOUT['icons']['size']) for prism_type in KScopyApp.MIRROR_TYPES]
        if len(icons) == 4:  # Need good way to arrange these automatically...
            icons = [icons[:2], icons[2:]]
        choice_ind = ChooseItemDialog(prompt="Select mirror Geometry:").ask_icons(icons)
        return KScopyApp.MIRROR_TYPES[choice_ind]

    def _user_shape_mirrors(self):
        """
        User sets parameters of selected mirror arrangement.
        :returns: results of the wait() call at the end (True, unless app was shut down during the wait.)
        """
        self._state = KScopeState.shaping  # now camera frames go to shaping method of prism
        shaping_instructions, mouse_callback = self._mirrors.start_shaping(self._window_name, self._waiter)
        self._pending_mouse_callback = mouse_callback  # needs to be registered in same thread as imshow
        self._status_bar.clear()
        self._status_bar.add_msgs(shaping_instructions, "shaping")
        return self._wait()  # for user to finish

    def _toggle_osd(self):
        self._OSD_on = not self._OSD_on
        logging.info("Toggling OSD:  now %s" % (self._OSD_on,))

    def _shutdown(self):
        self._state = KScopeState.shutdown
        cv2.destroyAllWindows()
        self._cam.shutdown()
        if hasattr(self, '_ray_tracer') and self._ray_tracer is not None:
            self._ray_tracer.shutdown()
        print("Main._shutdown() done.")

    def _update_k_map(self, img_map, stats):
        """
        Callback for renderer, to update the kaleidoscope mapping as it is created
        """
        self._k_map = img_map

    def _wait(self):
        """
        Waits for self._waiter (an Event object) to be set.
        For pausing for user input, etc.
        :returns:  True at the end, or False if app was shut down during the wait.
        """
        logging.info("Waiting...")
        self._waiter.clear()
        while self._state != KScopeState.shutdown:
            if self._waiter.wait(timeout=0.1):
                break
        logging.info("Done waiting.")
        if self._state == KScopeState.shutdown:
            logging.info("App shut down during wait()")
            return False
        return True

    def _proc_frame(self, in_frame, frame_time):
        """
        Frame processing callback.
        Can't throw shutdown exception, because this isn't the main thread.
        """

        self._in_frame = in_frame
        self._in_frame_time = frame_time

        # PROCESS
        if self._state == KScopeState.setup:
            # user picking camera & shape
            return

        elif self._state == KScopeState.shaping:
            out_frame = self._mirrors.get_masked_image(in_frame)
            self._status_bar.annotate_img(out_frame)

        elif self._state in (KScopeState.running, KScopeState.ray_tracing):
            out_frame = self._render(in_frame)
            # self._update_osd_info()
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
        
        if ord('q') == k & 0xff:
            logging.info("Hotkey shutdown starting...")
            self._shutdown()  # catch-all

        if self._state in[ KScopeState.running, KScopeState.ray_tracing]:
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
        # if self._k_map is None:
        ##    self._ray_tracer.get_current_map()
        # out_frame = self._renderer.render(frame, self._k_map['mapping'])
        return frame.copy()

    '''
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
            self._fps['fps_idle_pct'] = self._fps['idle % sum'] / elapsed
            self._fps['n_in'] = 0
            self._fps['n_out'] = 0
            self._fps['n_dropped'] = 0

            self._status_bar.add_msg("FPS:  in = %.3f,  out = %.3f,  drop = %i,  idle_pct = %i" % (
                self._fps['fps_in'], self._fps['fps_out'],
                self._fps['fps_drop'], self._fps['fps_idle_pct']), "fps")
            self._fps['last_update_time'] = time.time()

        if self._raytracing_stats is not None:
            self._status_bar.add_msg(self._raytracing_stats, "raytracing_stats", duration_sec=5.0)

        if self._render_stats is not None:
            self._status_bar.add_msg(self._render_stats, "render_stats")
        else:
            self._status_bar.remove_msg('render_stats')
    '''


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scope = KScopyApp()
    # _test_kscope_diagrams()
    # scope.view_live(0)
    # scope.view_image(cv2.imread('test_img.jpg'))

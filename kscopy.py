import numpy as np
import cv2
import logging
import time
from util import make_bounds, TextManager, Image
import threading
from ray_tracing import RayBundle, NGonPrism, MirrorTube
from enum import Enum


class KScopeState(Enum):
    stopped = 0
    running_live = 1
    running_image = 2
    remapping_live = 3
    remapping_image = 4
    shutdown = 99


class Kaleidoscope(object):
    """
    Handle  video.
    """

    def __init__(self,
                 mirrors,
                 output_resolution=(240, 320),
                 fov_x_deg=45.0,
                 ground_z_dist_cm=60.0,
                 image_plane_cm=4.0):
        """
        Geometry already defined
        :param mirrors:  MirrorTube object
        :param output_resolution:  (h, w) pixels
        :param fov_x_deg:  field of view for the h pixels
        :param ground_z_dist_cm:  added to bottom of kaleidoscope before image/video.
        """  # params
        theta_deg = 15.0
        self._output_resolution = output_resolution
        self._input_resolution = None
        self._mirrors = mirrors

        self._ground_z = ground_z_dist_cm + self._mirrors.get_vertical_span()[0]
        self._image_plane_z = image_plane_cm
        self._fov_x_deg = fov_x_deg
        self._rays = None
        self._img_map, self._dists, self._bounce = None, None, None

        self._set_rays()

        # state
        self._state = KScopeState.stopped
        self._image_in = None  # live view
        self._image_out = None  # live view
        self._image = None  # image view
        self._targ_fps_out_image_view = 1.0 / 0.010
        self._fps_disp_interval = 1.0
        self._cam_ind = None
        self._cam_thread = None
        self._fps_thread = None
        self._n_out_frames = None
        self._n_in_frames = None
        self._n_dropped_frames = None
        self._loop_n = None
        self._start_t = None
        self._render_lock = threading.Lock()
        self._ray_tracing_lock = threading.Lock()

        # GUI
        self._hotkeys = self._make_hotkeys()
        self._text = TextManager()
        self._out_window_name = "Kscopy"
        # cv2.namedWindow(self._out_window_name, cv2.WINDOW_NORMAL)  # broken in python3?
        self._mouse_state = {"l_button": None,
                             "r_button": None,
                             "pos": None,
                             "held_pos": None,
                             "abs_coords": None}
        self._colors = {'FPS': (0, 0, 0),
                        'recalculating': (30, 180, 40)}
        # cv2.setMouseCallback(self._out_window_name, self._mouse_event)

    def _set_rays(self):
        self._rays = RayBundle.from_resolution_and_fov(self._output_resolution,
                                                       self._image_plane_z,
                                                       fov_x_deg=self._fov_x_deg)

    def _set_image_map(self):
        self._img_map, self._dists, self._bounce = self._mirrors.get_image_map(rays=self._rays,
                                                                               max_reflect=30,
                                                                               ground_z_cm=self._ground_z,
                                                                               plot=False)

    def _image_view_loop(self):
        """
        while not self._finish:
        """
        raise NotImplementedError()

    def quit(self):
        logging.info("Quitting!")
        self.stop()
        self.shutdown()

    def _make_hotkeys(self):

        return {'q': {'name': 'Quit',
                      'dispatch': self.quit},
                }

    def _do_keyboard(self, k):
        for hotkey in self._hotkeys:
            if k & 0xff == ord(hotkey):
                params = {}
                if 'params' in self._hotkeys[hotkey]:
                    params.update(self._hotkeys[hotkey]['params'])
                self._hotkeys[hotkey]['dispatch'](**params)

    def _mouse_event(self, event, x, y, flags, param):
        pass

    def _annotate(self, img):
        """
        Add more text & ui stuff to image.
        :param img: input image
        :return:  image with added UI elements.
        """
        out_img = self._text.render(img)
        return out_img

    def _fps_thread_proc(self):
        logging.info("FPS monitor thread starting...")
        while self._state not in [KScopeState.stopped, KScopeState.shutdown]:
            t = time.time() - self._start_t

            fps_in = "FPS-in:  %.4f" % ((self._n_in_frames / t),) \
                if self._n_in_frames is not None else "FPS-in: 0"
            fps_out = "FPS-out:  %.4f" % ((self._n_out_frames / t),) \
                if self._n_out_frames is not None else "FPS-in: 0"
            logging.info(fps_in +", " + fps_out)
            self._text.add_text(fps_in, (10, 20), age=self._fps_disp_interval, font_scale=1, color=self._colors['FPS'])
            self._text.add_text(fps_out, (10, 40), age=self._fps_disp_interval, font_scale=1, color=self._colors['FPS'])
            self._reset_frame_counters()
            time.sleep(self._fps_disp_interval)
        logging.info("FPS monitor thread stopped.")

    def stop(self):
        logging.info("Stopping!")
        self._state = KScopeState.stopped
        if self._cam_thread is not None:
            self._cam_thread.join()

    def shutdown(self):
        self._state = KScopeState.shutdown
        # more ???

    def _reset_frame_counters(self):
        self._n_in_frames = 0
        self._n_out_frames = 0
        self._n_dropped_frames = 0
        self._start_t = time.time()

    def view_image(self, image, dpi=100.0):
        """
        logging.info("Starting with image:  %s (%s dpi)" % (self._image.shape, self._dpi))
        self._image = image
        self._dpi = dpi
        self._image = Image(self._image_in, px_per_cm=(self._dpi, self._dpi))
        self._reset_frame_counters()
        self._image_view_loop()
        logging.info("Image view Stopped.")
        """
        pass

    def view_live(self, cam_ind=None):
        self._cam_ind = cam_ind
        self._state = KScopeState.running_live
        logging.info("Starting live!")
        self._cam_thread = threading.Thread(target=self._live_proc)
        self._reset_frame_counters()
        self._cam_thread.start()
        self._fps_thread = threading.Thread(target=self._fps_thread_proc)
        self._fps_thread.start()

        logging.info("Live view started...")

    def _live_proc(self):

        self._cam = cv2.VideoCapture(self._cam_ind)
        logging.info("Acquired camera %i." % (self._cam_ind,))
        self._loop_n = 0
        while self._state in [KScopeState.running_live, KScopeState.remapping_live]:
            self._loop_n += 1
            r, frame = self._cam.read()
            if not r:
                logging.warning("No camera data, waiting for 1 sec...")
                time.sleep(1.0)
                continue
            self._n_in_frames += 1
            self._image_in = frame

            if self._render_lock.acquire(blocking=False):
                self._calc_live_image(self._image_in)  # reflect!
                self._render_lock.release()
            else:
                logging.info("Drop!")
                self._image_out = self._image_in.copy()
                self._n_dropped_frames += 1

            self._image_out = self._annotate(self._image_out)

            cv2.imshow(self._out_window_name, self._image_out)
            k = cv2.waitKey(1)  # time this better...
            self._do_keyboard(k)

        self._cam.release()
        logging.info("Released camera %i." % (self._cam_ind,))
        logging.info("Live view stopped.")

    def _start_recalculating_map(self, image):
        logging.info("Recalculating ray map thread started...")
        self._input_resolution = image.shape[:2]
        self._image = Image(image, self._input_resolution)
        self._set_rays()
        self._set_image_map()
        logging.info("Done recalculating ray map thread ending.")

    def _calc_live_image(self, image_in):
        """
        guaranteed to be called serially, can be slow to recalculate image
        """
        self._n_out_frames += 1
        if self._img_map is None:
            # not ready yet?
            if self._state != KScopeState.remapping_live:
                self._state = KScopeState.remapping_live  # only run once!
                recalc = threading.Thread(target=self._start_recalculating_map, args=(image_in,))
                recalc.start()

            self._image_out = self._image_in.copy()

            self._text.add_text("Recalculating...", (10, 70), age=-1,font_scale=2, color=self._colors['recalculating'])
        else:
            self._image.set_image(image_in)
            self._image_out = self._image.interpolate_integer(self._img_map)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    geom = NGonPrism(n=3, r=1.012341234, top=2.54, bottom=50.0)
    mirrors = MirrorTube(shape=geom)
    scope = Kaleidoscope(mirrors)
    scope.view_live(0)

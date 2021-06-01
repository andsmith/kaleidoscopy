import numpy as np
import cv2
import logging
import time
from util import make_bounds, TextManager, Image
import threading
from ray_tracing import RayBundle, NGonPrism, MirrorTube


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
        self._running = False
        self._finish = False

        self._ray_map = None
        self._image_in = None
        self._image_out = None
        self._fps_out = 1.0 / 0.010
        self._live = None
        self._cam = None

        self._cam_ind = None
        self._cam_thread = None
        self._n_out_frames = None
        self._n_in_frames = None
        self._start_t = None

        # GUI
        self._hotkeys = self._make_hotkeys()
        self._text = TextManager()
        self._out_window_name = "Kscopy"
        cv2.namedWindow(self._out_window_name, cv2.WINDOW_NORMAL)
        self._mouse_state = {"l_button": None,
                             "r_button": None,
                             "pos": None,
                             "held_pos": None,
                             "abs_coords": None}
        cv2.setMouseCallback(self._out_window_name, self._mouse_event)

    def _set_rays(self):
        self._rays = RayBundle.from_resolution_and_fov(self._output_resolution,
                                                       self._image_plane_z,
                                                       fov_x_deg=self._fov_x_deg)

    def _set_image_map(self):
        self._img_map, self._dists, self._bounce = self._mirrors.get_image_map(rays=self._rays,
                                                                               max_reflect=30,
                                                                               ground_z_cm=self._ground_z,
                                                                               plot=False)

    def _image_proc(self):
        """
        while not self._finish:
            if self._cur_img is None:
                logging.warning("No image to display.")
                time.sleep(.25)

            image = self._annotate(self._cur_img)
            cv2.imshow(self._out_window_name, image)
            k = cv2.waitKey(int(1.0 / self._fps_out))  # time this better...
            self._do_keyboard(k)
        """
        raise NotImplementedError()

    def quit(self):
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
        fps = self._n_out_frames / (time.time() - self._start_t)
        fps_out = "FPS out:  %.3f" % (fps,)
        if self._image is None:
            fps_in = self._n_in_frames / (time.time() - self._start_t)
            fps_out = "FPS in:  %.3f -- %s" % (fps_in, fps_out)
            # reset?
        self._text.add_text(fps_out, pos=(self._output_resolution[0] - 30, 30), age=-1)
        logging.info(fps_out)
        return self._text.render(img)

    def _set_img_bounds(self, dpi):
        width = self._image.shape[1] / dpi
        height = self._image.shape[0] / dpi
        self._image_bounds = make_bounds([[-width / 2.0, -height / 2.0, ],
                                          [width / 2.0, height / 2.0, ], ])

    def stop(self):
        logging.info("Stopping!")
        self._finish = True
        if self._cam_thread is not None:
            self._cam_thread.join()

    def _setup(self):
        self._n_in_frames = 0
        self._n_out_frames = 0
        self._start_t = time.time()
        self._finish = False
        self._started = True

    def view_image(self, image, dpi=100.0):
        logging.info("Starting with image:  %s (%s dpi)" % (self._image.shape, self._dpi))
        self._image = image
        self._dpi = dpi
        self._set_img_bounds(dpi)
        self._setup()
        self._image_proc()
        logging.info("Image view Stopped.")

    def view_live(self, cam_ind=None):
        self._cam_ind = cam_ind
        logging.info("Starting live!")
        self._cam_thread = threading.Thread(target=self._cam_proc)
        self._setup()
        self._cam_thread.start()
        logging.info("Live view started...")

    def _cam_proc(self):
        self._cam = cv2.VideoCapture(self._cam_ind)
        while not self._finish:
            r, frame = self._cam.read()
            if not r:
                logging.warning("No camera data, waiting for 1 sec...")
                time.sleep(1.0)
                continue
            self._n_in_frames += 1
            self._frame_in = frame
            self._update_out_image()

        logging.info("Live view stopped.")

    def _update_out_image(self):
        image = self._frame_in.copy()

        if self._img_map is None:
            # do setup we couldn't until now
            self._input_resolution = self._frame_in.shape[:2]
            self._set_rays()
            self._set_image_map()
            self._interp_image = Image(image, self._input_resolution)

            # return input img + msg
            msg = "ray-tracing.."
            self._text.add_text(msg, pos=(30, 30), age=-1)
            return self._text.render(image)

        self._interp_image.set_image(image)
        self.img_out = self._interp_image.interpolate_integer(self._img_map)


if __name__ == "__main__":
    geom = NGonPrism(n=3, r=1.012341234, top=2.54, bottom=50.0)
    mirrors = MirrorTube(shape=geom)
    scope = Kaleidoscope(mirrors)

    scope.view_live(0)

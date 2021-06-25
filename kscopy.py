import pylab as plt
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
    shutdown = 99


class Kaleidoscope(object):
    """
    Handle  video.
    """

    def __init__(self,
                 mirrors,
                 output_resolution=(240, 320),
                 fov_x_deg=45.0,
                 ground_z_cm=30.0,
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

        self._ground_z = ground_z_cm
        self._image_plane_z = image_plane_cm
        self._fov_x_deg = fov_x_deg
        self._rays = None
        self._img_map, self._dists, self._bounce = None, None, None
        self._img_map_old = False

        self._set_rays()

        # state
        self._state = KScopeState.stopped
        self._image_in = None
        self._image_offset = np.array([0.0, 0.0, 0.0])
        self._fixed_image = None  # Image object, for image view.
        self._dpi = None
        self._render = None  # camera frame or image after being reflected
        self._render_is_old = False  # input has changed wrt scope

        self._display_delay_ms = 20
        self._fps_disp_interval = 1.0
        self._cam_ind = None
        self._cam_thread = None
        self._fps_thread = None
        self._scope_thread = None
        self._n_frames_in = None
        self._n_frames_out = None
        self._n_frames_rendered = None
        self._start_t = None

        # GUI
        self._hotkeys = self._make_hotkeys()
        self._text = TextManager()
        self._win_name = "KScopy"

        self._mouse_state = {"l_button": None,
                             "r_button": None,
                             "pos": None,
                             "held_pos": None,
                             "abs_coords": None}
        self._colors = {'FPS': (0, 0, 0),
                        'recalculating': (30, 180, 40)}

    def draw_diagram(self):

        def plot_wrap(x, y, *args, **kwargs):
            y = np.array(y)

            # y = self._ground_z - y
            return plt.plot(x, y, *args, **kwargs)

        # draw eye
        import pylab as plt
        eye_h = plot_wrap(0, 0, '.b', markersize=10)

        # draw scope
        corners = self._mirrors.get_corners()
        z = self._mirrors.get_vertical_span()
        scope_h = None
        for i in range(corners.shape[0]):
            # plot x,z projection
            scope_h = plot_wrap((corners[i, 0], corners[i, 0]), (z[0], z[1]), 'r-', linewidth=2)

        # draw image_plane
        image_h = plot_wrap([-1, 1], [self._image_plane_z, self._image_plane_z], 'k:')

        # trace rays, just one row to see how it bounces
        shape = (1, 5)

        test_rays = RayBundle.from_resolution_and_fov(shape, self._ground_z, self._fov_x_deg)
        origins, directions = test_rays.get_active_rays()
        origins = origins.copy()
        directions = directions.copy()

        def _draw_rays(orgs, dirs, *args, **kwargs):
            line_starts = []
            line_stops = []
            for ray in range(dirs.shape[0]):
                line_starts.append([orgs[ray, 0], orgs[ray, 2]])
                line_stops.append([dirs[ray, 0], dirs[ray, 2]])

            line_starts, line_stops = np.vstack(line_starts), np.vstack(line_stops)

            x_coords = np.zeros(3 * line_starts.shape[0])
            y_coords = x_coords * 0
            x_coords[::3] = line_starts[:, 0]
            x_coords[1::3] = line_stops[:, 0]
            x_coords[2::3] = np.nan
            y_coords[::3] = line_starts[:, 1]
            y_coords[1::3] = line_stops[:, 1]
            y_coords[2::3] = np.nan
            plt.gca().invert_yaxis()

            return plot_wrap(x_coords, y_coords, *args, **kwargs)

        # _draw_rays(origins, directions, 'k-', linewidth=1)
        # draw rays

        _, _, _, bounce_hist = self._mirrors.trace(test_rays,
                                                   self._ground_z,
                                                   max_reflect=10,
                                                   plot=False,
                                                   record=True)
        rays_h = test_rays.plot_bounce_history(bounce_hist, linewidth=.5)
        x_coord_lists = [np.array(bounce_hist[0][i])[:,0].tolist() for i in range(len(bounce_hist[0]))]
        x_coords = [[xc for x_coord_list in x_coord_lists for xc in x_coord_list]]
        target_extent = np.array([np.min(x_coords), np.max(x_coords)])
        target_margin = 0.07 * (target_extent[1] - target_extent[0])
        target_extent[0] -= target_margin
        target_extent[1] += target_margin
        # draw viewed image
        target_h = plot_wrap(target_extent, [self._ground_z, self._ground_z], 'k-', linewidth=4)

        handles = [eye_h[0], scope_h[0], image_h[0], target_h[0], rays_h[0]]
        labels = ["eye", "mirrors", "image plane", "image/camera", "rays"]
        plt.legend(handles, labels)
        plt.title("Kaleidoscope diagram (scale cm)")

    def _set_rays(self):
        self._rays = RayBundle.from_resolution_and_fov(self._output_resolution,
                                                       self._image_plane_z,
                                                       fov_x_deg=self._fov_x_deg)

    def _set_image_map(self):
        self._img_map, self._dists, self._bounce = self._mirrors.get_image_map(rays=self._rays,
                                                                               max_reflect=30,
                                                                               ground_z_cm=self._ground_z,
                                                                               plot=False)

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

            fps_in = "FPS-in:  %.4f" % ((self._n_frames_in / t),) \
                if self._n_frames_in is not None else "FPS-in: 0"
            fps_out = "FPS-out:  %.4f" % ((self._n_frames_out / t),) \
                if self._n_frames_out is not None else "FPS-in: 0"
            fps_rend = "FPS-rendered:  %.4f" % ((self._n_frames_rendered / t),) \
                if self._n_frames_rendered is not None else "FPS-in: 0"
            logging.info(fps_in + ", " + fps_out)
            self._text.add_text(fps_in, (10, 20), age=self._fps_disp_interval, font_scale=1, color=self._colors['FPS'])
            self._text.add_text(fps_out, (10, 40), age=self._fps_disp_interval, font_scale=1, color=self._colors['FPS'])
            self._text.add_text(fps_rend, (10, 60), age=self._fps_disp_interval, font_scale=1,
                                color=self._colors['FPS'])
            self._reset_frame_counters()
            time.sleep(self._fps_disp_interval)
        logging.info("FPS monitor thread stopped.")

    def stop(self):
        logging.info("Stopping!")
        self._state = KScopeState.stopped

    def shutdown(self):
        self._state = KScopeState.shutdown
        # more ???

    def _reset_frame_counters(self):
        self._n_frames_in = 0
        self._n_frames_out = 0
        self._n_frames_rendered = 0
        self._start_t = time.time()

    def view_image(self, image, dpi=100.0):
        self._dpi = dpi
        logging.info("Starting with image:  %s (%s dpi)" % (image.shape, dpi))
        self._fixed_image = Image(image, px_per_cm=(self._dpi, self._dpi))
        self._image_in = self._fixed_image.get_image()

        self._reset_frame_counters()
        self._state = KScopeState.running_image
        self._main_loop()
        logging.info("Image view stopped.")

    def view_live(self, cam_ind=0, dpi=100.0):
        self._cam_idn = cam_ind
        self._dpi = dpi
        logging.info("Starting camera view.")
        self._state = KScopeState.running_live
        self._main_loop()
        logging.info("Camera view stopped.")

    def _main_loop(self):
        """
        Show webcam through kalleidoscope.
        :param cam_ind: which camera (cv2 index)
        """

        logging.info("Starting scope!")

        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback(self._out_window_name, self._mouse_event())

        self._fps_thread = threading.Thread(target=self._fps_thread_proc)
        self._reset_frame_counters()
        self._fps_thread.start()

        if self._state == KScopeState.running_live:
            self._cam_thread = threading.Thread(target=self._cam_proc)
            self._cam_thread.start()
        self._scope_thread = threading.Thread(target=self._scope_proc)
        self._scope_thread.start()

        # main display loop
        while self._state not in (KScopeState.shutdown, KScopeState.stopped):

            # get current output of scope (rendering) thread
            render = self._render  # copy reference, don't need lock
            if render is None:
                logging.warning("No data to display yet.")
                time.sleep(.25)
                continue
            self._image_out = self._annotate(render)
            cv2.imshow(self._win_name, self._image_out)
            k = cv2.waitKey(self._display_delay_ms)
            self._n_frames_out += 1
            self._do_keyboard(k)
        logging.info("Waiting for rendering thread to stop.")
        self._scope_thread.join()
        if self._state == KScopeState.running_live:
            logging.info("Waiting for camera thread to stop.")
            self._cam_thread.join()
        logging.info("Waiting for fps thread to stop.")
        self._fps_thread.join()
        cv2.destroyWindow(self._win_name)
        logging.info("Stopping scope.")

    def _cam_proc(self):
        self._cam = cv2.VideoCapture(self._cam_ind)
        logging.info("Acquired camera %i." % (self._cam_ind,))
        while self._state == KScopeState.running_live:
            r, frame = self._cam.read()
            if not r:
                logging.warning("No camera data, waiting...")
                time.sleep(.5)
                continue
            self._n_frames_in += 1
            self._image_in = frame
        self._cam.release()
        logging.info("Released camera %i." % (self._cam_ind,))

    def _raytrace_map(self, input_shape):
        logging.info("Recalculating ray map thread started...")
        self._input_resolution = input_shape
        self._set_rays()
        self._set_image_map()
        logging.info("Done recalculating ray map thread ending.")

    def _scope_proc(self, ):
        """
        Continually render images.
        """
        last_image_rendered = None
        while self._state not in [KScopeState.shutdown, KScopeState.stopped]:
            image = self._image_in  # not locked, copy ref
            if image is None:
                logging.warning("No input data to render!")
                time.sleep(.25)
                continue

            if self._img_map_old or self._img_map is None:
                self._raytrace_map(image.shape[:2])
                self._img_map_old = False
                self._render_is_old = True

            if self._render_is_old:
                self._render = self._fixed_image.interpolate_integer(self._img_map)
                self._n_frames_rendered += 1

        return


def _test_kscope_diagram():
    geom = NGonPrism(n=4, r=np.sqrt(2.0), top=1.54, bottom=12.54, phi=np.pi / 4.)
    mirrors = MirrorTube(shape=geom)
    scope = Kaleidoscope(mirrors, ground_z_cm=20.0, image_plane_cm=1.02)
    scope.draw_diagram()
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_kscope_diagram()
    # scope.view_live(0)
    # scope.view_image(cv2.imread('test_img.jpg'))

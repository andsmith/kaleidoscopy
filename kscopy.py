import numpy as np
import cv2
import logging
import time
from util import make_bounds, TextManager
import threading


class MirrorAssembly(object):
    """
    Define a ortho-prism (shape) assembly of mirrors, i.e. all perpendicular to flat, facing indwards.
    Input is an arbitrary list of 2-d polygon vertices.
    """

    def __init__(self, corners):
        """
        Define mirror prism.
        :param corners:  list of 2-d coordinates (numpy arrays), i.e. mirror closes loop of vertices
        """
        n = len(corners)

        # mirror centers are midpoints between corners, and in 3d
        centers = [(self._corners[i] + self._corners[i + 1]) / 2.0 for i in range(n - 1)]
        centers.append((self._corners[0] + self._corners[-1]) / 2.0)
        centers = [np.hstack(c, [0]) for c in centers]

        # Need two non-parallel, co-planar vectors whose cross-product will give us the normal for each mirror
        # First will connect each mirror's centers to a corner.
        co_planar_a = [centers[i] - corners[i] for i in range(self._n)]
        # Second will connect each mirror's first corner to a point 1cm above that corner.
        co_planar_b = [centers[i] - np.hstack((corners[i:2], [1.0])) for i in range(n)]
        normals = np.cross(co_planar_a, co_planar_b)

        self._bounds = make_bounds(corners)
        self._n = n
        self._centers = centers
        self._corners = corners
        self._normals = normals

    def get_bounds(self):
        return self._bounds


class IsosceleseMirrorAssembly(MirrorAssembly):
    def __init__(self, theta_deg, h_cm):
        theta = np.deg2rad(theta_deg)
        corners = [np.array([-np.sin(theta), 0]),
                   np.array([0, h_cm]),
                   np.array([np.sin(theta), 0]), ]
        super(IsosceleseMirrorAssembly, self).__init__(corners=corners)


class Kaleidoscope(object):

    def __init__(self,
                 mirrors=None,
                 resolution=(1024, 1024),
                 fov_deg=45.0,
                 eye_scope_cm=4.0,
                 scope_image_scm=20.0):
        """
        Kaleidoscope object, main GUI.
        :param mirrors: MirrorAssmebly object, defaults to Isosceles triangle if None
        :param resolution:  h, w (output)
        :param fov_deg:  field of view
        :param eye_scope_cm:  distance, float
        :param scope_image_scm:  distance, float
        """
        # params
        theta_deg = 15.0
        self._resolution = resolution
        self._mirrors = IsosceleseMirrorAssembly(theta_deg=theta_deg, h_cm=5.0) if mirrors is None else mirrors
        self._dims = {'eye_scope_cm': eye_scope_cm,
                      'fov': np.deg2rad(fov_deg),
                      'scope_img_cm': scope_image_scm,
                      'theta_deg': theta_deg}
        self._eye_scope_cm = eye_scope_cm

        # state
        self._running = False
        self._finish = False
        i, j = np.arange(resolution[0]), np.arange(resolution[1])
        self._all_coords = np.dstack(np.meshgrid(i, j)).reshape(-1, 2)
        self._cur_img = None
        self._ray_map = None
        self._image = None
        self._frame_in = None
        self._fps_out = 1.0 / 0.010
        self._image_bounds = None
        self._dpi = None
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

    def _image_proc(self):
        while not self._finish:
            if self._cur_img is None:
                logging.warning("No image to display.")
                time.sleep(.25)

            image = self._annotate(self._cur_img)
            cv2.imshow(self._out_window_name, image)
            k = cv2.waitKey(int(1.0 / self._fps_out))  # time this better...
            self._do_keyboard(k)

    def quit(self):
        self.stop()
        self.shutdown()

    def _adjust(self, param, dir):
        if param not in self._dims:
            raise Exception("parameter not found in dims:  %s" % (param,))
        self._dims[param] += dir
        self._generate_ray_map()

    def _make_hotkeys(self):
        def _toggle_state():
            if self._running:
                self.stop()
            else:
                self.start()

        return {'q': {'name': 'Quit',
                      'dispatch': self.quit},
                ' ': {'alt': '[SPACE]',
                      'name': 'Start / Stop',
                      'dispatch': _toggle_state},
                'a': {'name': "Increase eye-scope distance.",
                      'dispatch': self._adjust,
                      'params': ("eye_scope_cm", 1)},
                'z': {'name': "Decrease eye-scope distance.",
                      'dispatch': self._adjust,
                      'params': ("eye_scope_cm", -1)},
                's': {'name': "Increase scope-image distance.",
                      'dispatch': self._adjust,
                      'params': ("scope_image_cm", 1)},
                'x': {'name': "Decrease scope-image distance.",
                      'dispatch': self._adjust,
                      'params': ("scope_image_cm", -1)},
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
        self._text.add_text(fps_out, pos=(self._resolution[0] - 30, 30), age=-1)
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
        self._n_out_frames = 0
        self._start_t = time.time()
        self._finish = False
        self._started = True
        self._update_ray_map()

    def view_image(self, image, dpi=100.0):
        logging.info("Starting with image:  %s (%s dpi)" % (self._image.shape, self._dpi))
        self._image = image
        self._dpi = dpi
        self._set_img_bounds(dpi)
        self._setup()
        self._image_proc()
        logging.info("Image view Stopped.")

    def view_live(self, cam_ind):
        self._cam_ind = cam_ind
        logging.info("Starting live!")
        self._setup()
        self._cam_thread = threading.Thread(target=self._cam_proc)
        self._n_in_frames = None
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
        return np.zeros(shape=self._resolution)

    def _update_ray_map(self):
        # calculate dimensions of image plane & it's bounds of the rays.
        origin = np.ones(3).reshape(1, 3)

        # define image plane
        i_plane_z = np.arctan2(self._dims['fov'] / 2.0) * self._dims['d']
        asgaerga

        scope_extent = self._mirrors.get_bounds()
        height_px_per_cm = self._resolution[0] / (scope_extent['top'] - scope_extent['bottom'])
        width_px_per_cm = self._resolution[1] / (scope_extent['right'] - scope_extent['left'])

        if height_px_per_cm > width_px_per_cm:
            if scope_extent['top'] > -scope_extent['bottom']:
                scale = scope_extent['top'] / self._eye_scope_cm
            else:
                scale = -scope_extent['bottom'] / self._eye_scope_cm
        else:
            if scope_extent['left'] > -scope_extent['right']:
                scale = scope_extent['left'] / self._eye_scope_cm
            else:
                scale = -scope_extent['right'] / self._eye_scope_cm

        image_plane_scope = make_bounds([[]])


if __name__ == "__main__":
    scope = Kaleidoscope()
    img = cv2.imread('test_img.jpg')
    scope.start_image(img)

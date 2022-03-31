import pylab as plt
import numpy as np
import cv2
import logging
import time
from util import make_bounds, TextManager, Image
import threading
from ray_tracing import RayBundle, NGonPrism, MirrorTube, IsoscelesPrism
from enum import Enum
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec


class KScopeState(Enum):
    stopped = 0
    running_live = 1
    running_image = 2
    shutdown = 99


class Kaleidoscope(object):
    """
    Main app & scope object.  (FIX:  separate these)
    """

    def __init__(self,
                 mirrors,
                 output_resolution=(240, 320),
                 fov_deg=45.0,
                 ground_z_cm=20.0):
        """
        Eye is at (0, 0, 0), looking in the Z+ direction.
        The Image plane is at the eyepiece (top of kaleidoscope).
        The output pixels tile the image plane such that the eyepiece is just entirely enclosed.
        
        Geometry already defined by mirrors.
        :param mirrors:  MirrorTube object
        :param output_resolution:  (h, w) pixels
        :param fov:  field of view for the narrower dimension
        :param ground_z_cm:  added to bottom of kaleidoscope before image/video.
        """  # params
        theta_deg = 15.0
        self._output_resolution = output_resolution
        self._input_resolution = None
        self._mirrors = mirrors
        r = mirrors.get_view_rad()
        self._top_z = r / np.tan(np.deg2rad(fov_deg) / 2.0)
        logging.info("Scope initialized with eye at %.2f cm, and viewport of diameter %.2f cm." % (
            self._top_z, r * 2.0))
        self._ground_z = ground_z_cm
        self._image_plane_z = self._top_z  # image plane at top of `scope
        self._fov_deg = fov_deg
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

    def draw_3d(self, ax=None):

        n = self._mirrors.get_corners().shape[0]

        # set colors
        palette = cm.get_cmap('brg')
        color_indices = np.linspace(0, 1.0, n + 1)
        colors = [palette(i) for i in color_indices]

        # plot mirrors
        scope_h, ax = self._mirrors.plot_3d(ax=ax, z_offset=self._top_z)
        scope_h = [scope_h]

        # draw eye
        eye_h = [ax.scatter(0.0, 0.0, 0.0, color='b')]

        # trace rays, just one row to see how it bounces
        shape = (11, 11)
        test_rays = RayBundle.from_resolution_and_fov(shape, self._ground_z, self._fov_deg)
        origins, directions = test_rays.get_active_rays()
        origins = origins.copy()
        directions = directions.copy()

        test_dists, test_intersects = test_rays.get_plane_intersections(np.array([0, 0, self._top_z]),
                                                                        np.array([0, 0, 1.0]))

        # draw ray / image-plane intersections
        x = test_intersects[:, 0]
        y = test_intersects[:, 1]
        z = self._image_plane_z

        _ = ax.scatter(x, y, z, color='k')

        # set bounding box
        ax.set_zlim(0, self._ground_z)
        z_lim = np.array((0, self._ground_z))
        x_lim = z_lim - np.mean(z_lim)
        y_lim = z_lim - np.mean(z_lim)
        ax.scatter(x_lim, y_lim, z_lim, color=(0, 1, 1), alpha=0.0)

        return ax

    def draw_diagram(self):
        fig2 = plt.figure(constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
        top_ax = fig2.add_subplot(spec2[0, 0])
        side_ax = fig2.add_subplot(spec2[:, 1])
        bottom_ax = fig2.add_subplot(spec2[1, 0])

        # draw eye
        eye_h = top_ax.plot(0, 0, '.b', markersize=14)
        side_ax.plot(0, 0, '.b', markersize=14)

        # draw scope
        corners = self._mirrors.get_corners()
        z_span = self._top_z, self._top_z + self._mirrors.get_height()
        scope_h = None
        for i in range(corners.shape[0]):
            # plot x,z projection
            scope_h = side_ax.plot((corners[i, 0], corners[i, 0]),
                                   (z_span[0], z_span[1]), 'r-', linewidth=2)
            top_ax.plot((corners[i, 0], corners[i, 2]), (corners[i, 1], corners[i, 3]), 'r-', linewidth=2)

        # trace rays, just one row to see how it bounces
        shape = (6, 6)

        test_rays = RayBundle.from_resolution_and_fov(shape, image_plane_z=self._top_z, fov_deg=self._fov_deg,
                                                      square=True)

        # draw rays from top, intersecting image plane (do before trace() changes rays)
        _, img_plane_intersects = test_rays.get_plane_intersections(np.array([0, 0, self._top_z]),
                                                                    np.array([0, 0, 1.0]))
        trace = self._mirrors.trace(test_rays,
                                    ground_z_cm=self._ground_z,
                                    scope_top_z_cm=self._top_z,
                                    max_reflect=10,
                                    record=True)

        def _plot_ray_subset(ax, intersects, mask, *args, **kwargs):
            x = intersects[mask.reshape(-1), 0]
            y = intersects[mask.reshape(-1), 1]
            return ax.plot(x, y, *args, **kwargs)

        good_rays = np.logical_and(np.logical_not(trace['missed_scope']),
                                   np.logical_not(trace['hit_top']))

        missed_h = _plot_ray_subset(top_ax, img_plane_intersects, trace['missed_scope'], 'rx', markersize=8)
        hit_h = _plot_ray_subset(top_ax, img_plane_intersects, trace['hit_top'], 'gx', markersize=8)
        good_rays_h = _plot_ray_subset(top_ax, img_plane_intersects, good_rays, 'ko', markersize=3)

        # draw rays hitting targe

        missed_targ_h = _plot_ray_subset(bottom_ax,trace['image_map'], trace['missed_scope'], 'rx', markersize=8)
        good_rays_targ_h = _plot_ray_subset(bottom_ax,trace['image_map'], good_rays, 'ko', markersize=3)

        # draw inscribed circle
        rad = self._mirrors.get_view_rad()
        theta = np.linspace(0.0, np.pi * 2.0, 201)
        x = np.cos(theta) * rad
        y = np.sin(theta) * rad
        eyehole_h = top_ax.plot(x, y, 'g-', linewidth=3)

        # draw rays from side
        rays_h = test_rays.plot_bounce_history(trace['bounce_histories'], mask=good_rays, ax=side_ax, linewidth=.5)

        # draw image target
        max_x = np.max(trace['image_map'][good_rays, 0].reshape(-1))
        min_x = np.min(trace['image_map'][good_rays, 0].reshape(-1))
        target_extent = [min_x - 1, max_x + 1]
        target_h = side_ax.plot(target_extent, [self._ground_z, self._ground_z], 'k-', linewidth=4)

        # draw image_plane
        image_plane_h = side_ax.plot([-1.66, 1.66], [self._image_plane_z, self._image_plane_z], 'k:', linewidth=3)

        side_handles = [eye_h[0], image_plane_h[0], scope_h[0], rays_h[0], target_h[0]]
        side_labels = ["eye", "image_plane", "mirrors", "rays", "target/camera image"]
        top_handles = [eye_h[0], scope_h[0], eyehole_h[0], missed_h[0], hit_h[0], good_rays_h[0]]
        top_labels = ["eye", "mirrors", 'inscribed circle', 'rays missing scope', 'hitting scope top', 'entering scope']
        top_ax.legend(top_handles, top_labels)
        side_ax.legend(side_handles, side_labels, loc='lower right')
        side_ax.set_title("Side view (orthographic)")
        top_ax.set_title("Top view - rays at image plane")
        bottom_ax.set_title("Top view - rays at target")

        side_ax.set_xlabel('x (cm)')
        side_ax.set_ylabel('z')
        bottom_ax.set_xlabel('x')
        bottom_ax.set_ylabel('y')
        top_ax.set_ylabel('z (cm)')

        side_ax.axis('equal')
        bottom_ax.axis('equal')
        top_ax.axis('equal')
        side_ax.invert_yaxis()

    def _set_rays(self):
        self._rays = RayBundle.from_resolution_and_fov(self._output_resolution,
                                                       self._image_plane_z,
                                                       fov_deg=self._fov_deg)

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


def _test_kscope_diagrams():
    geom = NGonPrism(n=4, r=np.sqrt(2.0), height=11.323, phi=np.pi / 4.)
    # geom = IsoscelesPrism(theta_deg=45, h_cm=2.0, height=11.0, )
    mirrors = MirrorTube(prism=geom)
    scope = Kaleidoscope(mirrors, ground_z_cm=20.0, fov_deg=60.0)

    scope.draw_diagram()
    plt.show()
    # plt.show()
    # scope.draw_top_diagram(flat=True)
    # plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    _test_kscope_diagrams()
    # scope.view_live(0)
    # scope.view_image(cv2.imread('test_img.jpg'))

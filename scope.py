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

from raytracing import FakeRaytracer
from remap_img import remap
import numpy as np
import cv2
import logging
import time
import sys
from geom import TARG_Z, COLORS, BKG
from user_interface import UIModes, UILayer

WINDOW_NAME = "Kaleidoscopy!"


class ScopeApp(object):
    def __init__(self, output_size, input_size=None, input_img=None):
        """
        :param output_size: the size of the output video stream (width, height)
        :param input_size: the size of the input video stream (width, height), ignored if input_img is provided
        :param input_img: if provided, this image will be used instead of the webcam feed. Should be a numpy array.
        """
        self._bkg = input_img
        self.out_size = output_size
        self.in_size = input_size if input_img is None else (input_img.shape[1], input_img.shape[0])
        self._frame_out = np.zeros((self.out_size[1], self.out_size[0], 3), dtype=np.uint8)
        self._f_no = 0
        self._running = False
        self._mirrors = None

        self._init_input()
        self._fake_raytracer = FakeRaytracer(output_size)
        self._init_ui()

    @property
    def mirrors(self):
        return self._mirrors

    def shutdown(self):
        self._running = False

    def set_mirrors_and_restart(self, new_mirrors):
        logging.info("Setting mirrors: %s", new_mirrors)
        self._mirrors = new_mirrors
        # TODO: start real raytracer here

    def _init_ui(self):
        self._ui_layer = UILayer(self, window_name=WINDOW_NAME)

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
            logging.info("Starting camera...")
            self._cam = cv2.VideoCapture(0)
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.out_size[0])
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.out_size[1])
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

    def start(self):
        self._running = True
        cv2.namedWindow(WINDOW_NAME)

        while self._running:
            input_frame = self._get_img()
            if input_frame is None:
                time.sleep(0.1)
                continue

            if self._mirrors is not None:
                # Mirrors configured: show fake raytracer placeholder
                self._frame_out = self._fake_raytracer.render()
            else:
                # No mirrors yet: show raw input as background
                self._frame_out = self._make_img_frame(input_frame)

            self._ui_layer.draw_layer(self._frame_out)
            self._f_no += 1

            cv2.imshow(WINDOW_NAME, self._frame_out)
            k = cv2.waitKey(1)
            if not self._ui_layer.handle_keypress(k):
                break

        if self._cam is not None:
            self._cam.release()
        cv2.destroyAllWindows()


def start_scope():
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            logging.error("Could not read image: %s" % sys.argv[1])
            sys.exit(1)
    else:
        img = None
    out_size = 1280, 720
    app = ScopeApp(out_size, input_img=img)
    app.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_scope()

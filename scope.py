"""
Open a cv2 window to the kalleidoscope.
"""

from raytracing import Raytracer
from mirror import make_iso_mirrors,make_mirror_box
from remap_img import remap
import numpy as np
import cv2
from threading import Thread, Lock
import logging
import time
import sys
import os

from img_util import resize_and_pad

class ScopeApp(object):
    def __init__(self, size, threading=False, img = None):
        self._bkg = img 
        self._size = size if img is None else img.shape[:2][::-1]
        self._frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self._frame_out = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self._f_no = 0

        self._init_mirrors()
        self._init_geom(threading)
        self._init_cam()

    def _init_mirrors(self):
        # Create a polygon of mirrors.
        # Make sure (0, 0) is in the interior of the polygon, or it won't look right.
        self._mirrors = make_mirror_box(.6)  # make_iso_mirrors(20.0)

    def _init_geom(self, threaded):
        # Determine the field of view for the image plane
        # Create the raytracer.
        d_target = 2.0
        x_max = 1.0
        aspect = self._size[0] / self._size[1]
        y_max = x_max * aspect
        self._raytracer = Raytracer(size=self._size, mirrors=self._mirrors,
                                    targ_z=d_target,
                                    x_max=x_max, y_max=y_max,
                                    threaded=threaded)

    def _init_cam(self):
        # Create the camera.
        if self._bkg is not None:
            logging.info("Using %i x %i image instead of camera frames." % (self._bkg.shape[1], self._bkg.shape[0]))
            self._frame_out = self._bkg
            self._cam = None
        else:
            logging.info("Starting camera...")
            self._cam = cv2.VideoCapture(0)
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
            # TODO: Fullscreen!

        self._f_no = 0

    def _get_img(self):
        if self._cam is None:
            return self._bkg.copy()
        
        frame, ret = self._cam.read()

        if not ret:
            logging.warning("Failed to read frame.")
            return None
        
        return frame
    
    def start(self):
        self._running = True
        logging.info("Starting raytracer...")
        self._raytracer.start()

        while self._running:
            frame = self._get_img()
            if frame is None:
                time.sleep(0.1)
                continue

            px_map = self._raytracer.get_map()
            remap(frame, self._frame_out, px_map[0], px_map[1])
            self._f_no += 1

            cv2.imshow("Kaleidoscopy!", self._frame_out)
            k = cv2.waitKey(1)
            if k == ord('q'):
                self._running = False
                break
            # TODO: zoom/pan, load image, move mirror corners, etc.

        self._raytracer.stop()
        self._cam.release()



def start_scope():
    size = (7, 7)
    if len(sys.argv)>1:
        img = cv2.imread(sys.argv[1])
        img = resize_and_pad(img, size)
    else:
        img = None
    app = ScopeApp(size, img=img)
    app.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import ipdb; ipdb.set_trace()
    start_scope()

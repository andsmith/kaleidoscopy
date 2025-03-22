"""
Open a cv2 window to the kalleidoscope.
"""

from raytracing import Raytracer, make_iso_mirrors
from remap_img import remap
import numpy as np
import cv2
from threading import Thread, Lock
import logging
import time

class ScopeApp(object):
    def __init__(self, size, threading=False):
        self._size = size
        self._frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self._frame_out = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self._f_no = 0

        self._init_mirrors()
        self._init_geom(threading)
        self._init_cam()

    def _init_mirrors(self):
        # Create a polygon of mirrors.
        # Make sure (0, 0) is in the interior of the polygon, or it won't look right.
        self._mirrors = make_iso_mirrors(20.0)
    
    def _init_geom(self, threading):
        # Determine the field of view for the image plane
        # Create the raytracer.
        d_target = 2.0
        x_max = 1.0
        aspect = self._size[0] / self._size[1]
        y_max = x_max * aspect
        self._raytracer = Raytracer(self._mirrors, d_target, x_max, y_max, threading=threading)

    def _init_cam(self):
        # Create the camera.
        logging.info("Starting camera...")
        self._f_no = 0
        self._cam = cv2.VideoCapture(0)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        # TODO: Fullscreen!

    def start(self):
        self._running = True
        logging.info("Starting raytracer...")
        self._raytracer.start(w_px = self._size[0], h_px = self._size[1])
        
        while self._running:
            frame, ret = self._cam.read()
            if not ret:
                logging.warning("Failed to read frame.")
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
    size = (1920, 1080)
    app = ScopeApp(size)
    app.start()

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)
    start_scope()
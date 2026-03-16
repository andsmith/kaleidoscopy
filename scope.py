"""
Start the Kalleidoscope app:

Running mode: 
   * Run with an image as the input argument to focus the kalleidoscope on that image.
   * Run with no argument to use the webcam as the input.





"""

from raytracing import Raytracer
from mirror_tube import MirrorTube

from remap_img import remap
import numpy as np
import cv2
from threading import Thread, Lock
import logging
import time
import sys
import os
from enum import IntEnum

from geom import TARG_Z,COLORS

class RunMode(IntEnum):
    UI_SHOWING = 0  # user is changing mirrors (start-up mode, "space" to toggle w/RUNNING).
    RUNNING = 1  # Raytracer has started (may have even finished), video is mapping, this is the normal running mode.
    SHUT_DOWN = 99  # everything should stop if they see this.


from img_util import resize_and_pad

class ScopeApp(object):
    def __init__(self, output_size=(1920, 1080), input_size=(1200,1024), input_img = None):
        """
        :param output_size: the size of the output video stream (width, height)
        :param input_size: the size of the input video stream (width, height), ignored if input_img is provided
        :param input_img: if provided, this image will be used instead of the webcam feed. Should be a numpy array.
        """
        
        self._bkg = input_img 
        self.out_size = output_size
        self.in_size = input_size if input_img is None else input_img.shape[:2][::-1]
        self._frame_in = np.zeros((self.in_size[1], self.in_size[0], 3), dtype=np.uint8)
        self._frame_out = np.zeros((self.out_size[1], self.out_size[0], 3), dtype=np.uint8)
        self._f_no = 0
        
        self.mode = RunMode.UI_SHOWING

        #self._init_mirrors()
        self._init_geom(threading)
        self._init_cam()
        self._init_ui()
        
    def _init_ui(self):

    def _init_mirrors(self):
        # Create a polygon of mirrors.
        # Make sure (0, 0) is in the interior of the polygon, or it won't look right.
        self._mirrors = make_mirror_box(.6)  # make_iso_mirrors(20.0)

    def _init_geom(self, threaded):
        # Determine the field of view for the image plane
        # Create the raytracer.
        d_target = TARG_Z
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
    
    if len(sys.argv)>1:
        img = cv2.imread(sys.argv[1])
        img = resize_and_pad(img,  (7, 7))
    else:
        img = None
    out_size = 1920, 1080
    app = ScopeApp(out_size, input_img=img)
    app.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_scope()

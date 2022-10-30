"""
Given a kaleidoscope's input-output mapping, transform an image/frame
"""
from map_img import image_map

import numpy as np


class ImageMapper(object):
    """
    Class to render kaleidoscope images.
    """

    def __init__(self, input_shape, output_shape):
        self._map = None
        self._in_shape, self._out_shape = input_shape, output_shape
        self._out_blank = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)

    def update_mapping(self, new_mapping):
        self._map = new_mapping

    def render(self, input, mapping):
        if self._map is None:
            return input.copy()
        dest = self._out_blank.copy()
        image_map(img_src=input, img_dest=dest, img_map=mapping)
        return dest

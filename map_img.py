import timeit
import cv2
import matplotlib.pyplot as plt
import numpy as np
import ctypes
import os


def is_windows():
    return os.name == 'nt'


def _load_algorithm(name):
    """
    Load from shared library / dll
    """
    if is_windows():
        lib = ctypes.cdll.LoadLibrary('./map_img.dll')
    else:
        lib = ctypes.cdll.LoadLibrary('./map_img.so')
    alg = getattr(lib, name)

    def func(img_src, img_dest, img_map):
        return alg(ctypes.c_void_p(img_src.ctypes.data),
                   ctypes.c_void_p(img_dest.ctypes.data),
                   ctypes.c_void_p(img_map.ctypes.data),
                   ctypes.c_int(img_src.shape[0]),
                   ctypes.c_int(img_src.shape[1]))

    return func


# for numpy extension
image_map = _load_algorithm('map_img')


def test_img_map():
    img = 20-np.arange(20).astype(np.uint8).reshape(4, 5)
    img_map = (img * 0).astype(np.uint32)
    img_map[0, :3] = np.array([2, 4, 7])
    dest = img * 0

    image_map(img_src=img, img_dest=dest, img_map=img_map)

    import pprint
    print("Original:")
    pprint.pprint(img)
    print("Map:")
    pprint.pprint(img_map)
    print("Result:")
    pprint.pprint(dest)


if __name__ == "__main__":
    test_img_map()

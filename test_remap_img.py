import time
from pygame_app.pygame_app import PygameVideoIOApp
import numpy as np
import matplotlib.pyplot as plt
from remap_img import remap
import cv2
import logging
#from loop_timing.loop_profiler import LoopPerfTimer as LPT


def move_random_square(maps,square_size=30):
    x, y = maps
    h, w= x.shape
    def _rand_square():
        x0 = np.random.randint(0, w - square_size)
        y0 = np.random.randint(0, h - square_size)
        x1 = x0 + square_size
        y1 = y0 + square_size
        return x0, y0, x1, y1
    src = _rand_square()
    dest = _rand_square()
    x[src[1]: src[3], src[0]: src[2]] = x[dest[1]: dest[3], dest[0]: dest[2]]
    y[src[1]: src[3], src[0]: src[2]] = y[dest[1]: dest[3], dest[0]: dest[2]]
    return x, y

class RemapImgApp(PygameVideoIOApp):
    """
    Select N random squares and move them to different parts of the image.
    Test using pygame.
    """
    def __init__(self, size=(1920, 1080)):
        vid_out_props = {'size': size, 'fps': 30}
        vid_in_props = {'size': size, 'ind': 0, 'mirror': True}
        self._f_no =0

        super(RemapImgApp, self).__init__(name = 'remap test',
                                            vid_out_props = vid_out_props,
                                            vid_in_props = vid_in_props,
                                            threaded=False)

        self.map = np.meshgrid(np.arange(size[0]), np.arange(size[1], dtype=np.int32))
        for _ in range(30):
            self.map = move_random_square(self.map, square_size=200)
        self._blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self._frame_out = self._blank.copy()

        #LPT.reset(enable=True, burn_in = 30, display_after=30)

    #@LPT.time_function    
    def camera_update(self, new_frame, dt):
        image = np.ascontiguousarray(self.surface_to_array(new_frame))
        frame_out = self._blank.copy()
        _remap(image, frame_out, self.map[0], self.map[1])
        self._frame_out = frame_out
        self._f_no +=1

    #@LPT.time_function    
    def app_update(self, dt):
        new_frame = self.array_to_surface((self._frame_out))
        return new_frame
    
    def _handle_events(self):
        #LPT.mark_loop_start()
        return super()._handle_events()
    
    
#@LPT.time_function    
def _remap(img_src, img_dest, map_x, map_y):
    """
    Rearange an image, pixel-by-pixel.
    """
    remap(img_src, img_dest, map_x, map_y)

def _test_live():
    RemapImgApp().run()

def make_test_image(w,h):
    img = np.zeros((h,w, 3), dtype=np.uint8)
    for i in range(w):
        img[:, i,:] = int(255 * (i /w))
    #for i in range(h):
    #    img[i, :,1] = int(255 * ((h-i) /h))
        
    return img


def test_frame():
    # test a single image
    image = cv2.imread('test_img.png') #make_test_image(640, 480)
    frame_out = np.zeros(image.shape, dtype=np.uint8)
    map_x, map_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0], dtype=np.int32))
    for _ in range(10):
        map_x, map_y = move_random_square((map_x, map_y), square_size=200)

    remap(image, frame_out, map_x, map_y)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(frame_out)
    plt.show()

class NumpyRemapper(object):
    # remap images, not using c extension
    def __init__(self, map_x, map_y):
        self._x = map_x.reshape(-1)
        self._y = map_y.reshape(-1)
        self._shape = map_x.shape

    def remap(self, img_src, img_dest):
        shape = (self._shape[0], self._shape[1], img_src.shape[2])
        img_dest[:] = img_src[self._y, self._x].reshape(shape)


def test_live_2(use_extension=True):
    # test with CV2
    cam = cv2.VideoCapture(0)
    # set resolution to 1920x1080
    w, h=  1920, 1080
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h, dtype=np.int32))
    np_remapper = NumpyRemapper(map_x.copy(), map_y.copy())

    t_total = 0.0
    t_start = time.perf_counter()
    n_frames = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame_out = np.zeros(frame.shape, dtype=np.uint8)
        t0 = time.perf_counter()

        if use_extension:
            remap(frame, frame_out, map_x, map_y)
        else:
            np_remapper.remap(frame, frame_out)
        t_total += time.perf_counter() - t0
        n_frames += 1

        if n_frames % 100 == 0:
            now = time.perf_counter()
            print('FPS: ', n_frames / (now - t_start))
            print('Avg remap time: %.3f msec'%( t_total / n_frames * 1000,))


        cv2.imshow('frame', frame_out)
        k= cv2.waitKey(1) & 0xFF
        if k== ord('q'):
            break
        elif k == ord("s"):
            map_x,map_y = move_random_square((map_x, map_y), square_size=200)
            np_remapper = NumpyRemapper(map_x.copy(), map_y.copy())



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_live_2()
    #_test_live()
    #test_frame()
            
        

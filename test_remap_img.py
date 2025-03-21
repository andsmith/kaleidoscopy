from pygame_app.pygame_app import PygameVideoIOApp
import numpy as np
import matplotlib.pyplot as plt
from remap_img import remap
import cv2

def make_square_remap(w, h, n_squares=20, square_size=30):
    def _rand_square():
        x0 = np.random.randint(0, w - square_size)
        y0 = np.random.randint(0, h - square_size)
        x1 = x0 + square_size
        y1 = y0 + square_size
        return x0, y0, x1, y1

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    for _ in range(n_squares):
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
    def __init__(self, size=(640, 480)):
        vid_out_props = {'size': size, 'fps': 30}
        vid_in_props = {'size': size, 'ind': 0, 'mirror': True}
        self._f_no =0

        super(RemapImgApp, self).__init__(name = 'remap test',
                                            vid_out_props = vid_out_props,
                                            vid_in_props = vid_in_props,
                                            threaded=False)

        self.map = make_square_remap(size[0], size[1], n_squares=10, square_size=200)
        self._blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self._frame_out = self._blank.copy()

    def camera_update(self, new_frame, dt):
        image = np.ascontiguousarray(self.surface_to_array(new_frame))
        frame_out = self._blank.copy()
        _remap(image, frame_out, self.map[0], self.map[1])
        self._frame_out = frame_out
        self._f_no +=1

    def app_update(self, dt):
        new_frame = self.array_to_surface((self._frame_out))
        return new_frame
    
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
    map_x, map_y = make_square_remap(image.shape[1], image.shape[0], square_size=150, n_squares=1)
    print(map_x.shape, map_y.shape)
    print(image.shape, frame_out.shape)
    remap(image, frame_out, map_x, map_y)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(frame_out)
    plt.show()

def test_live_2():
    # test with CV2
    cam = cv2.VideoCapture(0)
    map_x, map_y =None, None

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame_out = np.zeros(frame.shape, dtype=np.uint8)
        if map_x is None:
            map_x, map_y = make_square_remap(frame.shape[1], frame.shape[0], square_size=150, n_squares=15)
        
        remap(frame, frame_out, map_x, map_y)
        cv2.imshow('frame', frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    #test_live_2()
    _test_live()
    #test_frame()
            
        

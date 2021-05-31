import logging
import time
import os
import numpy as np
import cv2
from scipy.interpolate import griddata


class Image(object):
    """
    Handle image transformations & interpolation
    """

    def __init__(self, data, px_per_cm=(100., 100.)):
        """

        :param data: image (H x W x 3)
        :param px_per_cm:  image scale, (h, w)
        """
        self._shape = data.shape
        self._img = data.reshape(-1, self._shape[2])
        self._scale = px_per_cm
        self._coords = None
        self.reset_image_coords()

    @staticmethod
    def from_file(filename, flip_bgr_rgb=False, **kwargs):
        data = cv2.imread(filename)
        logging.info('Loading image "%s":  %s.' % (filename, data.shape))
        if flip_bgr_rgb:
            data = data[:, :, ::-1]
        return Image(data, **kwargs)

    def get_image(self):
        return self._img.reshape(self._shape)

    def get_coords(self):
        return self._coords.reshape(list(self._img.shape[:2]) + [2])

    def reset_image_coords(self):

        span_x = self._shape[1] / self._scale[1]
        span_y = self._shape[0] / self._scale[0]
        x_coords = np.linspace(-span_x, span_x, self._shape[1])
        y_coords = np.linspace(-span_y, span_y, self._shape[0])
        self._coords = np.dstack(np.meshgrid(x_coords, y_coords)).reshape(-1, 2)
        logging.info("Reset image coordinates to:  %s x %s (x,y - cm)" % (span_x, span_y))

    def update_image(self, new_data):
        if not np.array_equal(new_data.shape, self._img.shape):
            raise Exception("Update image must have same shape.")
        self._img = new_data.reshape(-1, new_data.shape[2])

    def update_scale(self, new_scale):
        self._scale = new_scale

    def interpolate(self, coords, **kwargs):
        """
        get image interpolated at specified world coordinates
        :param coords: H x W x 2 (x, y coordinates) in world-frame
        :param kwargs:  passed to interpolator
        :return:  H x W x 3 (RGB)  image
        """
        logging.info("Interpolating on grid of coordinates:  %s" % (coords.shape,))
        colors = []
        for channel in range(self._img.shape[1]):
            colors.append(griddata(self._coords,
                                   self._img[:, channel],
                                   (coords[:, :, 0], coords[:, :, 1]),
                                   **kwargs))
        return cv2.merge(colors)


class TextManager(object):
    """
    For adding text to output streams.
    "Set it and forget it" -- add text with expiration times, etc.
    """

    def __init__(self):
        self._items = []

    def add_text(self, text, pos, age=0, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255),
                 thickness=1, linestyle=cv2.LINE_AA):
        """
        Most args are for cv2 puttext.
        :param text: string
        :param pos: (row, col)
        :param age: 0 for indefinite, or seconds, or -1 for just once
        :param font: cv2 param
        :param font_scale: cv2 param
        :param color: cv2 param
        :param thickness: cv2 param
        :param linestyle: cv2 param
        :return: index, used to delete before expiration
        """
        t = {'text': text,
             'pos': pos,
             'fontScale': font_scale,
             'font': font,
             'color': color,
             'thickness': thickness,
             'linestyle': linestyle,
             'age': age,
             'start': time.time()
             }
        (w, h), _ = cv2.getTextSize(t['text'], t['font'], t['fontScale'], t['thickness'])
        self._items.append(t)
        return w, h

    def add_lines(self, lines, pos, leading=0.4, *args, **kwargs):
        """
        Put multiple lines.
        :param lines:  list of text lines
        :param pos: lower left corner of first line of text
        :param leading: spacing of lines
        :param args:  additional args to add_text
        :param kwargs: additional args to add_text
        :return:  lower left corner of last line of text
        """

        px = pos[0]
        py = pos[1]
        for line in lines:
            _, h = self.add_text(line, (px, py), *args, **kwargs)
            py += int(float(h) * (1.0 + leading))
        return py

    def remove(self, item):
        self._items.pop(item)

    def _check_ages(self):
        valid = [i for i in self._items if i['age'] <= 0 or time.time() - i['start'] < i['age']]
        if len(valid) < len(self._items):
            logging.info("Text item(s) aged out.")
        self._items = valid

    def render(self, img):

        self._check_ages()

        image = img.copy()
        for t in self._items:
            image = cv2.putText(image, t['text'], t['pos'], t['font'], t['fontScale'], t['color'], t['thickness'],
                                t['linestyle'])
        singles_removed = [i for i in self._items if i['age'] >= 0]
        self._items = singles_removed
        return image


def get_index_grid(img_shape, grid_shape):
    r = np.linspace(0, img_shape[0] - 1, grid_shape[0] + 1).astype(np.int64)
    c = np.linspace(0, img_shape[1] - 1, grid_shape[1] + 1).astype(np.int64)
    return r, c


def make_int_grid(shape):
    return np.zeros(np.prod(shape), dtype=np.int64).reshape(shape)


def pct_str(n, d, fmt_str="%.3f"):
    return fmt_str % (100.0 * n / d)


def check_make_dir(path, uniquify=False):
    if os.path.exists(path):
        if uniquify:
            index = 0
            p_temp = "%s_%i" % (path, index)
            while os.path.exists(p_temp):
                index += 1
                p_temp = "%s_%i" % (path, index)
            path = p_temp
    else:
        os.mkdir(path)
    return path


def test_image():
    pic = Image.from_file('test_img.jpg')


def test_text_manager():
    tm = TextManager()
    img = np.zeros(500 * 350 * 3, dtype=np.uint8).reshape((350, 500, 3))
    tm.add_text("blah", (100, 30))
    tm.add_lines(['This', 'is', "a test."], (300, 60))
    """
    import matplotlib.pylab as plt
    plt.imshow(tm.render(img))
    plt.show()
    """


if __name__ == "__main__":
    test_text_manager()


def make_bounds(coords):
    max_vals = np.max(coords, axis=0)
    min_vals = np.min(coords, axis=0)
    return {'top': max_vals[1],
            'bottom': min_vals[1],
            'left': min_vals[0],
            'right': max_vals[0]}

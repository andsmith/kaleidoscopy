import time
import os
import numpy as np
import cv2


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
        print("Adding lines")
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
            print("Aged out!")
        self._items = valid

    def render(self, img):

        self._check_ages()

        image = img.copy()
        for t in self._items:
            image = cv2.putText(image, t['text'], t['pos'], t['font'], t['fontScale'], t['color'], t['thickness'],
                                t['linestyle'])
        singles_removed = [i for i in self._items if i['age']>=0]
        self._items = singles_removed
        return image


def get_index_grid(img_shape, grid_shape):
    r = np.linspace(0, img_shape[0] - 1, grid_shape[0] + 1).astype(np.int64)
    c = np.linspace(0, img_shape[1] - 1, grid_shape[1] + 1).astype(np.int64)
    return r, c


def make_int_grid(shape):
    return np.zeros(np.prod(shape), dtype=np.int64).reshape(shape)


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

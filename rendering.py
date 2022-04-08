class ImageMapper(object):
    def __init__(self, resolution):
        self._img_shape = resolution[1], resolution[0]

    def render(self, input, mapping):
        return input.copy()

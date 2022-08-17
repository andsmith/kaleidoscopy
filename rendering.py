class ImageMapper(object):
    def __init__(self, input_shape, output_shape):
        self._in_shape, self._out_shape = input_shape, output_shape

    def render(self, input, mapping):
        return input.copy()

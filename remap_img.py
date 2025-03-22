from distutils.core import setup, Extension
import numpy as np

remap_img_module = Extension(
    'remap_img',
    sources=['remap_img.c'],
    include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_compile_args=['-O2'],
)

setup(
    name='RemapImg',
    version='1.0',
    description='Rearange an image, pixel-by-pixel.',
    ext_modules=[remap_img_module],
)

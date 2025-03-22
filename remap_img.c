#include <Python.h>
#include <numpy/arrayobject.h>

// Function to perform image remapping

/* Image remapping function:  create a new image by specifying where in another image each pixel comes from.

    Let the following H x W arrays be the input:
        img_src (uint8), the source image, 
        img_dest (uint8), the destination image, 
        x_map (int32), the map for the x-coordinate (second index).
        y_map (int32), the map for the y-coordinate (first index).

    The remap() function will set each pixel (for all i and j):
      img_dest(i,j) <- img_src(y_map[i, j], x_map[i, j])
 */

 static PyObject* remap_img(PyObject *self, PyObject *args) {
    PyArrayObject *img_src, *img_dest, *x_map, *y_map;

    // Parse arguments: four NumPy arrays
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &img_src, &PyArray_Type, &img_dest, &PyArray_Type, &x_map, &PyArray_Type, &y_map)) {
        return NULL;
    }

    // Get dimensions
    int rows_src = (int)PyArray_DIM(img_src, 0);
    int cols_src = (int)PyArray_DIM(img_src, 1);
    int chans_src = (int)PyArray_DIM(img_src, 2);
    int rows_dest = (int)PyArray_DIM(img_dest, 0);
    int cols_dest = (int)PyArray_DIM(img_dest, 1);
    int chans_dest = (int)PyArray_DIM(img_dest, 2);
    int rows_x = (int)PyArray_DIM(x_map, 0);
    int cols_x = (int)PyArray_DIM(x_map, 1);
    int rows_y = (int)PyArray_DIM(y_map, 0);
    int cols_y = (int)PyArray_DIM(y_map, 1);

    // Check if matrices are all the same size
    if (rows_src != rows_dest || cols_src != cols_dest || rows_src != rows_x || cols_src != cols_x 
        || rows_src != rows_y || cols_src != cols_y || chans_src != chans_dest) {
        PyErr_SetString(PyExc_ValueError, "Incompatible matrix dimensions for remapping, need HxWx3, HxWx3, HxW, HxW.");
        return NULL;
    }

    // Perform remapping
    unsigned char *img_src_data = (unsigned char*)PyArray_DATA(img_src);
    unsigned char *img_dest_data = (unsigned char*)PyArray_DATA(img_dest);
    int *x_map_data = (int*)PyArray_DATA(x_map);
    int *y_map_data = (int*)PyArray_DATA(y_map);

    for (int i = 0; i < rows_src; ++i) {
        for (int j = 0; j < cols_src; ++j) {
            int index = i * cols_src + j;
            int src_index = y_map_data[index] * cols_src + x_map_data[index];
            if (src_index < 0 || src_index >= rows_src * cols_src) 
                continue;
            for (int k = 0; k < chans_src; ++k) {
                img_dest_data[index * chans_src + k] = img_src_data[src_index * chans_src + k];
            }
        }
    }

    Py_RETURN_NONE;
}

// Method definition table
static PyMethodDef RemapImgMethods[] = {
    {"remap", remap_img, METH_VARARGS, "Rearange an image, pixel-by-pixel."},
    {NULL, NULL, 0, NULL} // Sentinel value ending the table
};

// Module definition structure
static struct PyModuleDef remapmodule = {
    PyModuleDef_HEAD_INIT,
    "remap_img", // Module name
    NULL, // Module documentation, can be NULL
    -1, // Size of per-interpreter state or -1
    RemapImgMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_remap_img(void) {
    import_array(); // Initialize NumPy's array functionality.
    return PyModule_Create(&remapmodule);
}
#pragma once

#ifdef _WIN32
    #ifdef BUILD_MAPIMG_LIBRARY
        #pragma message("Defined!")
        #define EXPORT_SYMBOL __declspec(dllexport)
    #else
        #pragma message("undefined!")
        #define EXPORT_SYMBOL __declspec(dllimport)
    #endif
#else
    #define EXPORT_SYMBOL
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

void print_arr( int*arr, int n_cols, int n_rows, int print_cols, int print_rows);
void print_arr( unsigned char*arr, int n_cols, int n_rows, int print_cols, int print_rows);

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_SYMBOL void map_img(void *image_src,
                             void *image_dest,
                             void *image_map,
                             int row_count,
                             int col_count);

#undef EXPORT_SYMBOL
#ifdef __cplusplus
}
#endif

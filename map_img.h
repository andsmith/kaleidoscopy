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
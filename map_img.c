/*
 *
 * build:  gcc -O3 --shared --std:c11 map_img.c -o map_img.so
 * windows build:  cl  /LD /DBUILD_MAPIMG_LIBRARY map_img.c

 */
//
#include <stdio.h>
#include <stdlib.h>
#include "map_img.h"

void map_img(void *image_src_v,
             void *image_dest_v,
             void *image_map_v,
             int row_count,
             int col_count)
{
    int r,c, index, src_index;
    int *image_src= (int*)image_src_v;
    int *image_dest= (int*)image_dest_v;
    int *image_map = (int*)image_map_v;


    for (r=0;r<row_count;r++){
        for (c=0;c<col_count;c++){
            //image_dest[i] = image_src[image_map[i]]
            index = r * col_count + c;
            src_index = image_map[index];
            image_dest[index] = image_src[src_index];
        }
    }
}

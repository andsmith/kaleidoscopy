/*
 *
 * build:  gcc -O3 --shared --std:c11 map_img.c -o map_img.so
 * windows build:  cl /I C:\Python39\lib\site-packages\numpy\core\include /LD /DBUILD_MAPIMG_LIBRARY /I C:\Python39\include map_img.cpp /link /LIBPATH:C:\Python39\Libs\
 */
//
#include <stdio.h>
#include <stdlib.h>
#include <numpy\ndarraytypes.h>
#include <numpy\ndarrayobject.h>
#include "numpy\arrayobject.h"

#include "map_img.h"
void map_img(void *image_src_v,
             void *image_dest_v,
             void *image_map_v,
             int row_count,
             int col_count)
{
    int r,c, index, src_index,src_index_p1;
    char *image_src= (char*)image_src_v;
    char *image_dest= (char*)image_dest_v;
    int *image_map = (int*)image_map_v;
    char buf[1024];
    //print_arr(image_dest, row_count,col_count, row_count,col_count);

    image_dest[0]=1;
            puts("bstartinguf");


    //sprintf(buf, "rc:  %i, cc:  %i\n", row_count, col_count);
    //puts(buf);
    for (r=0;r<2;r++){
        for (c=0;c<2;c++){
            //image_dest[i] = image_src[image_map[i]]
            index = r * col_count + c;
            src_index = image_map[index];

            src_index_p1 = src_index==(row_count*col_count-1)?0:src_index;
            image_dest[index] = image_src[src_index];
            sprintf(buf,"r: %i, c: %i, index: %i, sv:  %i, val %i, val+1 %i\n",r,c,index, src_index, image_src[src_index],
              image_src[src_index_p1]);
            puts(buf);
            print_arr(image_dest, row_count,col_count, row_count,col_count);

        }
    }
    free(buf);
    puts("Done.\n");
    /**/
}



void print_arr( char*arr, int n_cols, int n_rows, int print_cols, int print_rows){
    // Print an array, or the upper left corner of one.

    int n_char, i,width = 9;  // chars per number
    char *bp, buf[1024],
         sbuf[1024];
    int row, col;

    if (print_cols==-1 || print_cols > n_cols) print_cols = n_cols;
    if (print_rows==-1 || print_rows > n_rows) print_rows = n_rows;

    for (row=0;row<print_rows;row++){
        bp=buf;
        for (col=0;col<print_cols;col++){
            //n_char=sprintf(sbuf, "%.2f", arr[row*n_cols + col]);
            n_char=sprintf(sbuf, "%i", arr[row*n_cols + col]);
            if (n_char < width)
                for (i=0;i<width-n_char;i++)
                    *bp++ = ' ';
            sprintf(bp,"%s", sbuf);
            bp+=n_char;
        }
        puts(buf);
    }
    buf[0]=0;
    puts(buf);
    free(buf);
}
int main(int argc, char*argv[]){
    int i,j;
    int w=5,h=4;
    int *mapping = (int*)PyMem_Malloc(sizeof(int)*w*h);
    char *dest = (char*)PyMem_Malloc(sizeof(char)*w*h);
    char *img = (char*)PyMem_Malloc(sizeof(char)*w*h);
    for (i=0;i<w*h;i++){
        img[i] = 20-i;
        dest[i]=0;
        mapping[i] = 0;
        }
    mapping[0]=2;
    mapping[1]=4;
    mapping[2]=7;

    print_arr(img, h,w,h,w);
    //print_arr(mapping, h,w,h,w);
    map_img(img, dest, mapping, h, w);
    print_arr(dest, h,w,h,w);

    return 0;
}
gcc -g -fPIE -I/home/andrew/.local/lib/python3.8/site-packages/numpy/core/include `python3-config --includes --libs --cflags --ldflags`  --shared map_img.cpp -o map_img.so


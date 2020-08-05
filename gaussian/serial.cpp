#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

double gaussian_value(int x, int y);
void init_kernel(int *kernel, int k);

int main() {
    for(int i = 0;)
}

double gaussian_value(int x, int y) {
    
}

void init_kernel(int *kernel, int kernel_length) {
    for(int i = 0; i < kernel_length; i++) {
        for(int j = 0; j < kernel_length; j++) {
            kernel[i * kernel_length + j] = gaussian_value(i, j);
        }
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

void convolve(int *matrix, int height, int width, int *kernel, int k);
void reduce_matrix(int *matrix_a, int *matrix_b, int *result_matrix, int height, int width);
void image_to_matrix(cv::Mat image, int *matrix);
void matrix_to_image(cv::Mat image, int *matrix);

int main() {
    std::string filename = "../images/harold.jpg";

    cv::Mat image = cv::imread(filename);
    int height = image.rows;
    int width = image.cols;

    cv::Mat grey_image(height, width, CV_8UC1, cv::Scalar(0, 0, 0)); 
    cv::cvtColor(image, grey_image, cv::COLOR_BGR2GRAY);

    int *matrix_x = (int*) malloc(height * width * sizeof(int));
    int *matrix_y = (int*) malloc(height * width * sizeof(int));

    // fill matrix with image's values
    image_to_matrix(image, matrix_x);
    image_to_matrix(image, matrix_y);

    int sobel_x[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    int sobel_y[9] = {
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1
    };

    int k = 1;
    convolve(matrix_x, height, width, sobel_x, k);
    convolve(matrix_y, height, width, sobel_y, k);

    int *result_matrix = (int*) malloc(height * width * sizeof(int));
    reduce_matrix(matrix_x, matrix_y, result_matrix, height, width);

    matrix_to_image(image, result_matrix);
    cv::imwrite("harold.jpg", image);

    free(matrix_x);
    free(matrix_y);
    free(result_matrix);

    return 0;
}

void convolve(int *matrix, int height, int width, int *kernel, int k) {
    int kernel_length = 2 * k + 1;

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {

            double sum = 0;

            for(int u = -k; u <= k; u++) {
                for(int v = -k; v <= k; v++) {

                    if(i + u >= 0 && i + u < height &&
                       j + v >= 0 && j + v < width) {

                           sum += kernel[(u + k) * kernel_length + (v + k)] * matrix[(i + u) * width + (j + v)];
                    }

                }
            }

            matrix[i * width + j] =  sum;

        }
    }
}

// assumes all matrices are equal size
// uses pythagorean theorem to get magnitudes of the 2 input matrices
void reduce_matrix(int *matrix_a, int *matrix_b, int *result_matrix, int height, int width) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int a = matrix_a[i * width + j];
            int b = matrix_b[i * width + j];

            result_matrix[i * width + j] = (int) sqrtf(a * a + b * b);
        }
    }
}

void image_to_matrix(cv::Mat image, int *matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            matrix[i * image.cols + j] = image.at<uchar>(i, j);
        }
    }
}

void matrix_to_image(cv::Mat image, int *matrix) {
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            image.at<uchar>(i, j) = matrix[i * image.cols + j];
        }
    }
}

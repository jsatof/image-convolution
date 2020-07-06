#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {

    std::string file_name = "harold.jpg";
    cv::Mat image;
    image = cv::imread(file_name, 1);

    if(!image.data) {
        std::cout << "No image data." << std::endl;
        return 1;
    }

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    cv::waitKey(0);

    return 0;
}



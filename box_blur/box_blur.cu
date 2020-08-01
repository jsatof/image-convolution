#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    // get image name from command args
    if(argc != 2) {
        printf("usage: ./boxblur <name of image from /images>\n");
        return 1;
    }

    std::string filename;
    if(strcmp(argv[1], "harold") == 0) 
        filename = "../images/harold.jpg";
    else if(strcmp(argv[1], "misha") == 0) 
        filename = "../images/misha_mansoor.jpg";
    else {
        printf("invalid image name\n");
        return 1;
    }

    cv::Mat image = cv::imread(filename);

    

    cv::imwrite("harold.jpg", image);

    return 0;
}
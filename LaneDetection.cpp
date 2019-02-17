#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "HoughTransform.cpp"

#define CUDA 1
#define SEQUENTIAL 2

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "usage LaneDetection inputImage [options]" << endl;
        cout << "   --cuda Perform hough transform using CUDA (default)" << endl;
        cout << "   --seq Perform hough transform sequentially on the CPU" << endl;
        return 1;
    }

    // Read input image
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    // Check which strategy to use for hough transform (CUDA or sequential)
    int houghStrategy = argc > 2 && !strcmp(argv[2], "--seq") ? SEQUENTIAL : CUDA;

    if (image.empty()) {
        cout << "Unable to open image" << endl;
        return -1;
    }

    // Apply necessary pre-processing steps

    // Perform hough transform
    if (houghStrategy == CUDA) {
        houghTransformCuda(image);
    } else if (houghStrategy == SEQUENTIAL) {
        houghTransformSeq(image);
    }

    // Show result image
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);

    waitKey(0);

    return 0;
}

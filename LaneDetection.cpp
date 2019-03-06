#include "commons.h"
#include "HoughTransform.h"
#include "Preprocessing.h"

#define CUDA 1
#define SEQUENTIAL 2


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
    image = filterLanes(image);
    image = applyGaussianBlur(image);
    image = applyCannyEdgeDetection(image);

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

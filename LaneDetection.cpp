#include "commons.h"
#include "HoughTransform.h"

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

    vector<Line> linesFound;

    // Apply necessary pre-processing steps

    // Perform hough transform
    if (houghStrategy == CUDA) {
        houghTransformCuda(image);
    } else if (houghStrategy == SEQUENTIAL) {
        linesFound = houghTransformSeq(image);
    }

    for (int i = 0; i < linesFound.size(); i++) {
        int x1 = 0;
        int x2 = image.cols;
        int y1 = (int)linesFound[i].getY(x1);
        int y2 = (int)linesFound[i].getY(x2);



        line(image, Point(x1, y1), Point(x2, y2), Scalar(150), 1, 8, 0);

    }

    imwrite("res2.png", image);


    // Show result image
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);

    waitKey(0);

    return 0;
}

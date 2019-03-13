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

    // Read input video
    VideoCapture capture(argv[1]);
    // Check which strategy to use for hough transform (CUDA or sequential)
    int houghStrategy = argc > 2 && !strcmp(argv[2], "--seq") ? SEQUENTIAL : CUDA;

    if (!capture.isOpened()) {
        cout << "Unable to open video" << endl;
        return -1;
    }

    VideoWriter video("out.avi", VideoWriter::fourcc('M','J','P','G'), 30, 
                      Size(FRAME_WIDTH, FRAME_HEIGHT), true);

    // Perform hough transform
    if (houghStrategy == CUDA) {
        houghTransformCuda(capture, video);
    } else if (houghStrategy == SEQUENTIAL) {
        houghTransformSeq(capture, video);
    }

    return 0;
}

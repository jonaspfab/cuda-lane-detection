#include "commons.h"
#include "HoughTransform.h"
#include <time.h>

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
    clock_t t;

    // Perform hough transform
    if (houghStrategy == CUDA) {
      cout<<"Processing video with CUDA"<<endl;
      t = clock();
      houghTransformCuda(capture, video);
      t = clock()-t;
      cout<<"Time: "<<(((float)t)/CLOCKS_PER_SEC)<<endl;
    } else if (houghStrategy == SEQUENTIAL) {
      cout<<"Processing video Sequentially"<<endl;
      t = clock();
      houghTransformSeq(capture, video);
      t = clock()-t;
      cout<<"Time: "<<(((float)t)/CLOCKS_PER_SEC)<<endl;
    }

    return 0;
}

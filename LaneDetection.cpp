#include "commons.h"
#include "HoughTransform.h"
#include "Preprocessing.h"

#define CUDA 1
#define SEQUENTIAL 2

/**
 * Detects lines in given image and returns new image with lines drawn into it
 * 
 * @param img Image on from which lines are detected and on which lines are 
 * drawn
 * @param houghStrategy Indicates which strategy shoulds be used to perform the
 * hough transform (1=CUDA, 2=Sequential)
 */
Mat detectLines(Mat img, int houghStrategy) {
    // Apply necessary pre-processing steps
    Mat preProcImg = filterLanes(img);
    preProcImg = applyGaussianBlur(preProcImg);
    preProcImg = applyCannyEdgeDetection(preProcImg);
    preProcImg = regionOfInterest(preProcImg);

    // cvtColor(preProcImg, img, COLOR_GRAY2BGR);
    // return img;

    vector<Line> linesFound;

    // Perform hough transform
    if (houghStrategy == CUDA) {
        linesFound = houghTransformCuda(preProcImg);
    } else if (houghStrategy == SEQUENTIAL) {
        linesFound = houghTransformSeq(preProcImg);
    }
    
    for (int i = 0; i < linesFound.size(); i++) {
        int y1 = img.rows;
        int y2 = (img.rows / 2) + (img.rows / 10);
        int x1 = (int) linesFound[i].getX(y1);
        int x2 = (int) linesFound[i].getX(y2);


        line(img, Point(x1, y1), Point(x2, y2), Scalar(255), 5, 8, 0);
    }

    return img;
}

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

    int frame_width = 1280;
    int frame_height = 720;
    VideoWriter video("out.avi", VideoWriter::fourcc('M','J','P','G'), 30, Size(frame_width,frame_height), true);

    Mat frame;
    for( ; ; ) {
        capture >> frame;
        if(frame.empty())
            break;

        frame = detectLines(frame, houghStrategy);
        video.write(frame);
    }

    return 0;
}

#include "HoughTransform.h"

#define STEP_SIZE 1
#define THRESHOLD 75

extern void drawLines(vector<Line> lines, Mat img);
extern void plotAccumulator(int nRows, int nCols, int *accumulator, const char *dest);
extern __host__ __device__ double calcRho(double x, double y, double theta);

void houghTransformSeq(VideoCapture capture, VideoWriter writer) {
	int nRows = (int) ceil(sqrt(FRAME_HEIGHT * FRAME_HEIGHT + FRAME_WIDTH * FRAME_WIDTH)) * 2;
	int nCols = 180 / STEP_SIZE;

	int *accumulator;
	accumulator = new int[nCols * nRows]();
    vector<Line> lines;

    Mat originalFrame, frame;
    for( ; ; ) {
        capture >> originalFrame;
        if(originalFrame.empty())
            break;

        frame = applyPreprocessing(originalFrame);

        memset(accumulator, 0, nCols * nRows * sizeof(int));
        lines.clear();
        for(int i = 0; i < FRAME_HEIGHT; i++) {
            for (int j = 0; j < FRAME_WIDTH; j++) {
                if ((int) frame.at<uchar>(i, j) == 0)
                    continue;
    
                for (int k = 0; k < nCols; k++) {
                    double theta = ((double) k) * STEP_SIZE;
                    int rho = calcRho(j, i, theta);
    
                    accumulator[(rho + (nRows / 2)) * nCols + k] += 1;
    
                    if(accumulator[(rho + (nRows / 2)) * nCols + k] == THRESHOLD)
                        lines.push_back( Line(theta, rho));
                }
            }
        }

        drawLines(lines, originalFrame);
        writer.write(originalFrame);
    }
}

__global__ void houghKernel(unsigned char* frame, int nRows, int nCols, int *accumulator) {
	int i = blockIdx.x;
	int j = blockIdx.y;
	int theta = threadIdx.x * STEP_SIZE;
	int rho = calcRho(j, i, theta);

	if (((uchar) frame[(j * FRAME_WIDTH) + i]) != 0)
		atomicAdd(&accumulator[(rho + (nRows / 2)) * nCols + threadIdx.x], 1);
}

void houghTransformCuda(VideoCapture capture, VideoWriter writer) {
    int frameSize = FRAME_WIDTH * FRAME_HEIGHT * sizeof(uchar);
	int nRows = (int) ceil(sqrt(FRAME_HEIGHT * FRAME_HEIGHT + FRAME_WIDTH * FRAME_WIDTH)) * 2;
	int nCols = 180 / STEP_SIZE;

	int *accumulator;
	accumulator = new int[nCols * nRows]();
    vector<Line> lines;

	// device space for original image
	uchar *d_frame;
	cudaMalloc<uchar>(&d_frame, frameSize);

	// device space for transformed image
	int *d_accumulator;
	cudaMalloc(&d_accumulator, nRows * nCols * sizeof(int));

	// kernell config
	const dim3 block(180 / STEP_SIZE);
    const dim3 grid(FRAME_HEIGHT, FRAME_WIDTH);

    Mat originalFrame, frame;
    for( ; ; ) {
        capture >> originalFrame;
        if(originalFrame.empty())
            break;

        frame = applyPreprocessing(originalFrame);

        cudaMemcpy(d_frame, frame.ptr(), frameSize, cudaMemcpyHostToDevice);
        cudaMemset(d_accumulator, 0, nRows * nCols * sizeof(int));

        houghKernel<<<grid,block>>>(d_frame, nRows, nCols, d_accumulator);
        cudaDeviceSynchronize();

        cudaMemcpy(accumulator, d_accumulator, nRows * nCols * sizeof(int), cudaMemcpyDeviceToHost);

        cudaError err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString( err ));

        lines.clear();
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                if(accumulator[(i * nCols) + j] >= THRESHOLD){
                    lines.push_back( Line(j, i - (nRows / 2)));
                }
            }
        }

        drawLines(lines, originalFrame);
        writer.write(originalFrame);
    }

	cudaFree(d_frame);
	cudaFree(d_accumulator);
}

void drawLines(vector<Line> lines, Mat img) {
	for (int i = 0; i < lines.size(); i++) {
        int y1 = img.rows;
        int y2 = (img.rows / 2) + (img.rows / 10);
        int x1 = (int) lines[i].getX(y1);
        int x2 = (int) lines[i].getX(y2);

        line(img, Point(x1, y1), Point(x2, y2), Scalar(255), 5, 8, 0);
    }
}

/**
 * Plots 'accumulator' and saves created image to 'dest' (This is for debugging
 * purposes only
 */
void plotAccumulator(int nRows, int nCols, int *accumulator, const char *dest) {
	Mat plotImg(nRows, nCols, CV_8UC1, Scalar(0));
	for (int i = 0; i < nRows; i++) {
  		for (int j = 0; j < nCols; j++) {
			plotImg.at<uchar>(i, j) = min(accumulator[(i * nCols) + j], 255);
  		}
  	}

  	imwrite(dest, plotImg);
}

/**
 * Calculates rho based on the equation r = x cos(θ) + y sin(θ)
 *
 * @param x X coordinate of the pixel
 * @param y Y coordinate of the pixel
 * @param theta Angle between x axis and line connecting origin with closest
 * point on tested line
 *
 * @return Rho describing distance of origin to closest point on tested line
 */
__host__ __device__ double calcRho(double x, double y, double theta) {
	double thetaRadian = (theta * PI) / 180.0;

	return x * cos(thetaRadian) + y * sin(thetaRadian);
}

// __global__ void hough_kernel( unsigned char* img, int icols, int irows,
// 	int* hough, int nCols, int nRows)
// {
// //2D Index of current thread
// int theta = blockIdx.x;
// double thetaRad = ((double)theta*3.14159265358979323846)/180.0;

// for(int i = 0; i < icols; i++) {
// for(int j = 0; j < irows; j++) {
// //Location of gray pixel in output
// int img_id  = (j * icols) + i;

// if (((uchar) img[img_id]) == 0)
// continue;

// int rho = (((double)i*cos(thetaRad)) + ((double)j*sin(thetaRad)));

// hough[(rho + (nRows / 2)) * nCols + theta] += 1;

// }
// }

// }

// __global__ void hough_kernel2( unsigned char* img, int icols, int irows,
// 	int* hough, int nCols, int nRows)
// {
// //2D Index of current thread
// int theta = blockIdx.x;
// double thetaRad = ((double)theta*3.14159265358979323846)/180.0;
// double rho = blockIdx.y - (nRows/2);
// int j;

// for(int i = 0; i < icols; i++) {

// j = (int)((rho - (((double)i)*cos(thetaRad)))/sin(thetaRad));
// if(j>=irows || j<0)
// continue;

// if (((uchar) img[(j * icols) + i]) == 0)
// continue;


// hough[(blockIdx.y) * nCols + blockIdx.x] += 1;

// }
// }

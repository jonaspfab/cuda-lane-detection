#include "HoughTransform.h"

#define STEP_SIZE 0.1
#define THRESHOLD 125
#define THETA_A 45.0
#define THETA_B 135.0
#define THETA_VARIATION 16.0

using namespace thrust;

class Line {
    
private:
    double theta;
    double rho;
    
public:
    
    __host__ __device__  Line(double theta, double rho){
        this->theta = theta;
        this->rho = rho;
    }
    
    __host__ __device__ double getY(double x) {
        double thetaRadian = (theta * PI) / 180.0;
        
        return (rho  - x * cos(thetaRadian)) / sin(thetaRadian);
    }
    
    __host__ __device__ double getX(double y) {
        double thetaRadian = (theta * PI) / 180.0;
        
        return (rho - y * sin(thetaRadian)) / cos(thetaRadian);
    }
};

void drawLine(Line line, Mat img);
extern void plotAccumulator(int nRows, int nCols, int *accumulator, const char *dest);
extern __host__ __device__ double calcRho(double x, double y, double theta);

void houghTransformSeq(VideoCapture capture, VideoWriter writer) {
	int nRows = (int) ceil(sqrt(FRAME_HEIGHT * FRAME_HEIGHT + FRAME_WIDTH * FRAME_WIDTH)) * 2;
	int nCols = 180 / STEP_SIZE;

	int *accumulator;
	accumulator = new int[nCols * nRows]();
	vector<Line> lines;

	Mat originalFrame, frame;

	clock_t loadTime = 0;
	clock_t prepTime = 0;
	clock_t houghTime = 0;
	clock_t drawTime = 0;
	clock_t t;

	for( ; ; ) {
		t = clock();
		capture >> originalFrame;
		loadTime += clock()-t;
		if(originalFrame.empty())
			break;

		t = clock();
		frame = applyPreprocessing(originalFrame);
		prepTime += clock()-t;

		t = clock();
		memset(accumulator, 0, nCols * nRows * sizeof(int));
		lines.clear();
        
        int rho;
        double theta;

		for(int i = 0; i < FRAME_HEIGHT; i++) {
			for (int j = 0; j < FRAME_WIDTH; j++) {
				if ((int) frame.at<uchar>(i, j) == 0)
					continue;
                
                // thetas of interest will be close to 45 and close to 135 (vertical lines)
                // we are doing 2 thetas at a time, 1 for each theta of Interest
                // we use thetas varying 15 degrees more and less
                for(int k = 0; k<2*THETA_VARIATION*(1/STEP_SIZE); k++){
                    theta = THETA_A-THETA_VARIATION + ((double)k*STEP_SIZE);
                    rho = calcRho(j, i, theta);
                    accumulator[(rho + (nRows / 2)) * nCols + (int)(theta/STEP_SIZE)] += 1;
                    
                    if (accumulator[(rho + (nRows / 2)) * nCols + (int)(theta/STEP_SIZE)] == THRESHOLD)
                        lines.push_back( Line(theta, rho));

                    theta = THETA_B-THETA_VARIATION + ((double)k*STEP_SIZE);
                    rho = calcRho(j, i, theta);
                    accumulator[(rho + (nRows / 2)) * nCols + (int)(theta/STEP_SIZE)] += 1;
                    
                    if (accumulator[(rho + (nRows / 2)) * nCols + (int)(theta/STEP_SIZE)] == THRESHOLD)
                        lines.push_back( Line(theta, rho));
                }
			}
		}
		houghTime += clock()-t;

		t = clock();
        for (int i = 0; i < lines.size(); i++)
            drawLine(lines[i], originalFrame);
		writer.write(originalFrame);
		drawTime += clock()-t;
	}
	cout<<"Read Time: "<<(((float)loadTime)/CLOCKS_PER_SEC)<<endl;
	cout<<"Prep Time: "<<(((float)prepTime)/CLOCKS_PER_SEC)<<endl;
	cout<<"Hough Time: "<<(((float)houghTime)/CLOCKS_PER_SEC)<<endl;
	cout<<"Write Time: "<<(((float)drawTime)/CLOCKS_PER_SEC)<<endl;
}

__global__ void houghKernel(unsigned char* frame, int nRows, int nCols, int *accumulator) {
	int i = blockIdx.x;
	int j = blockIdx.y;
	int theta = threadIdx.x * STEP_SIZE;
	int rho = calcRho(j, i, theta);

	if (((uchar) frame[(i * FRAME_WIDTH) + j]) != 0)
		atomicAdd(&accumulator[(rho + (nRows / 2)) * nCols + threadIdx.x], 1);
}

__global__ void houghKernel2(unsigned char* frame, int nRows, int nCols, int *accumulator) {
	int i = (blockIdx.x*blockDim.y)+threadIdx.y;
	int j = (blockIdx.y*blockDim.z)+threadIdx.z;
	double theta;
	int rho;

	if(i<FRAME_HEIGHT && j<FRAME_WIDTH && ((int) frame[(i * FRAME_WIDTH) + j]) != 0) {

		// thetas of interest will be close to 45 and close to 135 (vertical lines)
		// we are doing 2 thetas at a time, 1 for each theta of Interest
		// we use thetas varying 15 degrees more and less
		for(int k = threadIdx.x*(1/STEP_SIZE); k<(threadIdx.x+1)*(1/STEP_SIZE); k++){
			theta = THETA_A-THETA_VARIATION + ((double)k*STEP_SIZE);
			rho = calcRho(j, i, theta);
			atomicAdd(&accumulator[(rho + (nRows / 2)) * nCols + (int)(theta/STEP_SIZE)], 1);

			theta = THETA_B-THETA_VARIATION + ((double)k*STEP_SIZE);
			rho = calcRho(j, i, theta);
			atomicAdd(&accumulator[(rho + (nRows / 2)) * nCols + (int)(theta/STEP_SIZE)], 1);
		}
	}
}

__global__ void findLinesKernel(int nRows, int nCols, int *accumulator, Line *lines, int *lineCounter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (accumulator[i * nCols + j] < THRESHOLD)
        return;

    for (int i_delta = -50; i_delta <= 50; i_delta++) {
        for (int j_delta = -50; j_delta <= 50; j_delta++) {
            if (i + i_delta > 0 && i + i_delta < nRows && j + j_delta > 0 && j + j_delta < nCols &&
                accumulator[(i + i_delta) * nCols + j + j_delta] > accumulator[i * nCols + j]) {
                return;
            }
        }
    }

    int insertPt = atomicAdd(lineCounter, 1);
    lines[insertPt%10] = Line(j*STEP_SIZE, i - (nRows / 2));
}

void houghTransformCuda(VideoCapture capture, VideoWriter writer) {
	int frameSize = FRAME_WIDTH * FRAME_HEIGHT * sizeof(uchar);
	int nRows = (int) ceil(sqrt(FRAME_HEIGHT * FRAME_HEIGHT + FRAME_WIDTH * FRAME_WIDTH)) * 2;
	int nCols = 180 / STEP_SIZE;

    Line *lines;
    lines = (Line *) malloc(10 * sizeof(Line));
    Line *d_lines;
    cudaMalloc(&d_lines, 10 * sizeof(Line));
    int lineCounter = 0;
    int *d_lineCounter;
    cudaMalloc(&d_lineCounter, sizeof(int));

	// device space for original image
	uchar *d_frame;
	cudaMalloc<uchar>(&d_frame, frameSize);

	// device space for transformed image
	// TODO: we can reduce the the accumulator to accomodate only the thetas of interest
	int *d_accumulator;
	cudaMalloc(&d_accumulator, nRows * nCols * sizeof(int));

	// kernell config 1
	// const dim3 block(180 / STEP_SIZE);
	// const dim3 grid(FRAME_HEIGHT, FRAME_WIDTH);
	// kernell config 2
	const dim3 block(32, 5, 5);
	const dim3 grid(ceil(FRAME_HEIGHT/5), ceil(FRAME_WIDTH/5));
    
    const dim3 findLinesBlock(32, 32);
    const dim3 findLinesGrid(ceil(nRows / 32), ceil(nCols / 32));

	Mat originalFrame, frame;

	clock_t loadTime = 0;
	clock_t prepTime = 0;
	clock_t houghTime = 0;
	clock_t drawTime = 0;
	clock_t t;

	for( ; ; ) {
		t = clock();
		capture >> originalFrame;
		loadTime += clock()-t;

		if(originalFrame.empty()){
			break;
		}

		t = clock();
		frame = applyPreprocessing(originalFrame);
		prepTime += clock()-t;

		t = clock();
		cudaMemcpy(d_frame, frame.ptr(), frameSize, cudaMemcpyHostToDevice);
		cudaMemset(d_accumulator, 0, nRows * nCols * sizeof(int));

		houghKernel2<<<grid,block>>>(d_frame, nRows, nCols, d_accumulator);
		cudaDeviceSynchronize();

		cudaError err = cudaGetLastError();
		if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString( err ));
        
        cudaMemset(d_lineCounter, 0, sizeof(int));

        findLinesKernel<<<findLinesGrid, findLinesBlock>>>(nRows, nCols, d_accumulator, d_lines, d_lineCounter);
        cudaDeviceSynchronize();
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error2: %s\n", cudaGetErrorString( err ));
        
        cudaMemcpy(&lineCounter, d_lineCounter, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(lines, d_lines, 10 * sizeof(Line), cudaMemcpyDeviceToHost);

		houghTime += clock()-t;

		t = clock();
        for (int i = 0; i < lineCounter; i++)
            drawLine(lines[i], originalFrame);
        writer << originalFrame;
		drawTime += clock()-t;
	}

	cout<<"Read Time: "<<(((float)loadTime)/CLOCKS_PER_SEC)<<endl;
	cout<<"Prep Time: "<<(((float)prepTime)/CLOCKS_PER_SEC)<<endl;
	cout<<"Hough Time: "<<(((float)houghTime)/CLOCKS_PER_SEC)<<endl;
	cout<<"Write Time: "<<(((float)drawTime)/CLOCKS_PER_SEC)<<endl;

	cudaFree(d_frame);
	cudaFree(d_accumulator);
}

void drawLine(Line l, Mat img) {
    int y1 = img.rows;
    int y2 = (img.rows / 2) + (img.rows / 10);
    int x1 = (int) l.getX(y1);
    int x2 = (int) l.getX(y2);

    line(img, Point(x1, y1), Point(x2, y2), Scalar(255), 5, 8, 0);
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

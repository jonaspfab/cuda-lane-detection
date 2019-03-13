#include "HoughTransform.h"

#define STEP_SIZE 1
#define THRESHOLD 75

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
double calcRho(double x, double y, double theta) {
	double thetaRadian = (theta * PI) / 180.0;

	return x * cos(thetaRadian) + y * sin(thetaRadian);
}



/**
 * Performs sequential hough transform on given image
 *
 * @param img Input image on which hough transform is performed
 */
vector<Line> houghTransformSeq(Mat img) {
	int nRows = (int) ceil(sqrt(img.rows * img.rows + img.cols * img.cols)) * 2;
	int nCols = 180 / STEP_SIZE;

	int *accumulator;
	accumulator = new int[nCols * nRows]();
	vector<Line> lines;

	for(int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
   		if ((int) img.at<uchar>(i, j) == 0)
   			continue;

   		for (int k = 0; k < nCols; k++) {
   			double theta = ((double) k) * STEP_SIZE;

				int rho = calcRho(j, i, theta);

				accumulator[(rho + (nRows / 2)) * nCols + k] += 1;

				if(accumulator[(rho + (nRows / 2)) * nCols + k] == THRESHOLD){
					lines.push_back( Line(theta, rho));
				}

   		}
		}
	}

	// plotAccumulator(nRows, nCols, accumulator, "./res.jpg");

	return lines;

}


__global__ void hough_kernel( unsigned char* img, int icols, int irows,
															int* hough, int nCols, int nRows)
{
	//2D Index of current thread
	int theta = blockIdx.x;
	double thetaRad = ((double)theta*3.14159265358979323846)/180.0;

	for(int i = 0; i < icols; i++) {
		for(int j = 0; j < irows; j++) {
			//Location of gray pixel in output
			int img_id  = (j * icols) + i;

   		if (((uchar) img[img_id]) == 0)
   			continue;

			int rho = (((double)i*cos(thetaRad)) + ((double)j*sin(thetaRad)));

			hough[(rho + (nRows / 2)) * nCols + theta] += 1;

		}
	}

}

__global__ void hough_kernel2( unsigned char* img, int icols, int irows,
															int* hough, int nCols, int nRows)
{
	//2D Index of current thread
	int theta = blockIdx.x;
	double thetaRad = ((double)theta*3.14159265358979323846)/180.0;
	double rho = blockIdx.y - (nRows/2);
	int j;

	for(int i = 0; i < icols; i++) {

		j = (int)((rho - (((double)i)*cos(thetaRad)))/sin(thetaRad));
		if(j>=irows || j<0)
			continue;

   		if (((uchar) img[(j * icols) + i]) == 0)
   			continue;


		hough[(blockIdx.y) * nCols + blockIdx.x] += 1;

	}

}

/**
 * Performs hough transform on given image using CUDA
 *
 * @param img Input image on which hough transform is performed
 */
vector<Line> houghTransformCuda(Mat img) {
	int isize = img.cols * img.rows * sizeof(uchar);
	int nRows = (int) ceil(sqrt(img.rows * img.rows + img.cols * img.cols)) * 2;
	int nCols = 180 / STEP_SIZE;

	vector<Line> lines;

	int *accumulator;
	accumulator = new int[nCols * nRows]();

	// device space for original image
	uchar *d_img;
	cudaMalloc<uchar>(&d_img, isize);
	cudaMemcpy(d_img,img.ptr(),isize,cudaMemcpyHostToDevice);

	// device space for transformed image
	int *d_hough;
	cudaMalloc(&d_hough, nRows*nCols*sizeof(int));
	cudaMemcpy(d_hough,accumulator,nRows*nCols*sizeof(int),cudaMemcpyHostToDevice);

	// kernell config
	const dim3 block(1, 1);
	const dim3 grid(nCols, 1);

	hough_kernel<<<grid,block>>>(d_img, img.cols, img.rows, d_hough, nCols, nRows);
	cudaDeviceSynchronize();

	cudaError err = cudaGetLastError();
	if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString( err ));

	cudaMemcpy(accumulator,d_hough,nRows*nCols*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_img);
	cudaFree(d_hough);

	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			if(accumulator[(i * nCols) + j] >= THRESHOLD){
				lines.push_back( Line(j, i-(nRows / 2)));
			}
		}
	}
	// plotAccumulator(nRows, nCols, accumulator, "./res-cuda.jpg");

	return lines;
}

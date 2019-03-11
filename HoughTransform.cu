#include "HoughTransform.h"

#define STEP_SIZE 1
#define THRESHOLD 200

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

				if(accumulator[(rho + (nRows / 2)) * nCols + k] == THRESHOLD)
					lines.push_back( Line(theta, rho));

   		}
		}
	}

	plotAccumulator(nRows, nCols, accumulator, "./res.jpg");

	return lines;

}

__global__ void hough_kernel( unsigned char* img, int width, int height, int grayWidthStep,
															int* hough, int nRows, int nCols)
{
	//2D Index of current thread
	double theta = ((double)threadIdx.x*3.14159265358979323846)/180.0;

	for(int i = 0; i < width; i++) {
		for(int j = 0; j < height; j++) {
			//Location of gray pixel in output
			int gray_tid  = j * grayWidthStep + i;

   		if (((int) img[gray_tid]) == 0)
   			continue;

			int rho = ((double)j*cos(theta) + (double)i*sin(theta));

			hough[(rho + (nRows / 2)) * nCols + threadIdx.x] += 1;

		}
	}

}

/**
 * Performs hough transform on given image using CUDA
 *
 * @param img Input image on which hough transform is performed
 */
vector<Line> houghTransformCuda(Mat img) {
	const int grayBytes = img.step * img.rows;
	vector<Line> lines;

	int nRows = (int) ceil(sqrt(img.rows * img.rows + img.cols * img.cols)) * 2;
	int nCols = 180 / STEP_SIZE;

	int *accumulator;
	accumulator = new int[nCols * nRows]();

	// device space for original image
	unsigned char *d_img;
	cudaMalloc<unsigned char>(&d_img,grayBytes);
	cudaMemcpy(d_img,img.ptr(),grayBytes,cudaMemcpyHostToDevice);

	// device space for transformed image
	int *d_hough;
	cudaMalloc(&d_hough, nRows*nCols*sizeof(int));
	cudaMemcpy(d_hough,accumulator,nRows*nCols*sizeof(int),cudaMemcpyHostToDevice);


	//Specify a reasonable block size
	const dim3 block(1, 1);

	//Calculate grid size to cover the whole image
	const dim3 grid(nCols, 1);

	hough_kernel<<<grid,block>>>(d_img, img.cols, img.rows, img.step, d_hough, nRows, nCols);

	cudaDeviceSynchronize();

	cudaMemcpy(accumulator,d_hough,nRows*nCols*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_img);
	cudaFree(d_hough);

	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			if(accumulator[(i + (nRows / 2)) * nCols + j] == THRESHOLD)
				lines.push_back( Line(j, i));
		}
	}

	return lines;
}

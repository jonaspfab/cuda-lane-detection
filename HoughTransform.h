#include "commons.h"

/**
 * Performs sequential hough transform on given image
 *
 * @param image Input image on which hough transform is performed
 */
void houghTransformSeq(Mat image) ;

/**
 * Performs hough transform on given image using CUDA
 *
 * @param image Input image on which hough transform is performed
 */
void houghTransformCuda(Mat image) ;

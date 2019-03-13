#include "commons.h"

/**
 * Filters white and yellow color range to filter out lane markers
 * 
 * @param img Image from which lane markers are filtered from
 */
Mat filterLanes(Mat img);

Mat applyGaussianBlur(Mat img);

Mat applyCannyEdgeDetection(Mat img);

Mat regionOfInterest(Mat img);

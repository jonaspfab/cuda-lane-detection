#include <math.h>
#include "Line.h"
#include "Preprocessing.h"
#include "commons.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void houghTransformSeq(VideoCapture capture, VideoWriter writer);

void houghTransformCuda(VideoCapture capture, VideoWriter writer) ;

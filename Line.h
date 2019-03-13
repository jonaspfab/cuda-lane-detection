#ifndef __LINE_H__
#define __LINE_H__
#include "commons.h"


class Line {

private:
	double theta;
	double rho;

public:

	Line(double theta, double rho);

	double getY(double x);

	double getX(double y);
};


#endif
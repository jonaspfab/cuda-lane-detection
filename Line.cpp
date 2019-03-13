#include "Line.h"


Line::Line(double theta, double rho){
	this->theta = theta;
	this->rho = rho;
}

double Line::getY(double x) {
	double thetaRadian = (theta * PI) / 180.0;

	return (rho  - x * cos(thetaRadian)) / sin(thetaRadian);
}

double Line::getX(double y) {
	double thetaRadian = (theta * PI) / 180.0;

	return (rho - y * sin(thetaRadian)) / cos(thetaRadian);
}

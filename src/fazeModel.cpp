#include <math.h>
#include <stdlib.h>
#include <string>

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/viz.hpp"

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/gui_widgets.h"

#include "fixedBin.h"
#include "fazeModel.h"
#include "fazeStream.h"
#include "util.h"
#include "pupilDetectionCDF.h"
#include "pupilDetectionSP.h"

#ifndef PI
#define PI 3.14
#endif

void preprocessROI(cv::Mat& roi_eye) {
	GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
	equalizeHist( roi_eye, roi_eye );
}

double findSigma(int ln, int lf, double Rn, double theta) {
	double dz=0;
	double sigma;
	double m1 = ((double)ln*ln)/((double)lf*lf);
	double m2 = (cos(theta))*(cos(theta));

	if (m2 == 1)
	{
		dz = sqrt(	(Rn*Rn)/(m1 + (Rn*Rn))	);
	}
	if (m2>=0 && m2<1)
	{
		dz = sqrt(	((Rn*Rn) - m1 - 2*m2*(Rn*Rn) + sqrt(	((m1-(Rn*Rn))*(m1-(Rn*Rn))) + 4*m1*m2*(Rn*Rn)	))/ (2*(1-m2)*(Rn*Rn))	);
	}
	sigma = acos(dz);
	return sigma;
}

Faze::Faze() {
	Rn = 0.5;
	Rm = 0.5;

	MAG_NOR = 12.0;
	MAG_CR = 12.0;
	MAG_LR = 30.0;
	MAG_CP = 13.101;
	MAG_CM = 13.101;
	ALPHA = 30.0;
	THETA = 1.97920337176;

	origin.x = 0;
	origin.y = 0;
}

void Faze::assign(dlib::full_object_detection shape , cv::Mat image, int modePupil, int modeGaze) {
	assert(modePupil == MODE_PUPIL_SP || modePupil == MODE_PUPIL_CDF ||
		modeGaze == MODE_GAZE_VA || modeGaze == MODE_GAZE_QE);
	faceShape = shape;
	image.copyTo(imageColor);
	cv::cvtColor(imageColor, imageGray, CV_BGR2GRAY);

	descriptors.clear();
	normal.clear();
	normal.resize(3);
	localYAxis.clear();
	localYAxis.resize(3);

	computePupil(modePupil);
	computeFacialParams();
	computeGaze(modeGaze);
}

void Faze::computePupil(int mode) {
	assert(mode == MODE_PUPIL_SP || mode == MODE_PUPIL_CDF);

	std::vector<cv::Point> leftEyePoints = getDescriptors(INDEX_LEFT_EYE);
	cv::Rect rectLeftEye = cv::boundingRect(leftEyePoints);
	cv::Mat roiLeftEye = imageGray(rectLeftEye);
	preprocessROI(roiLeftEye);

	std::vector<cv::Point> rightEyePoints = getDescriptors(INDEX_RIGHT_EYE);
	cv::Rect rectRightEye = cv::boundingRect(rightEyePoints);
	cv::Mat roiRightEye = imageGray(rectRightEye);
	preprocessROI(roiRightEye);

	if(mode == MODE_PUPIL_SP) {
		descriptors.push_back(get_pupil_coordinates(roiLeftEye,rectLeftEye));
		descriptors.push_back(get_pupil_coordinates(roiRightEye,rectRightEye));
	}
	else {
		descriptors.push_back(computePupilCDF(roiLeftEye));
		descriptors.push_back(computePupilCDF(roiRightEye));
	}
}

dlib::full_object_detection Faze::getShape() {
	return faceShape;
}

void Faze::computeFacialParams() {
	cv::Point midEye = get_mid_point(cv::Point(faceShape.part(39).x(), faceShape.part(39).y()),
		cv::Point(faceShape.part(40).x(), faceShape.part(40).y()));

	cv::Point mouth = get_mid_point(cv::Point(faceShape.part(48).x(), faceShape.part(48).y()),
		cv::Point(faceShape.part(54).x(), faceShape.part(54).y()));

	cv::Point noseTip = cv::Point(faceShape.part(30).x(), faceShape.part(30).y());
	cv::Point noseBase = cv::Point(faceShape.part(33).x(), faceShape.part(33).y());
	cv::Point noseTopTip = cv::Point(faceShape.part(27).x(), faceShape.part(27).y());

	cv::Point ly = noseTopTip - noseTip;

	// symm angle - angle between the symmetry axis and the 'x' axis
	symm_x = get_angle_between(noseBase, midEye);
	// tilt angle - angle between normal in image and 'x' axis
	tau = get_angle_between(noseBase, noseTip);
	// theta angle - angle between the symmetry axis and the image normal
	theta = (abs(tau - symm_x)) * (PI/180.0);

	// sigma - slant angle
	sigma = findSigma(get_distance(noseTip, noseBase), get_distance(midEye, mouth), Rn, theta);

	normal[0] = (sin(sigma))*(cos((360 - tau)*(PI/180.0)));
	normal[1] = (sin(sigma))*(sin((360 - tau)*(PI/180.0)));
	normal[2] = -cos(sigma);

	double ly_z, mag;
	if( !normal[2] ) {
		ly_z = 0.0;
		mag = std::sqrt(ly.x*ly.x + ly.y*ly.y);
		localYAxis[0] = ly.x/mag;
		localYAxis[1] = ly.y/mag;
		localYAxis[2] = ly_z;
	}
	else {
		ly_z = -1.0*(ly.x*normal[0] + ly.y*normal[1]);
		ly_z /= normal[2];
		mag = std::sqrt(ly.x*ly.x + ly.y*ly.y + ly_z*ly_z);
		localYAxis[0] = ly.x/mag;
		localYAxis[1] = ly.y/mag;
		localYAxis[2] = ly_z/mag;
	}

	std::cout << localYAxis[0] << " " << localYAxis[1] << " " << localYAxis[2] << std::endl;

	pitch = acos(sqrt((normal[0]*normal[0] + normal[2]*normal[2])/(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])));
	if((noseTip.y - noseBase.y) < 0) {
		pitch = -pitch;
	}

	yaw = acos((abs(normal[2]))/(sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])));
	if((noseTip.x - noseBase.x) < 0) {
		yaw = -yaw;
	}
}

void Faze::computeGaze(int mode) {
	assert(mode == MODE_GAZE_VA || mode == MODE_GAZE_QE);
	if(mode == MODE_GAZE_QE) {
		// TODO : fill this
			//computeGazeGE();
	}
	else {
			//computeGazeVA(this, ALPHA, MAG_NOR);
	}
}

void Faze::setOrigin(cv::Point origin) {
	this->origin = origin;
}

void Faze::setOrigin(int mode) {
	assert(mode == ORIGIN_IMAGE || mode == ORIGIN_FACE_CENTRE);

	if (mode == ORIGIN_IMAGE) {
		origin.x = 0;
		origin.y = 0;
	}
	else if (mode == ORIGIN_FACE_CENTRE) {
		origin.x = faceShape.part(30).x();
		origin.y = faceShape.part(30).y();
	}
}

std::vector<double> Faze::getNormal() {
	return normal;
}

cv::viz::WPlane Faze::getFacialPlane(cv::Size2d size, cv::viz::Color color) {
	return cv::viz::WPlane(cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(normal[0], normal[1], normal[2]),
												 cv::Vec3d(localYAxis[0], localYAxis[1], localYAxis[2]), size, color);
}

cv::Point Faze::getPupil(int mode) {
	assert(mode == INDEX_LEFT_EYE_PUPIL || mode == INDEX_RIGHT_EYE_PUPIL);
	return descriptors[mode - INDEX_LEFT_EYE_PUPIL];
}

std::vector<double> Faze::getGaze() {
	return gaze;
}

std::vector<cv::Point> Faze::getIntermediateDescriptors(int index) {
	assert(index == INDEX_LEFT_EYE || index == INDEX_RIGHT_EYE || index == INDEX_LEFT_EYE_BROW || index == INDEX_RIGHT_EYE_BROW
		|| index == INDEX_NOSE_UPPER || index == INDEX_NOSE_LOWER || index == INDEX_MOUTH_OUTER || index == INDEX_MOUTH_INNER);

	if (index == INDEX_LEFT_EYE) {
		std::vector<cv::Point> leftEyePoints;
		for (int i=36; i<=41; i++){
			leftEyePoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return leftEyePoints;
	}
	else if (index == INDEX_RIGHT_EYE) {
		std::vector<cv::Point> rightEyePoints;
		for (int i=42; i<=47; i++){
			rightEyePoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return rightEyePoints;
	}
	else if (index == INDEX_LEFT_EYEBROW) {
		std::vector<cv::Point> leftEyeBrowPoints;
		for (int i=17; i<=21; i++){
			leftEyeBrowPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return leftEyeBrowPoints;
	}
	else if (index == INDEX_RIGHT_EYEBROW) {
		std::vector<cv::Point> rightEyeBrowPoints;
		for (int i=22; i<=26; i++){
			rightEyeBrowPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return rightEyeBrowPoints;
	}
	else if (index == INDEX_NOSE_UPPER)  {
		std::vector<cv::Point> noseUpperPoints;
		for (int i=27; i<=30; i++){
			noseUpperPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return noseUpperPoints;
	}
	else if (index == INDEX_NOSE_LOWER) {
		std::vector<cv::Point> noseLowerPoints;
		for (int i=31; i<=35; i++){
			noseLowerPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return noseLowerPoints;
	}
	else if (index == INDEX_MOUTH_OUTER) {
		std::vector<cv::Point> mouthOuterPoints;
		for (int i=48; i<59; i++){
			mouthOuterPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return mouthOuterPoints;
	}
	else if (index == INDEX_MOUTH_INNER) {
		std::vector<cv::Point> mouthInnerPoints;
		for (int i=60; i<=67; i++){
			mouthInnerPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return mouthInnerPoints;
	}
}

void Faze::relativeToOrigin(std::vector<cv::Point>& vec) {
	for(int i=0; i<(int)vec.size(); ++i) {
		vec[i].x -= origin.x;
		vec[i].y -= origin.y;
	}
}

std::vector<cv::Point> Faze::getDescriptors(int index, int mode) {
	assert(mode == DESCRIPTOR_GLOBAL || mode == DESCRIPTOR_LOCAL);
	std::vector<cv::Point> vec = getIntermediateDescriptors(index);
	if(mode == DESCRIPTOR_LOCAL)
		relativeToOrigin(vec);
	return vec;
}

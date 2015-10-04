#include <math.h>
#include <stdlib.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/gui_widgets.h"

#include "fazeModel.h"
#include "util.h"
#include "pupilDetectionCDF.h"
#include "pupilDetectionSP.h"

#ifndef PI
#define PI 3.14
#endif

Stream::Stream(int degree, int smooth) {
	assert(smooth == SMOOTH_AVG || smooth == SMOOTH_KALMAN);
	assert(degree > 0);

	_smooth = smooth;
	_bin = FixedBin();
	_bin.assign(degree);


	_kalmanFilter((_degree+1)*3, (_degree+1)*3)
	_measurements((_degree+1)*3, 1);
	_measurementsOld((_degree+1)*3, 1);
}

void Stream::push(Faze faze) {
	_bin.push(faze);
	int s = (_degree+1)*3;

	if(_bin.filled == 1) {
		std::vector<double> vec = faze.getNormal();
		for(int i=0; i<s;i+=3) {
			if(i==0) {
				_kalmanFilter.statePre.at<float>(i) = vec[i];
				_kalmanFilter.statePre.at<float>(i+1) = vec[i+1];
				_kalmanFilter.statePre.at<float>(i+2) = vec[i+2];
			}
			else {
				_kalmanFilter.statePre.at<float>(i) = 0;
				_kalmanFilter.statePre.at<float>(i+1) = 0;
				_kalmanFilter.statePre.at<float>(i+2) = 0;
			}
		}

		cv::Mat_<float>(s, s) pNC;
		for(int i=0; i<s; ++i) {
			for(int j=0; j<s; ++j) {
				tM<<((float)(rand()))/RAND_MAX;
			}
		}
		_kalmanFilter.processNoiseCov = *pNC;

		cv::Mat_<float>(s, s) tM;
		for(int i=0; i<=_degree; ++i) {
			for(int k=0; k<i; ++k) {
				tM<<0,0,0;
			}
			for(int j=i; j<=_degree; ++j) {
				float ff = fact(_degree-j);
				tM<<1.0/ff,0,0;
			}
			for(int k=0; k<i; ++k) {
				tM<<0,0,0;
			}
			for(int j=i; j<=_degree; ++j) {
				float ff = fact(_degree-j);
				tM<<0,1.0/ff,0;
			}
			for(int k=0; k<i; ++k) {
				tM<<0,0,0;
			}
			for(int j=i; j<=_degree; ++j) {
				float ff = fact(_degree-j);
				tM<<0,0,1.0/ff;
			}
		}

		_kalmanFilter.transitionMatrix = *tM;

		cv::setIdentity(_kalmanFilter.measurementMatrix);
		cv::setIdentity(_kalmanFilter.processNoiseCov,cv::Scalar::all(1e-4));
		cv::setIdentity(_kalmanFilter.measurementNoiseCov,cv::Scalar::all(1e-1));
		cv::setIdentity(_kalmanFilter.errorCovPost, cv::Scalar::all(.1));  
	}
	else {
		cv::Mat prediction = _kalmanFilter.predict();
		for(int i=0; i<s; i+=3) {
			if(i==0) {
				_measurements[i] = vec[0];
				_measurements[i+1] = vec[1];
				_measurements[i+2] = vec[2];
			}
			else {
				_measurements[i] = _measurements[i-3] - _measurementsOld[i-3];
				_measurements[i+1] = _measurements[i-2] - _measurementsOld[i-2];
				_measurements[i+2] = _measurements[i-1] - _measurementsOld[i-1];
			}
		}

		cv::Mat estimated = _kalmanFilter.correct(_measurements);
		std::vector<double> normal(3);
		normal[0] = estimated.at<float>(0);
		normal[1] = estimated.at<float>(1);
		normal[2] = estimated.at<float>(2);

		// normal - corrected normal
	}
}

int Stream::filled() {
	return _bin.filled();
}

int degree() {
	return _degree;
}

int smooth() {
	return _smooth;
}
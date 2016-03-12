#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/video.hpp"
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

int fact(int n) {
	assert(n>=0);
	if(n==0) return 1;
	else return n*fact(n-1);
}

Stream::Stream(int degree, int smooth) {
	assert(smooth == SMOOTH_AVG || smooth == SMOOTH_KALMAN);
	assert(degree > 0);

	_smooth = smooth;
	_bin = FixedBin<Faze>();
	_bin.assign(degree);

	_kalmanFilter = cv::KalmanFilter((_degree+1)*3, (_degree+1)*3, 0);
	_measurements((_degree+1)*3, 1);
	_measurementsOld((_degree+1)*3, 1);
}

void Stream::push(Faze faze) {
	_bin.push(faze);
	int s = (_degree+1)*3;
	std::vector<double> vec = faze.getNormal();

	if(_smooth == SMOOTH_KALMAN) {
		if(_bin.filled() == 1) {
			_faze.normal = vec;
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

			// cv::Mat_<float> pNC(s, s);
			// for(int i=0; i<s; ++i) {
			// 	for(int j=0; j<s; ++j) {
			// 		pNC<<((float)(rand()))/RAND_MAX;
			// 	}
			// }std::cout<<"L74"<<std::endl;
			// _kalmanFilter.processNoiseCov = pNC.clone();

			std::cout<<"L77"<<std::endl;

			cv::Mat_<float> tM(s, s);
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

			std::cout<<"L104"<<std::endl;
			_kalmanFilter.transitionMatrix = (cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
			std::cout<<"L106"<<std::endl;

			cv::setIdentity(_kalmanFilter.measurementMatrix);
			cv::setIdentity(_kalmanFilter.processNoiseCov,cv::Scalar::all(1e-4));
			cv::setIdentity(_kalmanFilter.measurementNoiseCov,cv::Scalar::all(1e-1));
			cv::setIdentity(_kalmanFilter.errorCovPost, cv::Scalar::all(.1));
			std::cout<<"L112"<<std::endl;
		}
		else {
			std::cout<<"L115"<<std::endl;
			cv::Mat prediction = _kalmanFilter.predict();
			std::cout << "hello";
			for(int i=0; i<s; i+=3) {
				if(i==0) {
					_measurements(i) = vec[0];
					_measurements(i+1) = vec[1];
					_measurements(i+2) = vec[2];
				}
				else {
					_measurements(i) = _measurements(i-3) - _measurementsOld(i-3);
					_measurements(i+1) = _measurements(i-2) - _measurementsOld(i-2);
					_measurements(i+2) = _measurements(i-1) - _measurementsOld(i-1);
				}
			}

			cv::Mat estimated = _kalmanFilter.correct(_measurements);
			std::vector<double> normal(3);
			normal[0] = static_cast<double>(estimated.at<float>(0));
			normal[1] = static_cast<double>(estimated.at<float>(1));
			normal[2] = static_cast<double>(estimated.at<float>(2));

			_faze.normal = normal;
		// normal - corrected normal
		}
	}
	else if(_smooth == SMOOTH_AVG) {
		// Smoothing by moving average
	}
}

Faze Stream::current() {
	return _faze;
}

int Stream::filled() {
	return _bin.filled();
}

int Stream::degree() {
	return _degree;
}

int Stream::smooth() {
	return _smooth;
}

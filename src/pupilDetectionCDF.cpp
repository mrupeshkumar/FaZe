#include <math.h>
#include <stdlib.h>

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/viz.hpp"

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/gui_widgets.h"

cv::Point computePupilCDF(cv::Mat roi) {

	std::vector<double> cdf(256);
	//Preprocessing
	GaussianBlur(roi, roi, cv::Size(3,3), 0, 0);
	cv::equalizeHist(roi, roi);

	cv::Mat mask;
	cv::Point pt_pupil;
	roi.copyTo(mask);
	double nf, temp, pos_pmi_i, pos_pmi_j;

	int erosion_size = 1;
	cv::Mat element_erode = cv::getStructuringElement( cv::MORPH_ELLIPSE,
		cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		cv::Point( erosion_size, erosion_size ) );

	for(int i = 0; i<256; i++) {
		cdf[i] = 0;
	}

	for(int i = 0; i < roi.rows; i++) {
		for(int j = 0; j < roi.cols; j++) {
			++cdf[roi.at<uchar>(i,j)];
		}
	}

	for(int i=0; i < 256; i++) {
		double value;
		if(i != 0) {
			value = cdf[i];
			cdf[i] += cdf[i-1];
		}
		//std::cout<<"CDF-"<<i<<" = "<<cdf[i]<<std::endl;
	}

	nf = cdf[0];
	for(int i=1; i<256;i++) {
		if(cdf[i] > nf) {
			nf = cdf[i];
		}
	}

	temp = roi.at<uchar>(0,0);
	pos_pmi_i = 0;
	pos_pmi_j = 0;

	cv::erode( mask, mask, element_erode );

	for(int i = 0; i < roi.rows; i++) {
		for(int j = 0; j < roi.cols; j++) {
			if(cdf[roi.at<uchar>(i,j)] >= 0.05 * nf) {
				mask.at<uchar>(i,j) = 0;
				//roi.at<uchar>(i,j) = 255;
			}
			else {
				if(roi.at<uchar>(i,j) <= temp) {
					pos_pmi_i = i;
					pos_pmi_j = j;
					temp = roi.at<uchar>(i,j);
				}
				mask.at<uchar>(i,j) = 255;
				//roi.at<uchar>(i,j) = 255;
			}
		}
	}

	double avg_PI = 0;
	int window_size;
	window_size = roi.cols*roi.rows/175;
	std::cout<<"window_size : "<<window_size<<std::endl;

	for(int i = pos_pmi_i - window_size; i < pos_pmi_i + window_size; i++) {
		for(int j = pos_pmi_j - window_size; j < pos_pmi_j + window_size; j++) {
			if(mask.at<uchar>(i,j)) {
				avg_PI += roi.at<uchar>(i,j);
			}
		}
	}

	for(int i = pos_pmi_i - window_size; i < pos_pmi_i + window_size; i++) {
		for(int j = pos_pmi_j - window_size; j < pos_pmi_j + window_size; j++) {
			if(roi.at<uchar>(i,j) > ((int)avg_PI)) {
				mask.at<uchar>(i,j) = 0;
			}
		}
	}

	cv::erode( mask, mask, element_erode );

	cv::Moments m = cv::moments(mask, 1);
	int pos_i = (int)(m.m10/m.m00), pos_j = (int)(m.m01/m.m00);

	std::cout<<"PMI : "<<pos_pmi_i<<", "<<pos_pmi_j<<std::endl;
	std::cout<<"Point : "<<pos_i<<", "<<pos_j<<std::endl;

	pt_pupil.x = pos_i;
	pt_pupil.y = pos_j;

	return pt_pupil;
}

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

#include "fazeModel.h"

using namespace std;
using namespace dlib;

void log_vec(std::string str, std::vector<double> vec) {
	std::cout<<str<<" : "<<vec[0]<<" "<<vec[1]<<" "<<vec[2]<<std::endl;
}

void cross_product(std::vector<double> vec1, std::vector<double> vec2, std::vector<double>& product) {
	product[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
	product[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
	product[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
}

void read_vector_from_file(char* file_name, std::vector<std::vector<double> >& arr) {

	/*
		Function to read a vector of measurements from a text file.

		@params:
		file_name	Name of the file to be read
		arr 		Vector to be constructed, passed by reference
	*/

	std::ifstream file_in(file_name);
	std::string line;

	while(std::getline(file_in, line)) {
		std::istringstream iss(line);
		std::vector<double> vec(3);

		if(!(iss >> vec[0] >> vec[1] >> vec[2])) {
			break;
		}

		//std::cout<<vec[0]<<" "<<vec[1]<<" "<<vec[2]<<std::endl;
		arr.push_back(vec);
	}
}

void blow_up_rect(cv::Rect& rect, double f) {
	rect.x = rect.x - (rect.width*(f - 1))/2.0;
	rect.y = rect.y - (rect.height*(f - 1))/2.0;
	rect.width = ((double)rect.width)*f;
	rect.height = ((double)rect.height)*f;
}

void show_images(int e ,int l, int h, std::vector<cv::Mat> imgs) {
	for(int i=l;i<=h;i++) {
		char str[2];
		str[0] = (char)(i+49+e*(e+2));
		str[1] = '\0';
		cv::imshow(str, imgs[i]);
	}
}

double get_distance(cv::Point p1, cv::Point p2) {
	double x = p1.x - p2.x;
	double y = p1.y - p2.y;

	return sqrt(x*x + y*y);
}

cv::Point get_mid_point(cv::Point p1, cv::Point p2) {
	return cv::Point((p1.x + p2.x)/2.0, (p1.y + p2.y)/2.0);
}

double get_vector_magnitude(double vec[], int size) {
	double mag = 0;

	for(int i=0;i<size;i++) {
		mag += vec[i]*vec[i];
	}

	return sqrt(mag);
}

void compute_vector_sum(std::vector<double> vec1, std::vector<double> vec2, std::vector<double>& vec_sum) {
	vec_sum[0] = (vec1[0] + vec2[0]);
	vec_sum[1] = (vec1[1] + vec2[1]);
	vec_sum[2] = (vec1[2] + vec2[2]);
}

double get_angle_between(cv::Point pt1, cv::Point pt2) {
	return 360 - cvFastArctan(pt2.y - pt1.y, pt2.x - pt1.x);
}

void make_unit_vector(std::vector<double> vec, std::vector<double>& unit_vector) {

	double magnitude = 0;

	for(int i=0;i<vec.size();i++) {
		magnitude += vec[i]*vec[i];
	}
	magnitude = sqrt(magnitude);

	for(int i=0;i<vec.size();i++) {
		unit_vector[i] = (((double)(vec[i])/magnitude));
	}
}

double scalar_product(std::vector<double> vec1, std::vector<double> vec2) {
	double dot = 0;

	if(vec1.size() != vec2.size()) {
		return 0;
	}

	for(int i=0;i<vec1.size();i++) {
		dot += vec1[i]*vec2[i];
	}

	return dot;
}

cv::Mat get_rotation_matrix_z(double theta) {
	cv::Mat rot_matrix(3,3, CV_64F);

	double sinx = sin(theta);
	double cosx = cos(theta);

	double* col = rot_matrix.ptr<double>(0);
	col[0] = cosx;
	col[1] = sinx;
	col[2] = 0;

	col = rot_matrix.ptr<double>(1);
	col[0] = -sinx;
	col[1] = cosx;
	col[2] = 0;

	col = rot_matrix.ptr<double>(2);
	col[0] = 0;
	col[1] = 0;
	col[2] = 1;

	return rot_matrix;
}


void get_rotated_vector(std::vector<double> vec, std::vector<double>& vec_rot) {

	double temp = vec[0];
	temp = temp/sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

	double theta = acos(temp);
	std::cout<<" theta-x : "<<theta<<" ";

	double sinx = sin(theta);
	double cosx = cos(theta);
/*
	//Rotation about the X-axis
	vec_rot[0] = vec[0];
	vec_rot[1] = vec[1]*cosx - vec[2]*sinx;
	vec_rot[2] = vec[1]*sinx + vec[2]*cosx;*/

	vec_rot = vec;
}

void get_reverse_vector(std::vector<double> vec, std::vector<double>& vec_rot) {
	double temp = vec[0];
	temp = temp/sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

	double theta = acos(temp);
	std::cout<<" theta-z : "<<theta<<" ";

	double sinx = sin(theta);
	double cosx = cos(theta);

	//Reverse - rotation about the X-axis
	vec_rot[0] = vec[0];
	vec_rot[1] = vec[1]*cosx + vec[2]*sinx;
	vec_rot[2] = -vec[1]*sinx + vec[2]*cosx;
}

double get_conversion_factor (dlib::full_object_detection shape, Faze face, double magnitude_normal, int mode) {
	cv::Point p1, p2;
    //mode : 1 for left eye, 2 for right eye
	if(mode == 1) {
		p1 = cv::Point(shape.part(42).x(), shape.part(42).y());
		p2 = cv::Point(shape.part(45).x(), shape.part(45).y());
	}
	else if(mode == 2) {
		p1 = cv::Point(shape.part(36).x(), shape.part(36).y());
		p2 = cv::Point(shape.part(39).x(), shape.part(39).y());
	}

	double dx = p1.x - p2.x, dy = p1.y - p2.y;
	double temp1, temp2, beta;
	std::vector<double> normal = face.getNormal();
	double n1 = normal[0], n2 = normal[1], n3 = normal[2];
	double beta_old = sqrt(dx*dx + dy*dy)/magnitude_normal;

	temp1 = dx*dx*(1.0 - n2*n2);
	temp2 = dy*dy*(1.0 - n1*n1);

	beta = sqrt(temp1 + temp2)/((double)(magnitude_normal*fabs(n3)));
	beta = 1.0/((double) beta);

	std::cout<<"Beta : "<<beta<<std::endl;
	return beta;
}

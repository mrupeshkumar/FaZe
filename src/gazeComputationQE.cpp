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

#include "util.h"
#include "faceDetection.h"

double get_conversion_factor (dlib::full_object_detection shape, FacePose* face_pose, double magnitude_normal, int mode) {
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
	double n1 = face_pose->normal[0], n2 = face_pose->normal[1], n3 = face_pose->normal[2];
	double beta_old = sqrt(dx*dx + dy*dy)/magnitude_normal;

	temp1 = dx*dx*(1.0 - n2*n2);
	temp2 = dy*dy*(1.0 - n1*n1);

	beta = sqrt(temp1 + temp2)/((double)(magnitude_normal*fabs(n3)));
	beta = 1.0/((double) beta);

	std::cout<<"Beta : "<<beta<<std::endl;
	return beta;
}

void compute_vec_LR (cv::Point p1, cv::Point p2, FacePose* face_pose, std::vector<double>& LR) {
	double scale = 20.784/30.0;

	LR[0] = p1.x - p2.x;
	LR[1] = p1.y - p2.y;
	LR[0] = LR[0]*scale;
	LR[1] = LR[1]*scale;
	LR[2] = -(LR[0]*face_pose->normal[0] + LR[1]*face_pose->normal[1])/face_pose->normal[3];
}

void get_quadratic_solution (std::vector<double> coeff, double& solution, int mode) {
	solution = (-coeff[1] + mode*sqrt(coeff[1]*coeff[1] - 4*coeff[0]*coeff[2]))/(2*coeff[0]);
	std::cout<<"soln : "<<solution<<std::endl;
}

void get_quadratic_equation (std::vector<double> coeff, std::vector<double>& quad_eqn) {
	quad_eqn[0] = coeff[0]*coeff[0];
	quad_eqn[1] = 2*coeff[0]*coeff[1];
	quad_eqn[2] = coeff[1]*coeff[1];
}

void solve(std::vector<double> coeff_1, double const_1, std::vector<double> coeff_2, double const_2, double mag, std::vector<double>& vec, int mode) {
	double det = coeff_1[0]*coeff_2[1] - coeff_1[1]*coeff_2[0];

	std::vector<double> linear_eqn_1(2), linear_eqn_2(2);
	linear_eqn_1[0] = (coeff_1[1]*coeff_2[2] - coeff_1[2]*coeff_2[1])/det;
	linear_eqn_1[1] = (const_1*coeff_2[1] - coeff_1[1]*const_2)/det;
	linear_eqn_2[0] = (coeff_1[2]*coeff_2[0] - coeff_1[0]*coeff_2[2])/det;
	linear_eqn_2[1] = (coeff_1[0]*const_2 - coeff_2[0]*const_1)/det;

	std::vector<double> quad_eqn_1(3), quad_eqn_2(3), quad_eqn_final(3);
	get_quadratic_equation(linear_eqn_1, quad_eqn_1);
	get_quadratic_equation(linear_eqn_2, quad_eqn_2);

	quad_eqn_final[0] = quad_eqn_1[0] + quad_eqn_2[0] + 1;
	quad_eqn_final[1] = quad_eqn_1[1] + quad_eqn_2[1];
	quad_eqn_final[2] = quad_eqn_1[2] + quad_eqn_2[2] - mag*mag;

	//std::cout<<"const_1 : "<<const_1<<" const_2 : "<<const_2<<std::endl;
	std::vector<double> coeff = quad_eqn_final;
	std::cout<<"Discriminant : "<<coeff[1]*coeff[1] - 4*coeff[0]*coeff[2]<<std::endl;
	log_vec("quad_eqn_final", quad_eqn_final);
	get_quadratic_solution (quad_eqn_final, vec[2], mode);
	vec[0] = linear_eqn_1[0]*vec[2] + linear_eqn_1[1];
	vec[1] = linear_eqn_2[0]*vec[2] + linear_eqn_2[1];
}

void get_section(cv::Point p1, cv::Point p2, cv::Point pupil, double& Y1, double& Y2, double& h) {
	std::vector<double> line(3);

	line[0] = p2.y - p1.y;
	line[1] = -(p2.x - p1.x);
	line[2] =  p1.y*(p2.x - p1.x) - p1.x*(p2.y - p1.y);

	cv::Point pupil_proj;
	pupil_proj.x = -(line[0]*pupil.x + line[1]*pupil.y + line[2])*line[0]/(line[0]*line[0] + line[1]*line[1]) + pupil.x;
	pupil_proj.y = -(line[0]*pupil.x + line[1]*pupil.y + line[2])*line[1]/(line[0]*line[0] + line[1]*line[1]) + pupil.y;

	Y1 = get_distance (p1, pupil_proj);
	Y2 = get_distance (p2, pupil_proj);
	h = get_distance (pupil, pupil_proj);
}

//List : Y1, Y2 can be interchanged. Magnitudes of the vectors in real world may be wrong.
//		 mag_LR square changed to just mag_LR.

void compute_vec_CP(cv::Point p1, cv::Point p2, cv::Point pupil, cv::Rect rect, FacePose* face_pose, 
	std::vector<double> vec_CR_u, double mag_CR, std::vector<double> vec_LR_u, double mag_LR, 
	std::vector<double> vec_UD_u, double mag_CP, std::vector<double>& vec_CP, double S2R, int mode) {
	double Y1, Y2, H;
	get_section(p1, p2, cv::Point(pupil.x + rect.x, pupil.y + rect.y), Y1, Y2, H);

	double const_1, const_2;
	const_1 = (S2R*H);///std::cos(face_pose->pitch);
	if(mode == 1) {
		const_2 = mag_CR*(scalar_product(vec_CR_u, vec_LR_u)) + ((mag_LR*Y1)/((double) (Y1 + Y2)));
	}
	else if(mode == 2) {
		const_2 = mag_CR*(scalar_product(vec_CR_u, vec_LR_u)) + ((mag_LR*Y2)/((double) (Y1 + Y2)));
	}

	//std::cout<<"Y1 : "<<Y1<<" Y2 : "<<Y2<<" H : "<<H<<std::endl;
	std::cout<<"CP - constants : "<<const_1<<" "<<const_2<<std::endl;

	solve(vec_UD_u, const_1, vec_LR_u, const_2, mag_CP, vec_CP, 1);
}

bool vec_isnan(std::vector<double>& vec) {
	int f = 0;
	for(int i=0; i<vec.size(); i++) {
		if(isnan(vec[i])) {
			f=1;
			break;
		}
	}

	if(f) {
		for(int i=0; i<vec.size(); i++) {
			vec[i] = 0.0;
		}
	}

	return (1-f);
}

void compute_eye_gaze (FacePose* face_pose, dlib::full_object_detection shape, cv::Rect rect, cv::Point pupil, double mag_CP, double mag_LR, double mag_CR, double mag_CM, double theta, int mode, std::vector<double>& vec_CP) {

	std::vector<double> vec_LR_u(3), vec_RP(3), vec_CR_u(3), vec_CM_u(3), vec_UD_u(3);
	std::vector<double> vec_CP_l(3), vec_CP_r(3);
	double S2R = get_conversion_factor(shape, face_pose, mag_CM, mode);

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

	vec_CP[0] = 1.0;
	vec_CP[1] = 1.0;
	vec_CP[2] = 1.0;

	compute_vec_LR(p1, p2, face_pose, vec_LR_u);
	make_unit_vector(vec_LR_u, vec_LR_u);

	//log_vec("LR", vec_LR_u);

	vec_CM_u[0] = face_pose->normal[0];
	vec_CM_u[1] = face_pose->normal[1];
	vec_CM_u[2] = face_pose->normal[2];

	cross_product(vec_CM_u, vec_LR_u, vec_UD_u);
	make_unit_vector(vec_UD_u, vec_UD_u);

	//log_vec("UD", vec_UD_u);

	double const_1 = std::cos(theta/2.0);
	double const_2 = 0.0;

	solve(vec_UD_u, const_1, vec_CM_u, const_2, 1.0, vec_CR_u, -1);
	make_unit_vector(vec_CR_u, vec_CR_u);

	//log_vec("CR", vec_CR_u);

	compute_vec_CP(p1, p2, pupil, rect, face_pose, vec_CR_u, mag_CR, vec_LR_u, mag_LR,
		vec_UD_u, mag_CP, vec_CP_l, S2R, 2);

	compute_vec_CP(p1, p2, pupil, rect, face_pose, vec_CR_u, mag_CR, vec_LR_u, mag_LR,
		vec_UD_u, mag_CP, vec_CP_r, S2R, 1);

	double f1 = vec_isnan(vec_CP_l);
	double f2 = vec_isnan(vec_CP_r);

	if(f1 || f2) {
			vec_CP[0] = (vec_CP_l[0] + vec_CP_r[0]);
			vec_CP[1] = (vec_CP_l[1] + vec_CP_r[1]);
			vec_CP[2] = (vec_CP_l[2] + vec_CP_r[2]);		
	}
	else {
		vec_CP[0] = (vec_CP_l[0] + vec_CP_r[0])/2.0;
		vec_CP[1] = (vec_CP_l[1] + vec_CP_r[1])/2.0;
		vec_CP[2] = (vec_CP_l[2] + vec_CP_r[2])/2.0;
	}

	log_vec("CP_l", vec_CP_l);
	log_vec("CP_r", vec_CP_r);
	log_vec("CP", vec_CP);
}
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
#include "faceModel.h"

std::vector<double> computeGazeVA(Face face, double alpha, double magNor) {
	std::vector<double> vec_ep_pos_l(3), vec_ep_pos_r(3);
	std::vector<double> vec_cp_pos_l(3), vec_cp_pos_r(3), vec_cp_pos(3);

	Cf_left = get_conversion_factor(face.shape, face.normal, alpha, 1);
	Cf_right = get_conversion_factor(face.shape, face.normal, alpha, 2);

	cv::Point pt_e_pos_l = get_mid_point(cv::Point(face.shape.part(42).x(), face.shape.part(42).y()),
		cv::Point(face.shape.part(45).x(), face.shape.part(45).y()));
	cv::Point pt_e_pos_r = get_mid_point(cv::Point(face.shape.part(36).x(), face.shape.part(36).y()),
		cv::Point(face.shape.part(39).x(), face.shape.part(39).y()));

	vec_ep_pos_l[0] = face.descriptors[0].x - pt_e_pos_l.x;
	vec_ep_pos_l[1] = face.descriptors[0].y - pt_e_pos_l.y;
	vec_ep_pos_l[2] = 0.0;

	vec_ep_pos_r[0] = face.descriptors[1].x - pt_e_pos_r.x;
	vec_ep_pos_r[1] = face.descriptors[1].y - pt_e_pos_r.y;
	vec_ep_pos_r[2] = 0.0;

	vec_cp_pos_l[0] = (magNor*Cf_left*face.normal[0]) + vec_ep_pos_l[0];
	vec_cp_pos_l[1] = (magNor*Cf_left*face.normal[1]) + vec_ep_pos_l[1];
	vec_cp_pos_l[2] = (magNor*Cf_left*face.normal[2]) + vec_ep_pos_l[2];

	vec_cp_pos_r[0] = (magNor*Cf_right*face.normal[0]) + vec_ep_pos_r[0];
	vec_cp_pos_r[1] = (magNor*Cf_right*face.normal[1]) + vec_ep_pos_r[1];
	vec_cp_pos_r[2] = (magNor*Cf_right*face.normal[2]) + vec_ep_pos_r[2];

	vec_cp_pos[0] = (vec_cp_pos_r[0] + vec_cp_pos_l[0])/2.0;
	vec_cp_pos[1] = (vec_cp_pos_r[1] + vec_cp_pos_l[1])/2.0;
	vec_cp_pos[2] = (vec_cp_pos_r[2] + vec_cp_pos_l[2])/2.0;

	return vec_cp_pos;
}
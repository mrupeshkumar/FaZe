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

using namespace std;
using namespace dlib;

int main(int argc, char** argv) {
	Faze faze = Faze();
	cv::VideoCapture cap(0);
	cv::Mat frame_clr;
	image_window win;

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("./res/shape_predictor_68_face_landmarks.dat") >> pose_model;

	while(!win.is_closed()) {
		cap >> frame_clr;
		cv::flip(frame_clr, frame_clr, 1);

		cv_image<unsigned char> cimg_gray(frame);
		cv_image<bgr_pixel> cimg_clr(frame_clr);

		vector<rectangle> faces = detector(cimg_gray);

		full_object_detection shape = pose_model(cimg_gray, faces[i]);

		faze.assign(shape, frame_clr);
	}
	return 0;
}
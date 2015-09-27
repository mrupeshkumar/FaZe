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
	cv::Mat frame_clr, frame;
	image_window win;

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("./res/shape_predictor_68_face_landmarks.dat") >> pose_model;

	while(!win.is_closed()) {
		cap >> frame_clr;
		cv::flip(frame_clr, frame_clr, 1);
		cv::cvtColor(frame_clr, frame, CV_BGR2GRAY);

		cv_image<unsigned char> cimg_gray(frame);
		cv_image<bgr_pixel> cimg_clr(frame_clr);

		std::vector<rectangle> faces = detector(cimg_gray);

		if((int)faces.size() != 0) {
			full_object_detection shape = pose_model(cimg_gray, faces[0]);
			faze.assign(shape, frame_clr);
			std::vector<double> normal = faze.getNormal();
			cout<<normal[0]<<" "<<normal[1]<<" "<<normal[2]<<endl;
		}
		else {
			cout<<"No faces"<<endl;
		}
	}
	return 0;
}
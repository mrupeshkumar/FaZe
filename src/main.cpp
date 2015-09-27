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

// Force resize to 300, 300 as dlib faces are nearly square

using namespace std;
using namespace dlib;

void log(std::vector<cv::Point> vec) {
	for(int i=0; i<(int)vec.size(); ++i)
		cout<<"("<<vec[i].x<<", "<<vec[i].y<<") ";
	cout<<endl;
}

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

		std::vector<rectangle> faces = detector(cimg_gray);

		if((int)faces.size() != 0) {
			rectangle face = faces[0];
			double rx = 300.0/(face.right() - face.left());
			double ry = 300.0/(face.bottom() - face.top());
			rectangle faceResized = rectangle(face.left()*rx, face.top()*ry, face.right()*rx, face.bottom()*ry);
			cv::Rect cvFaceRectResized = cv::Rect(faceResized.left(), faceResized.top(), 
				faceResized.right() - faceResized.left(), faceResized.bottom() - faceResized.top());
			cv::Size sizeNew = cv::Size(frame_clr.cols*rx, frame_clr.rows*ry);
			cv::resize(frame_clr, frame_clr, sizeNew);

			cv_image<bgr_pixel> cimg_clr(frame_clr);

			cv::Mat frameResized;
			cv::resize(frame, frameResized, sizeNew);			
			cv_image<unsigned char> cimg_gray_resized(frameResized);

			cout<<faceResized.right() - faceResized.left()<<", "<<faceResized.bottom() - faceResized.top()<<endl;
			full_object_detection shape = pose_model(cimg_gray_resized, faceResized);
			
			faze.assign(shape, frame_clr);
			faze.setOrigin(faze.ORIGIN_FACE_CENTRE);

			std::vector<double> normal = faze.getNormal();
			cout<<normal[0]<<" "<<normal[1]<<" "<<normal[2]<<endl;
			
			std::vector<cv::Point> mouthCtrsOut = faze.getDescriptors(faze.INDEX_MOUTH_OUTER, faze.DESCRIPTOR_LOCAL);
			std::vector<cv::Point> mouthCtrsIn = faze.getDescriptors(faze.INDEX_MOUTH_INNER, faze.DESCRIPTOR_LOCAL);

			log(mouthCtrsOut);
			log(mouthCtrsIn);

			win.clear_overlay();
			win.set_image(cimg_gray_resized);
			win.add_overlay(render_face_detections(shape));
		}
		else {
			cout<<"No faces"<<endl;
		}
	}
	return 0;
}
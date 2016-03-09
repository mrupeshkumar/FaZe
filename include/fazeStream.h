#ifndef FAZE_STREAM_H
#define FAZE_STREAM_H
#endif

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/video.hpp"

class Stream {

private:
	int _degree;
	int _smooth;
	FixedBin<Faze> _bin;
	Faze _faze;
	cv::KalmanFilter _kalmanFilter;
	cv::Mat_<float> _measurements, _measurementsOld;

public:
	const static int SMOOTH_AVG = 0;
	const static int SMOOTH_KALMAN = 1;

	Stream(int degree, int smooth = SMOOTH_AVG);
	int degree();
	int smooth();
	int filled();
	void push(Faze faze);
	Faze current();
};

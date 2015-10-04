#ifndef FAZE_STREAM_H
#define FAZE_STREAM_H
#endif

template <class T>
class FixedBin {

private:
	std::vector< T > _bin;
	int _size;
	int _filled;

public:
	FixedBin();
	void assign(int size);
	void push(T t);
	int size();
	int filled();
	T get(int pos);
	std::vector< T > clone();
};

class Stream {

private:
	int _degree;
	int _smooth;
	FixedBin<Faze> _fazes;
	cv::KalmanFilter _kalmanFilter;
	cv::Mat _measurements;

public:
	static int SMOOTH_AVG = 0;
	static int SMOOTH_KALMAN = 1;

	Stream(int degree, int smooth = SMOOTH_AVG);
	int degree();
	int smooth();
	int filled();
	void push(Faze faze);
};
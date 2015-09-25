#ifndef UTIL_H
#define UTIL_H

void log_vec(std::string str, std::vector<double> vec);
void read_vector_from_file(char* file_name, std::vector<std::vector<double> >& arr);
void blow_up_rect(cv::Rect& rect, double f);
void show_images(int e ,int l, int h, std::vector<cv::Mat> imgs);
double get_distance(cv::Point p1, cv::Point p2);
cv::Point get_mid_point(cv::Point p1, cv::Point p2);
double get_vector_magnitude(double vec[], int size);
void compute_vector_sum(std::vector<double> vec1, std::vector<double> vec2, std::vector<double>& vec_sum);
double get_angle_between(cv::Point pt1, cv::Point pt2);
void make_unit_vector(std::vector<double> vec, std::vector<double>& unit_vector);
double scalar_product(std::vector<double> vec1, std::vector<double> vec2);
cv::Mat get_rotation_matrix_z(double theta);
void get_rotated_vector(std::vector<double> vec, std::vector<double>& vec_rot);
void get_reverse_vector(std::vector<double> vec, std::vector<double>& vec_rot);
void cross_product(std::vector<double> vec1, std::vector<double> vec2, std::vector<double>& product);

#endif

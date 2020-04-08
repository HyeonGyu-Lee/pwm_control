#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <iostream>
#include <geometry_msgs/Twist.h>
#include <obstacle_detector/Obstacles.h>
#include <geometry_msgs/Point.h>

using namespace std;
using namespace cv;

class LaneDetector {
private:
        geometry_msgs::Twist msg_;
	ros::NodeHandle nh_;
	ros::Publisher pub_;
	ros::Subscriber image_sub_;
	ros::Subscriber obstacle_sub_;

	cv_bridge::CvImagePtr cv_ptr_;
	VideoCapture cap_;

	int size_of_segments;
	float dist_ob;

	geometry_msgs::Point first_point;
	geometry_msgs::Point last_point;
	geometry_msgs::Point middle_point;

	int select_;
	Size set_;
	Mat frame_, resized_frame_, warped_frame_, warped_back_frame_;
	Mat binary_frame_, sliding_frame_, result_frame_;

	vector<Point2f> corners_;
	vector<Point2f> warpCorners_;

	int sobel_min_th_;
	int sobel_max_th_;
	int hls_min_th_;
	int hls_max_th_;

	int last_Llane_base_;
	int last_Rlane_base_;

	vector<int> left_lane_inds_;
	vector<int> right_lane_inds_;
	vector<int>  left_x_;
	vector<int>  left_y_;
	vector<int>  right_x_;
	vector<int>  right_y_;

	Mat  left_coef_;
	Mat  right_coef_;
	float left_curve_radius_;
	float right_curve_radius_;
	float center_position_;

	/********** PID control ***********/
	int steer_, accel_, clicker_, throttle_, histo_;
	int prev_lane_, prev_pid_;
	float Kp_, Ki_, Kd_, dt_, result_;
	float Kp_term_, Ki_term_, Kd_term_;
	float err_, prev_err_, I_err_, D_err_;
public:
	LaneDetector(void) {
		initSetup();
	}

	~LaneDetector(void) {
		left_lane_inds_.clear();
		right_lane_inds_.clear();
	}

	void initSetup(void) {
		cout << "LaneDetector initialization setup..." << endl;
		
		nh_.getParam("/Kp", Kp_);
		nh_.getParam("/Ki", Ki_);
		nh_.getParam("/Kd", Kd_);
		nh_.getParam("/dt", dt_);
		nh_.getParam("/clicker", clicker_);
		nh_.getParam("/throttle", throttle_);
		nh_.getParam("/histo", histo_);

		/********** PID control ***********/
		center_position_ = 640;
		//Kp_ = 1.0f;
		//Ki_ = 0.00001f;
		//Kd_ = 0.005f;
		//dt_ = 0.1f;
		prev_err_ = 0;
		steer_ = clicker_;
		accel_ = throttle_;

		/********** pub config **********/
		cout << "[pub config]" << endl;
	        msg_.linear.x = steer_;
		msg_.angular.z = accel_;
	        pub_ = nh_.advertise<geometry_msgs::Twist>("twist_msg",10);

		/********** image config **********/
		cout << "[image config]" << endl;
		select_ = 0;
		image_sub_ = nh_.subscribe("/usb_cam/image_raw", 10, &LaneDetector::ImageCallback, this);
		obstacle_sub_ = nh_.subscribe("/raw_obstacles", 100, &LaneDetector::Obstacle_cb, this);

		/********** set window **********/
		cout << "[set windows]" << endl;
		namedWindow("ORIGINAL");
		moveWindow("ORIGINAL", 0, 0);
		namedWindow("WARPED");
		moveWindow("WARPED", 640, 0);
		namedWindow("RESULT");
		moveWindow("RESULT", 1280, 0);

		/********** set thresholds **********/
		cout << "[set thresholds]" << endl;
		sobel_min_th_ = 230;
		sobel_max_th_ = 255;
		hls_min_th_ = 230;
		hls_max_th_ = 255;

		last_Llane_base_ = 0;
		last_Rlane_base_ = 0;

		left_coef_ = Mat::zeros(3, 1, CV_32F);
		right_coef_ = Mat::zeros(3, 1, CV_32F);

		cout << "[INITIALIZATION DONE]" << endl;
	}

	void ImageCallback(const sensor_msgs::ImageConstPtr &image_msg) {
		try {
			cv_ptr_ = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
		}
		catch (cv_bridge::Exception& e) {
			ROS_ERROR("cv_bridge exception: %s", e.what());
		        return;
		}
		frame_ = cv_ptr_->image;
	}

	Mat warped_img(Mat _frame, int _width, int _height) {
		Mat result, trans;
		float wide_extra_upside, wide_extra_downside;
		corners_.resize(4); // lane coord
		warpCorners_.resize(4);
		//( 0 ,  0 )----------------------------------(MAX,  0 )
		//         
		//                    coordinate
		//         
		//( 0 , MAX)----------------------------------(MAX, MAX)
		corners_[0] = Point2f(300, 450); // left up (width, height)
		corners_[1] = Point2f(980, 450); // right up (width, height)
		corners_[2] = Point2f(50, 680); // left down (width, height)
		corners_[3] = Point2f(1230, 680); // right down (width, height)
		wide_extra_upside = 0;
		wide_extra_downside = 100;
		warpCorners_[0] = Point2f(wide_extra_upside, 0.0);  // left up (width, height)
		warpCorners_[1] = Point2f(_width - wide_extra_upside, 0.0); // right up (width, height)
		warpCorners_[2] = Point2f(wide_extra_downside, (float)_height); // left down (width, height)
		warpCorners_[3] = Point2f(_width - wide_extra_downside, (float)_height); // right down (width, height)
		trans = getPerspectiveTransform(corners_, warpCorners_);
		warpPerspective(_frame, result, trans, Size(_width, _height));

		return result;
	}

	Mat warped_back_img(Mat _frame, int _width, int _height) {
		Mat result, trans;
		trans = getPerspectiveTransform(warpCorners_, corners_);
		warpPerspective(_frame, result, trans, Size(_width, _height));

		return result;
	}

	Mat abs_sobel_thresh(Mat _frame) {
		Mat gray_frame, sobel_frame, abs_sobel_frame;
		double min, max, val;
		cvtColor(_frame, gray_frame, COLOR_BGR2GRAY);
		Sobel(gray_frame, sobel_frame, CV_64F, 1, 0);
		//abs_sobel_frame = abs(sobel_frame);
		sobel_frame.convertTo(abs_sobel_frame, CV_8U);

		minMaxLoc(abs_sobel_frame, &min, &max);

		Mat scaled_sobel_binary_frame = Mat::zeros(sobel_frame.rows, sobel_frame.cols, CV_8UC1);

		for (int j = 0; j < sobel_frame.rows; j++) {
			for (int i = 0; i < sobel_frame.cols; i++) {
				val = (abs_sobel_frame.at<uchar>(j, i)) * 255 / max;
				if ((val >= sobel_min_th_) && (val <= sobel_max_th_))
					scaled_sobel_binary_frame.at<uchar>(j, i) = 255;
			}
		}
		return scaled_sobel_binary_frame;
	}

	Mat hls_lthresh(Mat _frame) {
		Mat hls_frame;
		vector<Mat> hls_images(3);
		double min, max;

		cvtColor(_frame, hls_frame, COLOR_BGR2HLS);
		split(hls_frame, hls_images);

		minMaxLoc(hls_images[1], &min, &max);

		for (int j = 0; j < hls_images[1].rows; j++) {
			for (int i = 0; i < hls_images[1].cols; i++) {
				if (((hls_images[1].at<uchar>(j, i) * 255 / max) > hls_min_th_) && ((hls_images[1].at<uchar>(j, i) * 255 / max) <= hls_max_th_))
					hls_images[1].at<uchar>(j, i) = 255;
			}
		}

		return hls_images[1];
	}

	Mat pipeline_img(Mat _frame) {
		Mat abs_sobel_frame = abs_sobel_thresh(_frame);
		Mat l_frame = hls_lthresh(_frame);
		Mat combined_frame = Mat::zeros(_frame.rows, _frame.cols, CV_8UC1);
		for (int j = 0; j < combined_frame.rows; j++) {
			for (int i = 0; i < combined_frame.cols; i++) {
				if ((abs_sobel_frame.at<uchar>(j, i) > histo_) || (l_frame.at<uchar>(j, i) > histo_)) combined_frame.at<uchar>(j, i) = 255;
			}
		}
		return combined_frame;
	}

	int arrMaxIdx(int hist[], int start, int end, int Max) {
		int max_index = -1;
		int max_val = 0;

		if (end > Max)
			end = Max;

		for (int i = start; i < end; i++) {
			if (max_val < hist[i]) {
				max_val = hist[i];
				max_index = i;
			}
		}
		if (max_index == -1)
			cout << "ERROR : hist range" << endl;
		return max_index;
	}

	double gaussian(double x, double mu, double sig) {
		return exp((-1) * pow(x - mu, 2.0) / (2 * pow(sig, 2.0)));
	}

	Mat polyfit(vector<int> x_val, vector<int> y_val) {
		Mat coef(3, 1, CV_32F);
		int i, j, k, n, N;
		N = (int)x_val.size();
		n = 2;
		float *x, *y;
		x = new float[N];
		y = new float[N];
		for (int q = 0; q < N; q++) {
			x[q] = (float)(x_val[q]);
			y[q] = (float)(y_val[q]);
		}
		float *X;
		X = new float[2 * n + 1];                        //Array that will store the values of sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
		for (i = 0;i < (2 * n + 1);i++)
		{
			X[i] = 0;
			for (j = 0;j < N;j++)
				X[i] = X[i] + pow(x[j], i);        //consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
		}
		float **B, *a;
		B = new float*[n + 1];
		for (int i = 0; i < (n + 1); i++)
			B[i] = new float[n + 2];
		a = new float[n + 1];            //B is the Normal matrix(augmented) that will store the equations, 'a' is for value of the final coefficients
		for (i = 0;i <= n;i++)
			for (j = 0;j <= n;j++)
				B[i][j] = X[i + j];            //Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
		float *Y;
		Y = new float[n + 1];                    //Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
		for (i = 0;i < (n + 1);i++)
		{
			Y[i] = 0;
			for (j = 0;j < N;j++)
				Y[i] = Y[i] + pow(x[j], i)*y[j];        //consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
		}
		for (i = 0;i <= n;i++)
			B[i][n + 1] = Y[i];                //load the values of Y as the last column of B(Normal Matrix but augmented)
		n = n + 1;                //n is made n+1 because the Gaussian Elimination part below was for n equations, but here n is the degree of polynomial and for n degree we get n+1 equations

		for (i = 0;i < n;i++)                    //From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
			for (k = i + 1;k < n;k++)
				if (B[i][i] < B[k][i])
					for (j = 0;j <= n;j++)
					{
						float temp = B[i][j];
						B[i][j] = B[k][j];
						B[k][j] = temp;
					}

		for (i = 0;i < (n - 1);i++)            //loop to perform the gauss elimination
			for (k = i + 1;k < n;k++)
			{
				float t = B[k][i] / B[i][i];
				for (j = 0;j <= n;j++)
					B[k][j] = B[k][j] - t * B[i][j];    //make the elements below the pivot elements equal to zero or elimnate the variables
			}
		for (i = n - 1;i >= 0;i--)                //back-substitution
		{                        //x is an array whose values correspond to the values of x,y,z..
			a[i] = B[i][n];                //make the variable to be calculated equal to the rhs of the last equation
			for (j = 0;j < n;j++)
				if (j != i)            //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
					a[i] = a[i] - B[i][j] * a[j];
			a[i] = a[i] / B[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
			coef.at<float>(i, 0) = (float)a[i];
		}
		
		delete[] x;
		delete[] y;
		delete[] X;
		delete[] Y;
		delete[] B;
		delete[] a;

		return coef;
	}

	Mat detect_lines_sliding_window(Mat _frame, int _width, int _height) {
		Mat frame, result;
		_frame.copyTo(frame);

		Mat nonZero;
		findNonZero(frame, nonZero);

		vector<int> good_left_inds;
		vector<int> good_right_inds;
		int *hist = new int[_width];
		double *weight_distrib = new double[_width];
		for (int i = 0; i < _width; i++) {
			hist[i] = 0;
		}

		int hist_Max = 0;
		for (int j = 0; j < _height; j++) {
			for (int i = 0; i < _width; i++) {
				if (frame.at <uchar>(j, i) == 255) {
					hist[i] += 1;
				}
			}
		}

		hist_Max = arrMaxIdx(hist, 0, _width, _width);

		if (last_Llane_base_ != 0 || last_Rlane_base_ != 0) {

			int distrib_width = 100;
			double sigma = distrib_width / 12.8;

			int leftx_start = last_Llane_base_ - distrib_width / 2;
			int leftx_end = last_Llane_base_ + distrib_width / 2;

			int rightx_start = last_Rlane_base_ - distrib_width / 2;
			int rightx_end = last_Rlane_base_ + distrib_width / 2;

			for (int i = 0; i < _frame.cols; i++) {
				if ((i >= leftx_start) && (i <= leftx_end)) {
					weight_distrib[i] = gaussian(i, last_Llane_base_, sigma);
					hist[i] *= (int)weight_distrib[i];
				}
				else if ((i >= rightx_start) && (i <= rightx_end)) {
					weight_distrib[i] = gaussian(i, last_Rlane_base_, sigma);
					hist[i] *= (int)weight_distrib[i];
				}
			}
		}
		cvtColor(frame, result, COLOR_GRAY2BGR);

		int mid_point = _width / 2; // 320
		int quarter_point = mid_point / 2; // 160
		int n_windows = 9;
		int margin = 100;
		int min_pix = 200;

		int window_width = margin;
		int window_height = _height / n_windows;
		
		int offset = 0;
		int range = 120;
		int Lstart = quarter_point - offset; // 160 - 0
		int Rstart = mid_point + quarter_point + offset; // 480 - 0
		// mid_point = 320, Lstart +- range = 40 ~ 280
		int Llane_base = arrMaxIdx(hist, Lstart - range, Lstart + range, _width);
		int Rlane_base = arrMaxIdx(hist, Rstart - range, Rstart + range, _width);
		//int Llane_base = arrMaxIdx(hist, 10, mid_point, _width);
		//int Rlane_base = arrMaxIdx(hist, mid_point, _width-10, _width);

		int Llane_current = Llane_base;
		int Rlane_current = Rlane_base;

		last_Llane_base_ = Llane_base;
		last_Rlane_base_ = Rlane_base;

		unsigned int index;

		for (int window = 0; window < n_windows;window++) {
			int Ly_pos = _height - (window + 1)*window_height; // win_y_low , win_y_high = win_y_low - window_height
			int Ry_pos = _height - (window + 1)*window_height;

			int Lx_pos = Llane_current - margin; // win_xleft_low, win_xleft_high = win_xleft_low + margin*2
			int Rx_pos = Rlane_current - margin; // win_xrignt_low, win_xright_high = win_xright_low + margin*2

			rectangle(result, \
				Rect(Lx_pos, Ly_pos, window_width * 2, window_height), \
				Scalar(255, 50, 100), 2);
			rectangle(result, \
				Rect(Rx_pos, Ry_pos, window_width * 2, window_height), \
				Scalar(100, 50, 255), 2);

			uchar *data_output = result.data;
			int nZ_y, nZ_x;

			good_left_inds.clear();
			good_right_inds.clear();

			for (unsigned int index = 0; index < nonZero.total(); index++) {
				nZ_y = nonZero.at<Point>(index).y;
				nZ_x = nonZero.at<Point>(index).x;
				if ((nZ_y >= Ly_pos) && \
					(nZ_y < (_height - window_height * window)) && \
					(nZ_x >= Lx_pos) && \
					(nZ_x < (Lx_pos + margin * 2))) {
					result.at<Vec3b>(nonZero.at<Point>(index))[0] = 255;
					result.at<Vec3b>(nonZero.at<Point>(index))[1] = 0;
					result.at<Vec3b>(nonZero.at<Point>(index))[2] = 0;
					good_left_inds.push_back(index);
				}
				if ((nZ_y >= (Ry_pos)) && \
					(nZ_y < (_height - window_height * window)) && \
					(nZ_x >= Rx_pos) && \
					(nZ_x < (Rx_pos + margin * 2))) {
					result.at<Vec3b>(nonZero.at<Point>(index))[0] = 0;
					result.at<Vec3b>(nonZero.at<Point>(index))[1] = 0;
					result.at<Vec3b>(nonZero.at<Point>(index))[2] = 255;
					good_right_inds.push_back(index);
				}
			}
			int Lsum, Rsum;
			unsigned int _size;
			if (good_left_inds.size() > (size_t)min_pix) {
				Lsum = 0;
				_size = (unsigned int)good_left_inds.size();
				for (index = 0; index < _size; index++) {
					Lsum += nonZero.at<Point>(good_left_inds.at(index)).x;
				}
				Llane_current = Lsum / _size;
			}
			if (good_right_inds.size() > (size_t)min_pix) {
				Rsum = 0;
				_size = (unsigned int)good_right_inds.size();
				for (index = 0; index < _size; index++) {
					Rsum += nonZero.at<Point>(good_right_inds.at(index)).x;
				}
				Rlane_current = Rsum / _size;
			}
			left_lane_inds_.insert(left_lane_inds_.end(), good_left_inds.begin(), good_left_inds.end());
			right_lane_inds_.insert(right_lane_inds_.end(), good_right_inds.begin(), good_right_inds.end());
		}

		vector<int>::iterator iter;
		left_x_.clear();
		left_y_.clear();
		right_x_.clear();
		right_y_.clear();
		for (index = 0,iter = left_lane_inds_.begin(); iter != left_lane_inds_.end(); iter++, index++) {
			if ((int)(nonZero.total()) >= *iter) {
				left_x_.insert(left_x_.end(), nonZero.at<Point>(*iter).x);
				left_y_.insert(left_y_.end(), nonZero.at<Point>(*iter).y);
			}
		}
		for (index = 0,iter = right_lane_inds_.begin(); iter != right_lane_inds_.end(); iter++, index++) {
			if ((int)(nonZero.total()) >= *iter) {
				right_x_.insert(right_x_.end(), nonZero.at<Point>(*iter).x);
				right_y_.insert(right_y_.end(), nonZero.at<Point>(*iter).y);
			}
		}

		if (left_x_.size() != 0) {
			left_coef_ = polyfit(left_y_, left_x_);
		}
		if (right_x_.size() != 0) {
			right_coef_ = polyfit(right_y_, right_x_);
		}
		
		delete[] hist;
		delete[] weight_distrib;

		return result;
	}

	Mat draw_lane(Mat frame, int _width, int _height, Mat left_coef, Mat right_coef, Mat Minv) {
		Mat new_frame;
		frame.copyTo(new_frame);

		vector<Point> left_point;
		vector<Point> right_point;

		vector<Point2f> left_point_f;
		vector<Point2f> right_point_f;

		vector<Point2f> warped_left_point;
		vector<Point2f> warped_right_point;

		vector<Point> left_points;
		vector<Point> right_points;

		if ((!left_coef.empty()) && (!right_coef.empty())) {
			for (int i = 0; i <= _height; i++) {
				Point temp_left_point;
				Point temp_right_point;
				temp_left_point.x = (int)((left_coef.at<float>(2, 0) * pow(i, 2)) + (left_coef.at<float>(1, 0) * i) + left_coef.at<float>(0, 0));
				temp_left_point.y = (int)i;
				temp_right_point.x = (int)((right_coef.at<float>(2, 0) * pow(i, 2)) + (right_coef.at<float>(1, 0) * i) + right_coef.at<float>(0, 0));
				temp_right_point.y = (int)i;

				left_point.push_back(temp_left_point);
				right_point.push_back(temp_right_point);
				left_point_f.push_back(temp_left_point);
				right_point_f.push_back(temp_right_point);
			}
			const Point *left_points_point_ = (const cv::Point*) Mat(left_point).data;
			int left_points_number_ = Mat(left_point).rows;
			const Point *right_points_point_ = (const cv::Point*) Mat(right_point).data;
			int right_points_number_ = Mat(right_point).rows;

			polylines(sliding_frame_, &left_points_point_, &left_points_number_, 1, false, Scalar(255, 200, 200), 15);
			polylines(sliding_frame_, &right_points_point_, &right_points_number_, 1, false, Scalar(200, 200, 255), 15);

			perspectiveTransform(left_point_f, warped_left_point, Minv);
			perspectiveTransform(right_point_f, warped_right_point, Minv);

			for (int i = 0; i <= _height; i++) {
				Point temp_left_point;
				Point temp_right_point;

				temp_left_point.x = (int)warped_left_point[i].x;
				temp_left_point.y = (int)warped_left_point[i].y;
				temp_right_point.x = (int)warped_right_point[i].x;
				temp_right_point.y = (int)warped_right_point[i].y;

				left_points.push_back(temp_left_point);
				right_points.push_back(temp_right_point);
			}

			const Point *left_points_point = (const cv::Point*) Mat(left_points).data;
			int left_points_number = Mat(left_points).rows;
			const Point *right_points_point = (const cv::Point*) Mat(right_points).data;
			int right_points_number = Mat(right_points).rows;

			polylines(result_frame_, &left_points_point, &left_points_number, 1, false, Scalar(255, 100, 100), 10);
			polylines(result_frame_, &right_points_point, &right_points_number, 1, false, Scalar(100, 100, 255), 10);

			left_point.clear();
			right_point.clear();

			return new_frame;
		}
		return frame;
	}

	void clear_release() {
		left_lane_inds_.clear();
		right_lane_inds_.clear();
		left_x_.clear();
		left_y_.clear();
		right_x_.clear();
		right_y_.clear();
	}

	void calc_curv_rad_and_center_dist(int _width, int _height, Mat l_fit, Mat r_fit, vector<int> lx, vector<int> ly, vector<int> rx, vector<int> ry) {
		int car_position = _width / 2;
		int lane_center_position;
		float center_position;
		float left_cr;
		float right_cr;

		float ym_per_pix = 3.048f / 100.f;
		float xm_per_pix = 3.7f / 378.0f;

		Mat left_coef_cr(3, 1, CV_32F);
		Mat right_coef_cr(3, 1, CV_32F);

		if (lx.size() != 0 && rx.size() != 0) {
			for (int i = 0; i < lx.size(); i++) {
				lx[i] = lx[i] * xm_per_pix;
				ly[i] = ly[i] * ym_per_pix;
			}
			for (int i = 0; i < rx.size(); i++) {
				rx[i] = rx[i] * xm_per_pix;
				ry[i] = ry[i] * ym_per_pix;
			}

			left_coef_cr = polyfit(lx, ly);
			right_coef_cr = polyfit(rx, ry);

			left_cr = powf((1 + powf(2 * left_coef_cr.at<float>(2, 0) * 0 * xm_per_pix + left_coef_cr.at<float>(1, 0), 2)), 1.5f) / fabs(2 * left_coef_cr.at<float>(2, 0) + 0.000001f);
			right_cr = powf((1 + powf(2 * right_coef_cr.at<float>(2, 0) * 0 * xm_per_pix + right_coef_cr.at<float>(1, 0), 2)), 1.5f) / fabs(2 * right_coef_cr.at<float>(2, 0) + 0.000001f);

			left_curve_radius_ = left_cr;
			right_curve_radius_ = right_cr;
		}

		if (!l_fit.empty() && !r_fit.empty()) {
			lane_center_position = (l_fit.at<float>(0, 0) + r_fit.at<float>(0, 0)) / 2;
			if ((lane_center_position > 0) && (lane_center_position < (float)_width)) {
				center_position = (car_position - lane_center_position) * ym_per_pix;
				err_ = (float)(lane_center_position - center_position_);
				I_err_ += err_ * dt_;
				D_err_ = (err_ - prev_err_) / dt_;
				prev_err_ = err_;

				result_ = (Kp_ * err_) + (Ki_ * I_err_) + (Kd_ * D_err_); // PID
				line(sliding_frame_, Point(lane_center_position, 0), Point(lane_center_position, _height), Scalar(0, 255, 0), 5);

				center_position_ += (result_);
				steer_ = (int)(((center_position_- 640.0) / 640.0 * 400.0) + clicker_);// 1200~1800

				msg_.angular.z = (int)steer_;
				msg_.linear.x = (int)accel_;
			}
		}
	}

	void run(void) {
		if(select_ != 0)
			cap_ >> frame_;
		if(!frame_.empty()){
			resize(frame_, resized_frame_, Size(1280, 720));
			resized_frame_.copyTo(result_frame_);

			int width = resized_frame_.cols;
			int height = resized_frame_.rows;

		/********** Processing frame **********/
			warped_frame_ = warped_img(resized_frame_, width, height);
			binary_frame_ = pipeline_img(warped_frame_);
			sliding_frame_ = detect_lines_sliding_window(binary_frame_, width, height);
			resized_frame_ = draw_lane(resized_frame_, width, height, left_coef_, right_coef_, getPerspectiveTransform(warpCorners_, corners_));
			calc_curv_rad_and_center_dist(width, height, left_coef_, right_coef_, left_x_, left_y_, right_x_, right_y_);
			clear_release();
		
		/********** ROI LINE **********/
			string Text = "ROI";
			Point2f Text_pos(corners_[0]);
			putText(resized_frame_, Text, Text_pos, FONT_HERSHEY_DUPLEX, 2, Scalar(0, 0, 255), 5, 8);
			line(resized_frame_, corners_[0], corners_[2], Scalar(0, 0, 255), 5);
			line(resized_frame_, corners_[2], corners_[3], Scalar(0, 0, 255), 5);
			line(resized_frame_, corners_[3], corners_[1], Scalar(0, 0, 255), 5);
			line(resized_frame_, corners_[1], corners_[0], Scalar(0, 0, 255), 5);

		/********** PID TAG **********/
			Text = "STEER";
			putText(sliding_frame_, Text, Point2f(300, 300), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
			Text = to_string((int)steer_);
			putText(sliding_frame_, Text, Point2f(400, 300), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
			Text = "ACCEL";
			putText(sliding_frame_, Text, Point2f(300, 340), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
			Text = to_string((int)accel_);
			putText(sliding_frame_, Text, Point2f(400, 340), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);

		/********** resized_frame_Remark **********/
			set_ = Size(1280 / 2, 720 / 2);

		/********** update_image **********/
			resize(resized_frame_, resized_frame_, set_);
			imshow("ORIGINAL", resized_frame_);
			resize(sliding_frame_, sliding_frame_, set_);
			imshow("WARPED", sliding_frame_);
			resize(result_frame_, result_frame_, set_);
			imshow("RESULT", result_frame_);
		
		}
		/********** msg_publish **********/
		
		pub_.publish(msg_);
		waitKey(1);
	}

	void Obstacle_cb(const obstacle_detector::Obstacles& data)
	{
		first_point.x = 0.0;
		first_point.y = 0.0;
		last_point.x = 0.0;
		last_point.y = 0.0;
		size_of_segment = data.segments.size();

		for(int i = 0; i < size_of_segments ; i++){
			first_point.x += data.segments[i].first_point.x;
			first_point.y += data.segments[i].first_point.y;
			last_point.x += data.segments[i].last_point.x;
			last_point.y += data.segments[i].last_point.y;

		}

		if(size_of_segments != 0){
			first_point.x = first_point.x/size_of_segments;
			first_point.y = first_point.y/size_of_segments;
			last_point.x = last_point.x/size_of_segments;
			last_point.y = last_point.y/size_of_segments;
		}

		middle_point.x = -(first_point.x + last_point.x)/2;
		middle_point.y = -(first_point.y + last_point.y)/2;

		dist_ob = sqrt(pow(middle_point.x, 2) + pow(middle_point.y, 2));

		accel_ = 1700 + (int)(dist_ob/0.14)*7;

	}
};

int main(int argc, char **argv) {
        ros::init(argc, argv, "pwm_control_node");
	LaneDetector ld;
	while (ros::ok()){
		ld.run();
		ros::spinOnce();
	}

	return 0;
}



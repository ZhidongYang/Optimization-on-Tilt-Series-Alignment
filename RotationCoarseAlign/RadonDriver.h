#pragma once
/*
	A marker-free alignment method for electron tomography [1998]
	@Introduction: The Common Line Algorithm Implmentation
		an important algorithm in tilt series coarse alignment
	@Author : Zhidong YANG
	@Location : Institute of Computing Technology,
		Chinese Academy of Sciences, Beijing, China
	@Date : Dec. 28th. 2018 - Jan.13th. 2018
*/

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\contrib\contrib.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <io.h>
#include "fftw3.h"
using namespace std;
using namespace cv;

class Radon {
public:
	/*public constructors*/
	Radon(Mat &origin);	// without specified angles range, the default will be 1-180
	Radon(Mat &origin, vector<float> &angles) :origin(origin), angles(angles) {  }	// with specified angles range
	cv::Mat radonExecution();	// interface
private:
	Mat origin;
	vector<float> angles;
	/*methods*/
	float sum_proj(Mat img, int j);
	void dft_radon(Mat img, Mat &mag);
	cv::Mat radonImplementation();
};

Radon::Radon(Mat &origin):origin(origin) {
	for (int i = 0; i < 180; i++) {
		angles.push_back(i + 1);
	}
}

cv::Mat Radon::radonExecution() {
	Mat res = radonImplementation();
	return res;
}

float Radon::sum_proj(Mat img, int j) {
	float summ = 0;
	for (int i = 0; i<img.cols; i++)
	{
		summ += (float)img.ptr<float>(j)[i];
	}
	return summ;
}

void Radon::dft_radon(Mat img, Mat &mag) {
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;//将原图像的大小变为m*n的大小，补充的位置填0，
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));//这里是获取了两个mat，一个用于存放dft变换的实部，一个用于存放虚部，初始的时候，实部就是图像本身，虚部全为0
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;//将几个单通道的mat融合成一个多通道的mat，这里融合的complexImg即有实部，又有虚部
	merge(planes, 2, complexImg);//dft变换，因为complexImg本身就是两个通道的mat，所以dft变换的结果也可以保存在其中
	dft(complexImg, complexImg);    //将complexImg重新拆分成两个mat，一个是实部，一个是虚部
	split(complexImg, planes);

	// compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
	//这一部分是为了计算dft变换后的幅值，以便于显示幅值的计算公式如上
	magnitude(planes[0], planes[1], planes[0]);//将两个mat对应位置相乘
	mag = planes[0];
	mag += Scalar::all(1);
	log(mag, mag);

	//修剪频谱，如果图像的行或者列是奇数的话，那其频谱是不对称的，因此要修剪
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(mag, mag, 0, 1, CV_MINMAX);
}

cv::Mat Radon::radonImplementation() {
	//Fourier Transformation
	Mat img2;
	dft_radon(this->origin, img2);
	Mat DFT_changed;
	img2.convertTo(DFT_changed, CV_8UC1, 255, 0);

	//Binarization
	Mat img3 = img2.clone();
	threshold(img2, img3, 0.43, 1, CV_THRESH_BINARY);
	Mat Binarization;
	img3.convertTo(Binarization, CV_8UC1, 255, 0);

	//radon transformation
	int length = img3.cols;
	int width = img3.rows;
	Mat img4;
	Mat rotation;
	Mat radon = Mat(origin.rows, angles.size(), CV_32FC1);// radoned image
	for (int i = 0; i<angles.size(); i++) {
		rotation = getRotationMatrix2D(Point2f(length / 2, width / 2), angles[i], 1.0);
		warpAffine(img3, img4, rotation, Size(length, width));// rotate by angle i (matrix rotation)
		for (int j = 0; j<origin.rows; j++) {
			radon.at<float>(j, i) = sum_proj(img4, j); // sum_projcetion
		}
	}

	normalize(radon, radon, 0, 1, CV_MINMAX);
	Mat Radon;
	radon.convertTo(Radon, CV_32S);
	return radon;
}
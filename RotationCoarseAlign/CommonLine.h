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
#include "RadonDriver.h"

using namespace std;
using namespace cv;

#define GET_MAX 0
#define GET_MIN 1

class CommonLineDriver
{
public:
	// constructors
	CommonLineDriver() {
		init();
	}

	CommonLineDriver(string prefix_path, string store_prefix_path, string in_path, string theta_path, string result_path):
		prefixPath(prefix_path),
		storePrefixPath(store_prefix_path),
		inPath(in_path),
		thetaPath(theta_path),
		resultPath(result_path) { 
		init();
	}

	// main pipeline implementation functions
	void correlation_2d_for_Mat_data(Mat &image1, Mat &image2, double *Corr_out, int Nx, int Ny);
	void optimizer(/*vector<Mat> &sinogramsList,*/ vector<string> &filenames);
	void radonTransformation(/*vector<Mat> &sinogramsList*/);
	void rotationExecution();
	void execution();
	// paths setters and getters
	void setThetaPath(std::string path) { this->thetaPath = path; }
	std::string getThetaPath() { return this->thetaPath; }
	void setPrefixPath(std::string path) { this->prefixPath = path; }
	std::string getPrefixPath() { return this->prefixPath; }
	void setStorePrefixPath(std::string path) { this->storePrefixPath = path; }
	std::string getStorePrefixPath() { return this->storePrefixPath; }
	void setResultPath(std::string path) { this->resultPath = path; }
	std::string getResultPath() { return this->resultPath; }

private:
	template <class Type> int argmax(Type begin, Type end);
	int getSinogramIndex(string filename);
	float getExtremeValue(Mat img, int value_type);
	float avgProjection(float* arr);
	Mat partialAverage(int num_avg, vector<Mat> &sinoList);
	void mat2double(Mat &image, double* out_image);
	void init() {
		// basic datas
		img_column = 1024;
		img_row = 1024;
		sino_column = angle;
		sino_row = 1024;
		section_num = 63;

		cout << "Initializing ... ... " << endl;
		cout << "read in source images from MRC file ... ... " << endl;
		// read in source images
		char _image_name[20];
		for (int i = 0; i < section_num; i++) {
			sprintf(_image_name, "BBa%d.jpg", i + 1);         // read images for indices 1 to 63
			Mat _sourceImage = imread(prefixPath + _image_name);
			sourceImages.push_back(_sourceImage);
		}
		cout << "Finish ... ... " << endl;
	}
	// file paths for different requirements
	/*
		@prefixPath: read in your images generated from mrc file
		@storePrefixPath: sinograms storage path
		@inPath : same as prefixPath, but it is used by sprintf
		@thetaPath : optimized thetas storage path
		@resultPath : final aligned images storage path
	*/
	std::string prefixPath = "D:\\用户目录\\我的文档\\Visual Studio 2015\\Projects\\ImageProcessingBeginner\\BBaTiltSeries\\BBaImages\\";
	std::string storePrefixPath = "D:\\用户目录\\我的文档\\Visual Studio 2015\\Projects\\ImageProcessingBeginner\\BBaTiltSeries\\Sinogramsd\\";
	std::string inPath = "D:\\用户目录\\我的文档\\Visual Studio 2015\\Projects\\ImageProcessingBeginner\\BBaTiltSeries\\BBaImages\\*.jpg";
	std::string thetaPath = "D:\\用户目录\\我的文档\\Visual Studio 2015\\Projects\\ImageProcessingBeginner\\BBaTiltSeries\\thetas.txt";
	std::string resultPath = "D:\\用户目录\\我的文档\\Visual Studio 2015\\Projects\\ImageProcessingBeginner\\BBaTiltSeries\\RotatedResults\\";
	// original images size
	int img_column;
	int img_row;
	// sinogram size
	int sino_column;
	int sino_row;
	// number of sections
	int section_num;
	vector<int> opt_thetas;
	vector<Mat> sourceImages;
	vector<Mat> sinogramsList;
	vector<Mat> coarseAlignedImages;
	const int angle = 180;
};

/*
	@Private Methods Implementations
*/
template <class Type>
int CommonLineDriver::argmax(Type begin, Type end)
{
	return std::distance(begin, std::max_element(begin, end));
}

float CommonLineDriver::getExtremeValue(Mat img, int value_type)
{
	/**
	@function: get maximum or minimum pixel in image according to value_type
	input params:
	float *arr : pointer to a float array
	int value_type : 0 for getting Maximum , 1 for getting Minimum
	**/
	float extremeValue = 0;
	if (value_type == 0) {
		extremeValue = -1;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<float>(i, j) > extremeValue) {
					extremeValue = img.at<float>(i, j);
				}
			}
		}
	}

	else {
		extremeValue = 6666;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<float>(i, j) < extremeValue) {
					extremeValue = img.at<float>(i, j);
				}
			}
		}
	}
	return extremeValue;
}

float CommonLineDriver::avgProjection(float *arr)
{
	/**
	@function: compute linear integration using average value
	(this function did bad in radon transformation result, do not use it)
	input param:
	float *arr : pointer to a float array
	**/
	float sum_pixels = 0.0f;
	for (int i = 0; i < img_row; i++) {
		sum_pixels += arr[i];
	}
	return (sum_pixels / img_row);
}

void CommonLineDriver::mat2double(Mat &image, double* out_image)
{
	/*
	@function: convert Mat image into a array storing double elements
	input params:
	Mat &image: image to be converted
	double *out_image: a double array as output
	*/
	int img_rows = image.rows;
	int img_cols = image.cols;
	for (int i = 0; i < img_cols; i++) {
		int *temp = image.ptr<int>(i);
		for (int j = 0; j < img_rows; j++) {
			out_image[i + j*img_cols] = (double)temp[j];
		}
	}
}

Mat CommonLineDriver::partialAverage(int num_avg, vector<Mat> &sinoList)
{
	/**
		@function: compute partial average image for sinogram under processed
		input params:
		int num_avg : index for current sinogram under computing
		vector<Mat> &sinoList : sinogram list
	**/
	Mat _temp_summary = Mat(sino_row, sino_column, CV_32S, cv::Scalar::all(0));
	Mat _temp_current_sinogram = Mat(sino_row, sino_column, CV_32S, cv::Scalar::all(0));
	for (int i = 0; i < section_num; i++) {
		imshow("temp", sinoList[i]);
		waitKey(0);
		if (i == num_avg)	continue;
		else {
			_temp_current_sinogram = sinoList[i];
			_temp_current_sinogram.convertTo(_temp_current_sinogram, CV_8UC1);
			cvtColor(_temp_current_sinogram, _temp_current_sinogram, CV_BGR2GRAY);
			_temp_current_sinogram.convertTo(_temp_current_sinogram, CV_32S);
			_temp_summary += _temp_current_sinogram;
		}
	}
	return _temp_summary;
}

int CommonLineDriver::getSinogramIndex(string filename)
{
	/**
	@function: convert the index string into a integer
	input params:
	string filename : pointer to a float array
	**/
	int _dot_pos = filename.find(".", 0);
	string _sino_index = filename.substr(0, _dot_pos);
	int ind = atoi(_sino_index.c_str());
	return ind;
}

/*
	@Public Methods Implementations
*/
void CommonLineDriver::correlation_2d_for_Mat_data(Mat &image1, Mat &image2, double *Corr_out, int Nx, int Ny)
{
	/**
	@function: implemetation for cross correlation algorithm based on two sinograms for Radon Transformation
	input params:
	Mat &image1 : index for current sinogram under computing
	Mat &image2 : sinogram list
	double corr_out : result image for correlation computation
	int Nx, Ny : input images size
	**/
	unsigned long n;
	if (Nx % 2 == 1 || Ny % 2 == 1) {        // A constraint to image size
		fprintf(stdout, "Error in fuction 'correlation_2d()'! Image size must be even in both directon!\n");
		exit(1);
	}

	unsigned long image_in_size2d = Nx*Ny;
	unsigned long image_out_size2d = (Nx / 2 + 1)*Ny;
	double sqrt_xy = sqrt((double)Nx*Ny);

	fftw_complex *Fft_out1, *Fft_out2, *Fft_mult;
	Fft_out1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*image_out_size2d);
	Fft_out2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*image_out_size2d);
	Fft_mult = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*image_out_size2d);

	double *image1_double, *image2_double;
	image1_double = new double[Nx*Ny];
	image2_double = new double[Nx*Ny];

	mat2double(image1, image1_double);
	mat2double(image2, image2_double);

	// executing fft......
	fftw_plan Plan;

	Plan = fftw_plan_dft_r2c_2d(Ny, Nx, image1_double, Fft_out1, FFTW_ESTIMATE);
	fftw_execute(Plan);
	fftw_destroy_plan(Plan);

	Plan = fftw_plan_dft_r2c_2d(Ny, Nx, image2_double, Fft_out2, FFTW_ESTIMATE);
	fftw_execute(Plan);
	fftw_destroy_plan(Plan);

	//mult fft_out1 and conjugate of fft_out2
	//(a + bi)(c - di) = (ac + bd) + (bc - ad)i
	for (n = 0; n<image_out_size2d; n++) {
		//normallization
		Fft_out1[n][0] /= sqrt_xy;
		Fft_out1[n][1] /= sqrt_xy;
		Fft_out2[n][0] /= sqrt_xy;
		Fft_out2[n][1] /= sqrt_xy;

		Fft_mult[n][0] = Fft_out1[n][0] * Fft_out2[n][0] + Fft_out1[n][1] * Fft_out2[n][1];
		Fft_mult[n][1] = Fft_out1[n][1] * Fft_out2[n][0] - Fft_out1[n][0] * Fft_out2[n][1];
	}

	//executing inverse fft
	Plan = fftw_plan_dft_c2r_2d(Ny, Nx, Fft_mult, Corr_out, FFTW_ESTIMATE);
	fftw_execute(Plan);
	fftw_destroy_plan(Plan);

	//calculate the image module for normalization
	double Img_mod1, Img_mod2, Norm_factor;
	Img_mod1 = 0.0;
	Img_mod2 = 0.0;

	for (n = 0; n<image_in_size2d; n++) {
		Img_mod1 += image1_double[n] * image1_double[n];
		Img_mod2 += image2_double[n] * image2_double[n];
	}
	Norm_factor = sqrt(Img_mod1*Img_mod2);

	//normalize corr_out and convert it to float
	for (n = 0; n<image_in_size2d; n++) {
		Corr_out[n] = Corr_out[n] / Norm_factor;
	}

	//free fft_out1, fft_out2, fft_mult;
	fftw_free(Fft_out1);
	fftw_free(Fft_out2);
	fftw_free(Fft_mult);
	return;
}

void CommonLineDriver::optimizer(/*vector<Mat> &sinogramsList, */vector<string> &filenames)
{
	// compute optimized angles
	double *corr_out = new double[sino_column*sino_row];
	double avg_pixel_in_corr;
	int fine_theta = 0, index;
	vector<double> _temp_row_pixel;
	ofstream fout(thetaPath);
	Mat avg_sino, temp;
	for (int i = 0; i < section_num; i++) {
		// convert file name into sinogram index
		index = getSinogramIndex(filenames[i]);
		cout << "Processing Image : " << index << " ... ..." << endl;
		// compute partial average
		avg_sino = partialAverage(index-1, this->sinogramsList);
		// compute correlation for current section image
		temp = this->sinogramsList.at(index-1);
		temp.convertTo(temp, CV_32S);
		correlation_2d_for_Mat_data(temp, avg_sino, corr_out, temp.rows, temp.cols);
		// find optimized rotation theta for current sinogram
		// algorithm: brute force
		_temp_row_pixel.clear();
		avg_pixel_in_corr = 0;
		for (int k = 0; k < avg_sino.cols; k++) {
			avg_pixel_in_corr = 0;
			for (int j = 0; j < avg_sino.rows; j++) {
				avg_pixel_in_corr += corr_out[k + j*avg_sino.cols];
			}
			//cout << "In column " << k << " " << avg_pixel_in_corr << endl;
			_temp_row_pixel.push_back(avg_pixel_in_corr/sino_row);
		}
		// write thetas into file
		fine_theta = argmax(_temp_row_pixel.begin(), _temp_row_pixel.end());
		opt_thetas.push_back(fine_theta - 90);
		fout << fine_theta - 90 << endl;
		cout << "Image : " << index << " finished !\n" << endl;
	}
	fout.close();
	return;
}

void CommonLineDriver::radonTransformation(/*vector<Mat> &sinogramsList*/)
{
	/**
		@function: radon transformation
		input params:
		vector<Mat> &sinogramsList : sinograms storage
	**/
	int _file_counter = 0;
	Mat _sourceImage;
	char _store_name[20];
	// A loop for Radon Algorithm
#pragma omp parallel
	while (_file_counter < section_num) {
		Mat _radon_image = Mat(img_row, angle, CV_32S, cv::Scalar::all(0));  // initialized radon sinogram
		//sprintf(_image_name, "BBa%d.jpg", _file_counter+1);         // read images for indices 1 to 63
		_sourceImage = sourceImages[_file_counter];
		cout << "Processing : image " << _file_counter+1 << endl;
		cvtColor(_sourceImage, _sourceImage, CV_BGR2GRAY);
		Radon radonDriver(_sourceImage);
		_radon_image = radonDriver.radonExecution();

		// image normalization : normalize the pixel value into 0-255 which is a correct range for storage
		float max_value = getExtremeValue(_radon_image, GET_MAX);
		float min_value = getExtremeValue(_radon_image, GET_MIN);
		for (int i = 0; i < _radon_image.rows; i++) {
			for (int j = 0; j < _radon_image.cols; j++) {
				_radon_image.at<float>(i, j) = (_radon_image.at<float>(i, j) - min_value) * 255 / (max_value - min_value);
			}
		}
		this->sinogramsList.push_back(_radon_image);
		_radon_image.convertTo(_radon_image, CV_32S);
		sprintf(_store_name, "%d.jpg", _file_counter+1);
		imwrite(storePrefixPath + _store_name, _radon_image);
		cout << "Finished !\n" << endl;
		_file_counter++;
	}
	return;
}

void CommonLineDriver::execution()
{
	cout << "Step 1 Radon Transformation" << endl;
	radonTransformation(/*sinogramsList*/);
	Directory _sinogram_dir;
	vector<string> filenames = _sinogram_dir.GetListFiles(storePrefixPath, "*.jpg", false);
	std::sort(filenames.begin(), filenames.end());
	//Mat ptrSrc;
	//string name, fullname;
	//for (int i = 0; i < filenames.size(); i++) {
	//	name = filenames[i];
	//	fullname = storePrefixPath + name;
	//	ptrSrc = imread(fullname);
	//	this->sinogramsList.push_back(ptrSrc);
	//}
	cout << "Step 2 Cross Correlation Optimization" << endl;
	optimizer(/*sinogramsList, */filenames);
	cout << "Processing Finished !" << endl;
	system("pause");
	rotationExecution();
	return;
}

void CommonLineDriver::rotationExecution()
{
	Mat ptrSrc_new, temp, m0, m1, mr, rotation, rotated;
	string name, fullname;
	cout << "Execute Results Now ... ... ..." << endl;
	Directory prefix;
	vector<string> filenames_new = prefix.GetListFiles(prefixPath, "*.jpg", false);
	std::sort(filenames_new.begin(), filenames_new.end());
	int center = img_row / 2;
	float shift0[] = {
		1,0,-center,
		0,1,-center,
		0,0,1
	};
	float shift1[] = {
		1,0,center,
		0,1,center,
		0,0,1
	};
	float R[] = {
		1,1,0,
		1,1,0,
		0,0,1
	};
	m0 = Mat(3, 3, CV_32FC1, shift0);
	m1 = Mat(3, 3, CV_32FC1, shift1);
	for (int t = 0; t < filenames_new.size(); t++) {
		temp = this->sourceImages[t];
		temp.convertTo(temp, CV_32FC1);
		double theta = (opt_thetas[t] * CV_PI) / 180;
		R[0] = cos(theta);
		R[1] = sin(theta);
		R[3] = -sin(theta);
		R[4] = cos(theta);
		mr = Mat(3, 3, CV_32FC1, R);
		rotation = m1*mr*m0; // 将图像平移至坐标原点, 完成theta角旋转, 再将其移回至原始中心位置
		warpPerspective(temp, rotated, rotation, Size(temp.rows, temp.cols), WARP_INVERSE_MAP);		// 透视变换
		rotated.convertTo(rotated, CV_32S);
		imwrite(resultPath + filenames_new[t], rotated);
	}
	cout << "Finish ! " << endl;
	return;
}
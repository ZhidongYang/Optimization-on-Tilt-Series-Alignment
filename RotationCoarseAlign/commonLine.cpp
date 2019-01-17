#include "CommonLine.h"

const float PI = 3.14159265358979f;

int main()
{
	CommonLineDriver driver;
	driver.execution();
	//Mat temp_sino = Mat(1024, 180, CV_32S, cv::Scalar::all(0));
	//temp_sino = imread("D:\\用户目录\\我的文档\\Visual Studio 2015\\Projects\\ImageProcessingBeginner\\BBaTiltSeries\\Sinogramsd\\1.jpg");
	//temp_sino.convertTo(temp_sino, CV_8U);
	//cvtColor(temp_sino, temp_sino, CV_BGR2GRAY);
	//imshow("dfsds", temp_sino);
	//imwrite("test.jpg", temp_sino);
	//waitKey(0);
	return 0;
}
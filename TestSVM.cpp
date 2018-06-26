#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

using namespace cv::ml;
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if( argc != 3 )
	{
		return -1;
	}
	
	Ptr<cv::ml::SVM> svm;
	svm = Algorithm::load<SVM>(argv[1]);
	
	Size pic = Size(128,128);
	Mat testimg = imread(argv[2]);
	resize(testimg, testimg, Size(128,128));
	
	HOGDescriptor hog;
    hog.winSize = pic;
    Mat testimggray;
	cvtColor( testimg, testimggray, COLOR_BGR2GRAY );
    vector< float > featureVectortest;
	hog.compute( testimggray, featureVectortest, Size( 8, 8 ), Size( 0, 0 ) );
	Mat testt = Mat( featureVectortest ).clone();
	const int cols = (int)std::max( testt.cols, testt.rows );
	
	Mat testtmp( 1, cols, CV_32FC1 ); // transposition Mat

    transpose( testt, testtmp );
	
	float erg = svm->predict(testtmp);
	cout << "SVM Output: " << erg << endl;
}
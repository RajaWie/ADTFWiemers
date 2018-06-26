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

void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages, Size img_size );
void createHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst);
void cropTo( const int Size, const Mat inputimg, const String savedirname, const String name );//cut input to many pieces

void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages = false , Size img_size = Size(128,128))
{
    vector< String > files;
    glob( dirname, files );

    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] );  // load the image
		resize(img, img, img_size); // bring image in size (INTER_LINEAR - a bilinear interpolation (used by default))
        if ( img.empty() )             // invalid image, skip it.
        {
            cout << files[i] << " is invalid!" << endl;
            continue;
        }

        if ( showImages )
        {
			namedWindow("image", WINDOW_AUTOSIZE);
            imshow( "image", img );
            waitKey( 1 );
        }
        img_lst.push_back( img );
    }
}
void createHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst )
{
    HOGDescriptor hog;
    hog.winSize = wsize;
    Mat imggray;
    vector< float > featureVector;

    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
			cvtColor( img_lst[i], imggray, COLOR_BGR2GRAY );	
			hog.compute( imggray, featureVector, Size( 8, 8 ), Size( 0, 0 ) );
			gradient_lst.push_back( Mat( featureVector ).clone() );
    }
}

void cropTo( const int iSize, const Mat inputimg, const String savedir, const String name ){
	int roundsx = inputimg.cols/iSize;
	int roundsy = inputimg.rows/iSize;
	int counter = 0;
	
	for( size_t x = 0 ; x < roundsx; x++ )
    {
		for( size_t y = 0 ; y < roundsy; y++ )
		{
			String filename = savedir + name;
			filename += "_" + std::to_string(counter) + ".ppm";
			Rect myROI(x*iSize, y*iSize, iSize, iSize);
			Mat finpic = inputimg(myROI);
			vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PXM_BINARY);
			compression_params.push_back(1);
			imwrite(filename, finpic, compression_params);
			counter++;
		}
	}
}

int main(int argc, char* argv[])
{
	if( argc != 4 )
	{
		printf("usage: ./TrainSVM <posititve_folder> <negative_folder> <save_location_svm>\n");
		return -1;
	}
	String pos_dir = argv[1];
	String neg_dir = argv[2];
	if (pos_dir.empty()|| neg_dir.empty())
	{
		cout << "missing images" << endl;
		return -1;
	}
	vector< Mat > pos_lst, neg_lst, gradient_lst;
	vector< int > labels;
	Size pic = Size(128,128);
	bool visualization = true;
	
	//load positive
	load_images( pos_dir, pos_lst, visualization, pic );
	createHOGs( pic, pos_lst, gradient_lst);
	//fill positive labels
	size_t pos_count = gradient_lst.size();
    labels.assign( pos_count, +1 );
	cout << pos_count << " positive HOGs done..." << endl;
	
	//load negative
	load_images( neg_dir, neg_lst, visualization, pic );
	createHOGs( pic, neg_lst, gradient_lst);
	//fill negative labels
	size_t neg_count = gradient_lst.size() - pos_count;
    labels.insert( labels.end(), neg_count, -1 );
	cout << neg_count << " negative HOGs done..." << endl;
	
	cout << gradient_lst.size() << " HOGs done..." << endl;
	
	const int rows = (int)gradient_lst.size();
    const int cols = (int)std::max( gradient_lst[0].cols, gradient_lst[0].rows );
	Mat tmp( 1, cols, CV_32FC1 ); // transposition Mat
    Mat trainData = Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < gradient_lst.size(); ++i )
    {
        transpose( gradient_lst[i], tmp );
        tmp.copyTo( trainData.row( (int)i ) );
    }
	cout << "Training Matrix Done... " << endl;
	
	cout << " SVM start training..." << endl;

	Ptr< SVM > svm = SVM::create();
    /* Default values to train SVM */
    svm->setCoef0( 0.0 );
    svm->setDegree( 3 );
    svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
    svm->setGamma( 0 );
    svm->setKernel( SVM::LINEAR );
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 0.01 ); // From paper, soft classifier
    svm->setType( SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    svm->train( trainData, ROW_SAMPLE, labels );
	
    cout << "...[done]" << endl;
	
	svm->save(argv[3]);
	cout << "SVM File saved to: " << argv[3] << endl;
	
	waitKey( 1 );

	

}

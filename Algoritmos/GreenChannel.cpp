#include "Source.cpp"


void func1(Mat src) {
	clock_t tStart = clock();
	
		resize(640, src);		
		getEachBGR(GREEN, src);
		medianFilter(11, src);
		//medianFilter(3, src);
		namedWindow( "Source", CV_WINDOW_AUTOSIZE );
		imshow( "Source", src );
		
	cout<<endl<<"Manual: "<<(double)(clock() - tStart)/CLOCKS_PER_SEC;
}

void func2(Mat srcOCV) {
	clock_t tStart = clock();
		//cvtColor(srcOCV, srcOCV, CV_BGR2GRAY);
		getEachBGR(GREEN, srcOCV);
		medianBlur( srcOCV, srcOCV, 51);
		
		namedWindow( "OCV", CV_WINDOW_AUTOSIZE );
		imshow( "OCV", srcOCV );
		
	cout<<endl<<"OpenCV: "<<(double)(clock() - tStart)/CLOCKS_PER_SEC;
}



Mat medFil(Mat src, int kernelSize) {
	int bounds = (kernelSize-1)/2;
	//Mat crop(src.rows-bounds*2,src.cols-bounds*2,CV_8UC1,0);
	
	Rect roi(bounds, bounds, src.cols-bounds*2, src.rows-bounds*2);
	Mat sub( src, roi );
	
	cout<<sub.rows<<":"<<sub.cols<<endl;
	
	return sub;
	//src = sub.clone();
}



void cropImage(Mat src, int kernelSize, int subImages) {
	
	Vector<Mat> subMats(subImages);
	int rowCol = sqrt(subImages);
	int bounds = (kernelSize-1)/2;
	int subCols = (src.cols/3)+bounds*2;
	int subRows = (src.rows/3)+bounds*2;
	
	Mat out = src.clone();
	
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(100);
	
	copyMakeBorder( src, src, bounds, bounds, bounds, bounds, BORDER_REPLICATE );
	imwrite("border.jpg", src, compression_params);
	
	cout<<src.cols<<" : "<<src.rows<<endl;
	
	
	int k = 0;
	for(int i = 0; i < rowCol; i++) {
		for(int j = 0; j < rowCol; j++) {
			cout<<(subCols-bounds*2)*j<<" : "<<(subRows-bounds*2)*i<< " # "<<subCols<<" : "<<subRows<<endl;
			Rect roi((subCols-bounds*2)*j, (subRows-bounds*2)*i, subCols, subRows);
			Mat sub( src, roi );
			//ostringstream s;
			//s <<"sub"<<i<<j<<".jpg";
			//string query(s.str());
			subMats[k] = sub.clone();
			//imwrite(query, subMats[k], compression_params);
			k++;
			
		}
	}
	
	
	//thread k1 (medFil, subMats[0], kernelSize);
	//k1.join();
	
	imwrite("ffffff.png", subMats[0], compression_params);
	
	Mat k1 = medFil(subMats[0], kernelSize);
	Mat k2 = medFil(subMats[1], kernelSize);
	Mat k3 = medFil(subMats[2], kernelSize);
	Mat k4 = medFil(subMats[3], kernelSize);
	Mat k5 = medFil(subMats[4], kernelSize);
	Mat k6 = medFil(subMats[5], kernelSize);
	Mat k7 = medFil(subMats[6], kernelSize);
	Mat k8 = medFil(subMats[7], kernelSize);
	Mat k9 = medFil(subMats[8], kernelSize);
	
	
	//auto k10 = std::async(medFil(subMats[0], kernelSize));
	
	
	Mat n1, n2, n3, new_image;

	vconcat(k1,k4,n1);
	vconcat(n1,k7,n1);
	
	vconcat(k2,k5,n2);
	vconcat(n2,k8,n2);
	
	vconcat(k3,k6,n3);
	vconcat(n3,k9,n3);
	
	hconcat(n1, n2, new_image);
	hconcat(new_image, n3, new_image);
	

	imwrite("mer.png", new_image, compression_params);
	
	/*for (int j = 0; j < image1.rows; j++) {
		for (int i = 0; i < image1.cols; i++) {
		    mergeMat.at<cv::Vec3b>(j,i) = image1.at<cv::Vec3b>(j,i);
		}
        for (int i = image1.cols; i < mergeMat.cols; i++) {
            mergeMat.at<cv::Vec3b>(j,i) = image2.at<cv::Vec3b>(j,i);
        }*/
	
	
}


/**
 * @function main
 */ 
int main( int argc, char** argv ) {

	Mat src, srcOCV;

  	/// Load image
  	src = imread( argv[1], 1 );
  	srcOCV = src;

	if( !src.data ) {
		return -1;
	}
	
	
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(100);
	
	
	clock_t tStart = clock();
	
		resize(640, src);		
		getEachBGR(GREEN, src);
		medianFilter(7, src);
		medianFilter(3, src);
		//localTresh(src);
		//medianFilter(9, src);
		namedWindow( "Source", CV_WINDOW_AUTOSIZE );
		imwrite("11*11.png", src, compression_params);
		imshow( "Source", src );
		
	cout<<endl<<"Manual: "<<(double)(clock() - tStart)/CLOCKS_PER_SEC;
	
	cropImage(src, 51, 9);
	
	//thread one(func1, src);
	//thread two(func2, srcOCV);
	
	//one.join();
	//two.join();	

	cout<<endl;	
	
	waitKey(0);

  	return 0;

}

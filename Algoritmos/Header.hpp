#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <math.h>
#include <string>
#include <sstream>
#include <numeric>

using namespace std;
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0 
#define A 1
#define B 1
 
void getEachBGR(int, Mat&);

void resize(float, Mat&);
void medianFilter(int, Mat&);
void localTresh(Mat&);

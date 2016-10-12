//#include "Source.cpp"
#include <iostream>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<numeric>
#include <math.h>
#include<vector>

using namespace std;
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0
#define DILATACION 0
#define EROSION 1
#define OR 0
#define AND 1
#define NOT 2
#define A 1
#define B 1
/*void getEachBGR(int opt, Mat& src);
void histograma(Mat im, int hist[]);
void cumhist(int hist[], int cumhist[]);
void histDisplay(int histogram[], const char* name);
Mat eqHist(Mat im);
Mat morfologia(Mat im, Mat ker, int type);*/
void getEachBGR(int opt, Mat& src) {
	//Definimos un vector de matrices donde estaran alojados cada canal BGR
	vector<Mat> bgr_planes;
	//Separamos la imagen a color en tres imagenes en escala de grises, una por cada canal
	split(src, bgr_planes);
	//Igualamos la imagen de entrada con el plano elegido en el parametro opt
	switch (opt) {
	case BLUE:
		src = bgr_planes[0];
		break;
	case GREEN:
		src = bgr_planes[1];
		break;
	case RED:
		src = bgr_planes[2];
		break;
	default:
		break;
	}


}
void resize(float factor, Mat& src) {

	float tempVal;
	int xr, yr;

	//Obteniendo valores de factor de escala para no perder proporcionalidad.
	if (src.cols == src.rows) {
		xr = factor;
		yr = factor;
	}
	else if (src.cols > src.rows) {
		xr = factor;
		tempVal = src.rows / (src.cols / factor);
		yr = (int)round(tempVal);
	}
	else {
		yr = factor;
		tempVal = src.cols / (src.rows / factor);
		yr = (int)round(tempVal);
	}

	//Se ocupa la funcion resize de opencv con el parametro INTER_AREA, el mas adecuado para reducir imagenes.
	resize(src, src, Size(xr, yr), 0, 0, INTER_AREA);
}

void histograma(Mat im, int hist[])
{
	// inicializa los valores a 0
	for (int i = 0; i < 256; i++)
	{
		hist[i] = 0;
	}

	// calcular el # de pixeles por intensidad
	for (int y = 0; y < im.rows; y++)
		for (int x = 0; x < im.cols; x++)
			hist[(int)im.at<uchar>(y, x)]++;

}

void cumhist(int hist[], int cumhist[]){
	cumhist[0] = hist[0];

	for (int i = 1; i < 256; i++)
	{
		cumhist[i] = hist[i] + cumhist[i - 1];
	}
}

void eqHist(Mat& im) {
	// generar wl histograma
	int hist[256];
	histograma(im, hist);

	// Calucular tamaño de la imagen
	int size = im.rows * im.cols;
	float alpha = 255.0 / size;

	// Calcular probabilidades
	float PrRk[256];
	for (int i = 0; i < 256; i++)
	{
		PrRk[i] = (double)hist[i] / size;
	}

	// Generate histograma de frecuncia acumulada
	int cumhisto[256];
	cumhist(hist, cumhisto);

	// Escalando histograma
	int Sk[256];
	for (int i = 0; i < 256; i++)
	{
		Sk[i] = cvRound((double)cumhisto[i] * alpha);
	}

	// Generabdo histograma equalizado
	float PsSk[256];
	for (int i = 0; i < 256; i++) {
		PsSk[i] = 0;
	}

	for (int i = 0; i < 256; i++) {
		PsSk[Sk[i]] += PrRk[i];
	}

	int final[256];
	for (int i = 0; i < 256; i++)
		final[i] = cvRound(PsSk[i] * 255);
	// Generando imagen equalizada
	Mat new_image = im.clone();

	for (int y = 0; y < im.rows; y++)
		for (int x = 0; x < im.cols; x++)
			new_image.at<uchar>(y, x) = saturate_cast<uchar>(Sk[im.at<uchar>(y, x)]);

	// Mostrar imagen original
	//namedWindow("Imagen Original");
	//imshow("Imagen Original", im);

	// Mostrar Histograma original
	//histDisplay(hist, " Histograma original");

	// Mostrar  imgen equalizada
	//namedWindow("imgen equalizada");
	//imshow("Imagen Original", new_image);

	// Mostrar  Histogrma equalizado
	//histDisplay(final, "Histogrma equalizado");
	im= new_image;

}
 void morfology(Mat& im, Mat ker, int type) {
	int size = ker.rows / 2;
	//cout << im.cols << "," << im.rows << endl;
	//cout << im.cols + size << "," << im.rows +size << endl;
	Mat out = im.clone();
	//Mat aux;
	//copyMakeBorder(im, aux, size, size, size, size, BORDER_REPLICATE);
	vector<int> lista;
	int max = 0;
	int min = 255;
	int val = 0;
	for (int y = 0; y < im.rows; y++) {
		for (int x = 0; x < im.cols; x++) {
			//cout << x << "," << y << endl;
			if (x ==0 || y == 0 || x == im.cols - 1 || y == im.rows - 1) {
				out.at<uchar>(y, x) = saturate_cast<uchar>(im.at<uchar>(y, x));
			}
			else {
				lista.clear();
				for (int j = 0; j < ker.rows; j++) {
					//cout << x << "," << y << endl;
					for (int i = 0; i < ker.cols; i++) {
						if (ker.at<int>(j, i) == 1) {
							//cout << ker.at<int>(1,1)<<" ";
							lista.push_back((int)im.at<uchar>(y + j - size, x + i - size));
						}
					}
				}if (type == DILATACION) {
					for (int a = 0; a < lista.size(); a++) {
						if (max < lista[a]) {
							max = lista[a];

						}
					}
					val = max;
					max = 0;
				}
				else if (type == EROSION) {
					for (int a = 0; a < lista.size(); a++) {
						if (min > lista[a]) {
							min = lista[a];

						}
					}
					val = min;
					min = 255;
				}
				//cout << max << endl;
				out.at<uchar>(y, x) = saturate_cast<uchar>(val);


			}
		}
	}

	//return out;
	im = out;
}
 void medianFilter(int size, Mat& src) {
 	//Valores para  el manejo del kernel
 	int midValue = (size*size-1)/2;
	int bounds = (size-1)/2;
        vector<int> window(size*size);
 
 	Mat dst;
        dst = src.clone();
        
        //Replicamos bordes para poder iterar toda la matriz
        copyMakeBorder( src, src, bounds, bounds, bounds, bounds, BORDER_REPLICATE );
 
 	//Se itera para cada valor dentro de la matriz
 	//Se inicia desde el valor bounds para evitar valores basura fuera de la imagen de entrada src
        for(int y = bounds; y < src.rows-bounds; y++){
            for(int x = bounds; x < src.cols-bounds; x++){
                // Para cada pixel x,y, se itera una submatriz de tamaño size*size que sera el kernel de trabajo
                int p = 0;
                for(int i = -bounds; i <= bounds; i++) {
                	for(int j = -bounds; j <= bounds; j++) {
                		//Se guarda cada valor i, j en el vector destinado al kernel
                		window[p] = src.at<uchar>(y+i, x+j);
                		p++;		
                	}
                }
 
 		//Se utiliza la funcion nth_element que obtiene el valor deseado de manera rapida, ordenando solo
 		//los elementos necesarios para obtener el valor deseado
                nth_element(window.begin(), window.begin()+midValue, window.end());
                //La matriz destino es llenada con el valor de mediana del kernel
                dst.at<uchar>(y-bounds,x-bounds) = window[midValue];
            }
        }
        //Se iguala con la matriz de entrada src
        //src = src0;
        src = dst;
	
}
void estimation(Mat &src, int s, int f) {

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(100);

	imwrite("source.png", src, compression_params);
	Mat gray = src;
	medianFilter(s, src);
	imwrite("smooth.png", src, compression_params);
	//sub(gray, src);
	Mat tmp;
	subtract(src, gray, tmp);
	gray = tmp;
	imwrite("subtract.png", gray, compression_params);
	medianFilter(f, gray);
	imwrite("filter.png", gray, compression_params);
	
	src = gray;
		
}
void Tresh(Mat& src) {
	int size = 3;

	//Valores para  el manejo del kernel
	int midValue = (size*size - 1) / 2;
	int bounds = (size - 1) / 2;

	vector<double> window(size*size);

	Mat dst;
	dst = src.clone();

	//Replicamos bordes para poder iterar toda la matriz
	copyMakeBorder(src, src, bounds, bounds, bounds, bounds, BORDER_REPLICATE);

	//Se itera para cada valor dentro de la matriz
	//Se inicia desde el valor bounds para evitar valores basura fuera de la imagen de entrada src
	for (int y = bounds; y < src.rows - bounds; y++) {
		for (int x = bounds; x < src.cols - bounds; x++) {
			// Para cada pixel x,y, se itera una submatriz de tamaño size*size que sera el kernel de trabajo
			int p = 0;
			for (int i = -bounds; i <= bounds; i++) {
				for (int j = -bounds; j <= bounds; j++) {
					//Se guarda cada valor i, j en el vector destinado al kernel
					window[p] = src.at<uchar>(y + i, x + j);
					p++;
				}
			}

			//Se utiliza la funcion nth_element que obtiene el valor deseado de manera rapida, ordenando solo
			//los elementos necesarios para obtener el valor deseado

			double sum = std::accumulate(window.begin(), window.end(), 0.0);
			double mean = sum / window.size();

			double sq_sum = std::inner_product(window.begin(), window.end(), window.begin(), 0.0);
			double stdev = std::sqrt(sq_sum / window.size() - mean * mean);

			double thresh = A*stdev + B*mean;

			int value = 0;

			if (dst.at<uchar>(y - bounds, x - bounds) > thresh)
				value = 0;
			else
				value = 255;


			//nth_element(window.begin(), window.begin()+midValue, window.end());
			//La matriz destino es llenada con el valor de mediana del kernel
			//int median = average;

			dst.at<uchar>(y - bounds, x - bounds) = value;
		}
	}
	//Se iguala con la matriz de entrada src
	//src = src0;
	src = dst;


}
void Thg(Mat& im, int t) {
	for (int y = 0; y < im.rows; y++) {
		for (int x = 0; x < im.cols; x++) {
			if (im.at<uchar>(y, x) > t) {
				im.at<uchar>(y, x) = saturate_cast<uchar>(255);
			}
			else {
				im.at<uchar>(y, x) = saturate_cast<uchar>(0);
			}
		}
	}
}

void mopen(Mat& src, Mat ker) {
	Mat aux = src;
	morfology(aux, ker, EROSION);
	morfology(aux, ker, DILATACION);
	src = aux;
}

Mat restar(Mat m1, Mat m2) {
	Mat result = m1.clone();
	int tmp;
	for (int y = 0; y < result.rows; y++) {
		for (int x = 0; x < result.cols; x++) {
			// Para cada pixel x,y, se itera una submatriz de tamaño size*size que sera el kernel de trabajo
			tmp = m1.at<uchar>(y, x) - m2.at<uchar>(y, x);
			tmp = 255 - tmp;
			if (tmp < 0) {
				result.at<uchar>(y, x) = 0;
				cout << endl << tmp << endl;
			}
			else
				result.at<uchar>(y, x) = tmp;
		}
	}

	return result;
}
Mat restb(Mat m1, Mat m2) {
	Mat out = m1.clone();
	int sal = 0;
	for (int y = 0; y < m1.rows; y++) {
		for (int x = 0; x < m1.cols; x++) {
			sal = m1.at<uchar>(y, x) - m2.at<uchar>(y, x);
			if (sal < 0) {
				sal = 0;
			}			
				out.at<uchar>(y, x) = saturate_cast<uchar>(sal);
				//cout << sal<<endl;
			
			
			
		}
	}
	return out;
}
Mat logic(Mat m1, Mat m2, int type) {
	Mat out = m1.clone();
	for (int y = 0; y < m1.rows; y++) {
		for (int x = 0; x < m1.cols; x++) {
			//cout << x << "," << y << endl;
			if (type == OR) {
				if (m1.at<uchar>(y, x) == 255 || m2.at<uchar>(y, x) == 255) {
					out.at<uchar>(y, x) = saturate_cast<uchar>(255);
				}
				else {
					out.at<uchar>(y, x) = saturate_cast<uchar>(0);
				}
			}
			if (type==AND) {
				if (m1.at<uchar>(y, x) == 255 && m2.at<uchar>(y, x) == 255) {
					out.at<uchar>(y, x) = saturate_cast<uchar>(255);
				}
				else {
					out.at<uchar>(y, x) = saturate_cast<uchar>(0);
				}
			}
			if (type == NOT) {
				if (m1.at<uchar>(y, x) == 255) {
					out.at<uchar>(y, x) = saturate_cast<uchar>(0);
				}
				else {
					out.at<uchar>(y, x) = saturate_cast<uchar>(255);
				}
			}
		}
	}
	return out;
}

void derivativeFilter(Mat & im) {
	Mat out = im.clone();
	int pixel=0;
	int i1, i2, i3, i4, i5, i6, i7, i8, i9;
	for (int y = 0; y < im.rows; y++) {
		for (int x = 0; x < im.cols; x++) {
			//cout << x << "," << y << endl;
			if (x == 0 || y == 0 || x == im.cols - 1 || y == im.rows - 1) {
				out.at<uchar>(y, x) = saturate_cast<uchar>(im.at<uchar>(y, x));
			}
			else {
				i1 = im.at<uchar>(y - 1, x - 1);
				i2 = im.at<uchar>(y - 1, x);
				i3 = im.at<uchar>(y - 1, x + 1);
				i4 = im.at<uchar>(y , x - 1);
				i5 = im.at<uchar>(y, x);
				i6 = im.at<uchar>(y, x + 1);
				i7 = im.at<uchar>(y + 1, x - 1);
				i8 = im.at<uchar>(y + 1, x);
				i9= im.at<uchar>(y + 1, x + 1);
				pixel=abs((i7+i8+i9)-(i1+i2+i3))+abs((i3+i6+i9)-(i1+i4+i7));
				out.at<uchar>(y,x)= saturate_cast<uchar>(pixel);

			}
		}
	}
	im = out;
}
Mat createKernel(int kernel_size) {

	Mat ker = Mat::zeros(kernel_size, kernel_size, CV_32F);
	// Desviacion estandar igual a 1
	double sigma = 1.0;
	double r, s = 2.0 * sigma * sigma;

	// Suma para normalizar el kernel
	double sum = 0.0;
	double tmpVal = 0.0;

	int bounds = (kernel_size - 1) / 2;

	// Generando un kernel 5 x 5
	for (int x = -bounds; x <= bounds; x++)
	{
		for (int y = -bounds; y <= bounds; y++)
		{
			r = sqrt(x*x + y*y);

			tmpVal = (exp(-(r*r) / s)) / (3.1416 * s);
			ker.at<float>(x + bounds, y + bounds) = tmpVal;
			sum += tmpVal;

		}
	}

	// Normalizar el kernel
	for (int i = 0; i < kernel_size; ++i)
		for (int j = 0; j < kernel_size; ++j)
			ker.at<float>(j, i) /= sum;

	return ker;

}
void gaussianFilter(Mat& src, int k_size) {

	Mat gker = createKernel(k_size);

	//Valores para  el manejo del kernel
	int midValue = (k_size*k_size - 1) / 2;
	int bounds = (k_size - 1) / 2;

	//Vector<double> window(k_size*k_size);

	Mat dst;
	dst = src.clone();

	//Replicamos bordes para poder iterar toda la matriz
	copyMakeBorder(src, src, bounds, bounds, bounds, bounds, BORDER_REPLICATE);

	//Se itera para cada valor dentro de la matriz
	//Se inicia desde el valor bounds para evitar valores basura fuera de la imagen de entrada src
	for (int y = bounds; y <= dst.rows; y++) {
		for (int x = bounds; x <= dst.cols; x++) {
			// Para cada pixel x,y, se itera una submatriz de tamaño size*size que sera el kernel de trabajo
			int sum = 0;
			for (int i = 0; i < k_size; i++) {
				for (int j = 0; j < k_size; j++) {
					//Se guarda cada valor i, j en el vector destinado al kernel
					double pixel = src.at<uchar>(y + i - bounds, x + j - bounds)*gker.at<float>(i, j);
					sum += pixel;
				}
			}

			//Se utiliza la funcion nth_element que obtiene el valor deseado de manera rapida, ordenando solo
			//los elementos necesarios para obtener el valor deseado
			dst.at<uchar>(y - bounds, x - bounds) = (int)round(sum);
		}
	}
	//Se iguala con la matriz de entrada src
	//src = src0;
	src = dst;

}
void thining(Mat& src, Mat ker) {
	Mat a = src;
	Mat b = src;
	Mat c;
	Mat kernel = ker;
	morfology(a, kernel, EROSION);
	morfology(b, kernel, DILATACION);
	//cv::subtract(a, b, c);
	//cv:subtract(src, c, src);
	Mat cr = restb(a,b);
	src = restb(src,cr);

}

void spur(Mat& src, Mat ker) {
	Mat x1 = src;
	Mat x3 = src;
	Mat a = src;
	Mat b = src;
	Mat kernelaux = (Mat_<int>(3, 3) << 0, 0, 0, 1, 1, 1, 0, 0, 0);
	Mat kernelaux1 = (Mat_<int>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
	Mat kernelaux2 = (Mat_<int>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
	thining(x1, kernelaux2);
	thining(x1, kernelaux2);
	thining(x1, kernelaux2);
	morfology(a,kernelaux2,DILATACION);
	morfology(b,ker,EROSION);
	x3=restb(b,a);
	morfology(x3,kernelaux1,DILATACION);
	morfology(x3,kernelaux1,DILATACION);
	morfology(x3,kernelaux1,DILATACION);
	src=logic(x1,x3,OR);

}
void skel(Mat& im){
	Mat src = im;
	Mat skeleton(src.size(), CV_8UC1, cv::Scalar(0));
	int cont = src.cols*src.rows;
	Mat temp;
	Mat eroded;
	Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	//bool done = true;
	int i = 0;
	/*while (done)
	{
		// eroding + dilating = opening
		cv::erode(src, eroded, element);
		cv::dilate(eroded, temp, element);
		cv::subtract(src, temp, temp);
		cv::bitwise_or(skeleton, temp, skeleton);
		eroded.copyTo(src);
		cout << cv::countNonZero(src) << endl;
		done = (cv::countNonZero(src) == 0);
		++i;
	}*/
	bool done;
	do
	{
		cv::morphologyEx(im, temp, cv::MORPH_OPEN, element);
		//mopen(im, element);
		cv::bitwise_not(temp, temp);
		cv::bitwise_and(im, temp, temp);
		cv::bitwise_or(skeleton, temp, skeleton);
		cv::erode(im, im, element);

		double max;
		cv::minMaxLoc(im, 0, &max);
		done = (max == 0);
	} while (!done);
	im=skeleton;
}
int main(int, char** argv)
{
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(100);
	// cargar imagen
	Mat im = imread(argv[1], 1);
	Mat kernel = (Mat_<int>(3, 3) << 1, 0, 1, 0, 1, 0, 1, 0, 1);
	Mat kernela = (Mat_<int>(3, 3) << 1,1 , 1, 1, 1, 1, 1, 1, 1);
	int morph_size = 2;
	//Mat kernela = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	//cout << kernel.at<int>(0, 0);
	clock_t time = clock();
	getEachBGR(GREEN, im);
	imwrite("grayscale.png", im, compression_params);
	//resize(640, im);
	//namedWindow("I");
	//imshow("I", im);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	clahe->apply(im, im);
	imwrite("hist.png", im, compression_params);
	morfology(im, kernel, DILATACION);

	imwrite("dilatation.png", im, compression_params);
	morfology(im, kernel, EROSION);
	imwrite("erosion.png", im, compression_params);
	estimation(im,51,3);

	Thg(im, 6);
	imwrite("escalado.png", im, compression_params);
	//kernel = np.ones((3, 3), np.uint8)
	//morphologyEx(im, im, MORPH_OPEN, kernela);
	mopen(im, kernela);
	imwrite("apertura.png", im, compression_params);
	//gaussianFilter(im, 9);
	//imwrite("gauss.png", im, compression_params);
	
	spur(im, kernel);
	imwrite("spur.png", im, compression_params);
	skel(im);
	imwrite("skel.png", im, compression_params);
	//Mat el = im.clone();
	
	//mopen(im, kernela);
	//imwrite("casifinal.png", im, compression_params);
	//cv::subtract(im,el, im);
	//im = el - im;
	//spur(im, kernel);
	//imwrite("final.png", im, compression_params);
	cout << "tiempo" << (double)(clock() - time) / CLOCKS_PER_SEC;
	//namedWindow("Imagen");
	//imshow("Imagen", im);
	//Mat out;
	//cv::subtract(im, im, out);

	waitKey();
	return 0;
}



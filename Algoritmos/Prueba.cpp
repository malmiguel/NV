//#include "Source.cpp"
#include <iostream>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<numeric>

using namespace std;
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0
#define DILATACION 0
#define EROSION 1
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

void cumhist(int hist[], int cumhist[])
{
	cumhist[0] = hist[0];

	for (int i = 1; i < 256; i++)
	{
		cumhist[i] = hist[i] + cumhist[i - 1];
	}
}

void histDisplay(int histogram[], const char* name)
{
	int hist[256];
	for (int i = 0; i < 256; i++)
	{
		hist[i] = histogram[i];
	}
	// dibular el histograma
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / 256);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

	// encontrar el maximo nivel
	int max = hist[0];
	for (int i = 1; i < 256; i++) {
		if (max < hist[i]) {
			max = hist[i];
		}
	}

	// normlizar el histograma

	for (int i = 0; i < 256; i++) {
		hist[i] = ((double)hist[i] / max)*histImage.rows;
	}


	// dibujar las intensidades
	for (int i = 0; i < 256; i++)
	{
		line(histImage, Point(bin_w*(i), hist_h),
			Point(bin_w*(i), hist_h - hist[i]),
			Scalar(0, 0, 0), 1, 8, 0);
	}

	// mostrar
	namedWindow(name, CV_WINDOW_AUTOSIZE);
	imshow(name, histImage);
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
	cout << im.cols << "," << im.rows << endl;
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
			if (x == 0 || y == 0 || x == im.cols - 1 || y == im.rows - 1) {
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
	 int midValue = (size*size - 1) / 2;
	 int bounds = (size - 1) / 2;
	 vector<int> window(size*size);

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
			 nth_element(window.begin(), window.begin() + midValue, window.end());
			 //La matriz destino es llenada con el valor de mediana del kernel
			 dst.at<uchar>(y - bounds, x - bounds) = window[midValue];
		 }
	 }
	 //Se iguala con la matriz de entrada src
	 //src = src0;
	 src = dst;

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
				im.at<uchar>(y, x) = saturate_cast<uchar>(0);
			}
			else {
				im.at<uchar>(y, x) = saturate_cast<uchar>(255);
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
void thining(Mat& src) {
	Mat kernel = (Mat_<int>(3, 3) << 1, 0, 1, 0, 1, 0, 1, 0, 1);
	morfology(src, kernel, EROSION);


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
void estimation(Mat& src, int a, int b) {
	Mat aux = src;
	medianFilter(a,aux);
	Mat ma=restar(src,aux);
	medianFilter(b, ma);
	src = ma;
}
int main(int, char** argv)
{
	// cargar imagen
	Mat im = imread(argv[1], 1);



	Mat kernel = (Mat_<int>(3, 3) << 1, 0, 1, 0, 1, 0, 1, 0, 1);
	//cout << kernel.at<int>(0, 0);
	clock_t time = clock();
	
	getEachBGR(GREEN, im);
	resize(640, im);
	namedWindow("I");
	imshow("I", im);

	//medianFilter(7, im);
	eqHist(im);
	morfology(im, kernel, DILATACION);
	morfology(im, kernel, EROSION);
	//medianFilter(30, im);
	estimation(im, 40, 3);
	//Tresh(im);
	Thg(im, 215);
	mopen(im, kernel);
	cout << "tiempo" << (double)(clock() - time) / CLOCKS_PER_SEC;

	namedWindow("Imagen");
	imshow("Imagen", im);
	//Mat out;
	//cv::subtract(im, im, out);




	waitKey();
	return 0;
}



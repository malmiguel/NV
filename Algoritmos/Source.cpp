#include "Header.hpp"

/**
 *Funcion para obtener cada uno de los canales de color en un esquema RGB
 *Tener en cuenta que OpenCV maneja un orden de canales BGR
 * @function getEachBGR
 */
void getEachBGR(int opt, Mat& src) {
	//Definimos un vector de matrices donde estaran alojados cada canal BGR
	vector<Mat> bgr_planes;
	//Separamos la imagen a color en tres imagenes en escala de grises, una por cada canal
	split( src, bgr_planes );
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

/**
 * Funcion para redimencionar una imagen a escala factor sin perder proporcionalidad
 * @function resize
 */
void resize(float factor, Mat& src) {
	
	float tempVal;
	int xr, yr;
	
	//Obteniendo valores de factor de escala para no perder proporcionalidad.
	if(src.cols == src.rows){
    		xr = factor;
    		yr = factor;
	}
	else if(src.cols > src.rows) {
		xr = factor;
		tempVal = src.rows/(src.cols/factor);
		yr = (int)round(tempVal);
	}
	else {
		yr = factor;
		tempVal = src.cols/(src.rows/factor);
		yr = (int)round(tempVal);
	}

	//Se ocupa la funcion resize de opencv con el parametro INTER_AREA, el mas adecuado para reducir imagenes.
	resize(src, src, Size(xr, yr), 0, 0, INTER_AREA);
}

/**
 *Filtro de mediana de kernel size*size, dada una matriz de entrada src
 * @function medianFilter
 */
void medianFilter(int size, Mat& src) {
 	//Valores para  el manejo del kernel
 	int midValue = (size*size-1)/2;
	int bounds = (size-1)/2;
        Vector<int> window(size*size);
 
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

/**
 *Umbrazlicacion local que utiliza las variables A y B definidas en Header.hpp
 * @function localTresh
 */
void localTresh(Mat& src) {

	int size = 9;

 	//Valores para  el manejo del kernel
 	int midValue = (size*size-1)/2;
	int bounds = (size-1)/2;
	
	Vector<double> window(size*size);
        
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
 		
 		double sum = std::accumulate(window.begin(), window.end(), 0.0);
		double mean = sum / window.size();

		double sq_sum = std::inner_product(window.begin(), window.end(), window.begin(), 0.0);
		double stdev = std::sqrt(sq_sum / window.size() - mean * mean);
		
		double thresh = A*stdev + B*mean;
		
		int value = 0;
		
		if(dst.at<uchar>(y-bounds,x-bounds) > thresh) 
			value = 255;
		else
			value = 0;
		
 		
                //nth_element(window.begin(), window.begin()+midValue, window.end());
                //La matriz destino es llenada con el valor de mediana del kernel
                //int median = average;
                
                dst.at<uchar>(y-bounds,x-bounds) = value;
            }
        }
        //Se iguala con la matriz de entrada src
        //src = src0;
        src = dst;
	
}



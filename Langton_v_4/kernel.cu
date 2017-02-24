#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <curand.h>
#include <curand_kernel.h>

__global__ void searchKrenel(int *dev_width, int *dev_Ants_Inf, int *dev_Ants_Size, int *dev_Ant_Array, int *dev_Ant_Map){
	int i = threadIdx.x;
	int x = dev_Ant_Array[*dev_Ants_Inf * i];
	int y = dev_Ant_Array[*dev_Ants_Inf * i + 1];

	curandState_t state;
	curand_init(0, i, 0, &state);

	for (int search = 0; search < *dev_Ants_Size; search++){
		if (search != i && x >= (dev_Ant_Array[search * *dev_Ants_Inf] - 2) && x <= dev_Ant_Array[search * *dev_Ants_Inf] 
			&& y / (*dev_width) <= dev_Ant_Array[search* *dev_Ants_Inf + 1] / (*dev_width) + 2 && y >= dev_Ant_Array[search* *dev_Ants_Inf + 1]){
			printf("\n Mrowka %i znalazla %i, x:%i y:%i k:%i", i, search, dev_Ant_Array[i], dev_Ant_Array[i + 1] / (*dev_width), dev_Ant_Array[i + 2]);
			// czyszcenie poprzedniej pozycji
				dev_Ant_Map[y + x] = 0;
			// pozycja x
				dev_Ant_Array[*dev_Ants_Inf * i] = 1 + (curand(&state) % (*dev_width - 2));
			// pozycja y
				dev_Ant_Array[*dev_Ants_Inf * i + 1] = *dev_width * (1 + (curand(&state) % (*dev_width - 2)));
			// kierunek
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 1 + (curand(&state) % 4);
			//wypisanie
				printf(" x2:%i y2:%i k2:%i", dev_Ant_Array[i], dev_Ant_Array[i + 1] / (*dev_width), dev_Ant_Array[i + 2]);
				dev_Ant_Map[dev_Ant_Array[*dev_Ants_Inf *i + 1] + dev_Ant_Array[*dev_Ants_Inf *i]] = dev_Ant_Array[*dev_Ants_Inf *i + 2];
		}
		__syncthreads();	//synchronizacja w¹tków
	}
}
__global__ void findFailsKernel(int *dev_width, int *dev_Ants_Inf, int *dev_Ants_Size, int *dev_Ant_Array, int *dev_Ant_Map){
	int i = threadIdx.x;

	int x = dev_Ant_Array[*dev_Ants_Inf * i];
	int y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
	int k = dev_Ant_Array[*dev_Ants_Inf * i + 2];

	curandState_t state;
	curand_init(1, i, 0, &state);
	if (x >= *dev_width || x < 0|| y < 0 || y >= *dev_width * *dev_width || k > 4 || k < 0){
		printf("\nPoprawiam mrowke %i\n", i);
		// czyszcenie nieprawid³owej pozycji
			dev_Ant_Map[y + x] = 0;
		// usuwanie b³êdów
			if (x >= *dev_width || x < 0){
				dev_Ant_Array[*dev_Ants_Inf * i] = 1 + (curand(&state) % (*dev_width - 2));
				x = dev_Ant_Array[*dev_Ants_Inf * i];
			}
			__syncthreads();	//synchronizacja w¹tków
			if (y >= *dev_width * *dev_width || y < 0){
				dev_Ant_Array[*dev_Ants_Inf * i + 1] = *dev_width * (1 + (curand(&state) % (*dev_width - 2)));
				y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
			}
			__syncthreads();	//synchronizacja w¹tków
			if (k > 4 || k < 0){
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 1 + (curand(&state) % 4);
				k = dev_Ant_Array[*dev_Ants_Inf * i + 2];
			}
			__syncthreads();	//synchronizacja w¹tków
		// zapisanie do nowej prawid³owej pozycji
			dev_Ant_Map[dev_Ant_Array[*dev_Ants_Inf * i + 1] + dev_Ant_Array[*dev_Ants_Inf * i]] = dev_Ant_Array[*dev_Ants_Inf * i + 2];
	}
	__syncthreads();	//synchronizacja w¹tków		
}
__global__ void alghoritmKernel(int *dev_width, int *dev_Ants_Inf, int *dev_Ant_Array, int *dev_Ant_Map, int *dev_Col_Map){
	int i = threadIdx.x;
	//for (int steps = 0; steps < 100; steps++){ //obliczanie danej liczby kroków na raz przed przes³¹niem na cpu

		int x = dev_Ant_Array[*dev_Ants_Inf * i];
		int y = dev_Ant_Array[*dev_Ants_Inf * i + 1];

		if (dev_Col_Map[y + x] == 0){
			dev_Col_Map[y + x] = 1;

			if (dev_Ant_Map[y + x] == 1){ // obrócona do góry
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 2; // zmian kierunku
				dev_Ant_Map[y + x] = 0;

				if ( x == *dev_width - 1){
					dev_Ant_Array[*dev_Ants_Inf * i] = 0;	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = dev_Ant_Array[*dev_Ants_Inf * i + 1]; // zmian poz y
				}else{
					dev_Ant_Array[*dev_Ants_Inf * i]		= dev_Ant_Array[*dev_Ants_Inf * i] + 1;	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1]	= dev_Ant_Array[*dev_Ants_Inf * i + 1]; // zmian poz y
				}

				x = dev_Ant_Array[*dev_Ants_Inf * i];
				y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
				dev_Ant_Map[y + x] = dev_Ant_Array[*dev_Ants_Inf * i + 2];

			}else if (dev_Ant_Map[y + x] == 2){ // obrócona w prawo
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 3; // zmian kierunku
				dev_Ant_Map[y + x] = 0;
				if (y/(*dev_width) == *dev_width - 1){
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i];	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = 0; // zmian poz y
				}
				else{
					dev_Ant_Array[*dev_Ants_Inf * i]		= dev_Ant_Array[*dev_Ants_Inf * i];	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1]	= ((dev_Ant_Array[*dev_Ants_Inf * i + 1]/ *dev_width) + 1)*(*dev_width) ;// zmian poz y
				}

				x = dev_Ant_Array[*dev_Ants_Inf * i];
				y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
				dev_Ant_Map[y + x] = dev_Ant_Array[*dev_Ants_Inf * i + 2];

			}else if (dev_Ant_Map[y + x] == 3){ // obrócona w dó³
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 4; // zmian kierunku
				dev_Ant_Map[y + x] = 0;
				if (x == 0){
					dev_Ant_Array[*dev_Ants_Inf * i] = *dev_width-1;	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = dev_Ant_Array[*dev_Ants_Inf * i + 1];// zmian poz y
				}
				else{
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i] - 1;	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = dev_Ant_Array[*dev_Ants_Inf * i + 1];// zmian poz y
				}

				x = dev_Ant_Array[*dev_Ants_Inf * i];
				y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
				dev_Ant_Map[y + x] = dev_Ant_Array[*dev_Ants_Inf * i + 2];

			}else if (dev_Ant_Map[y + x] == 4){ // obrócona w lewo
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 1; // zmian kierunku
				dev_Ant_Map[y + x] = 0;
				if (y/(*dev_width) == 0){
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i];	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = (*dev_width - 1)*(*dev_width);// zmian poz y
				}
				else{
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i];	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = ((dev_Ant_Array[*dev_Ants_Inf * i + 1] / *dev_width) - 1)*(*dev_width);// zmian poz y
				}

				x = dev_Ant_Array[*dev_Ants_Inf * i];
				y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
				dev_Ant_Map[y + x] = dev_Ant_Array[*dev_Ants_Inf * i + 2];

			}
		}else if (dev_Col_Map[y + x] == 1){
			dev_Col_Map[y + x] = 0;
		
			if (dev_Ant_Map[y + x] == 1){ // obrócona do góry
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 4; // zmian kierunku
				dev_Ant_Map[y + x] = 0;
				if (x == 0){
					dev_Ant_Array[*dev_Ants_Inf * i] = *dev_width - 1 ;	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = dev_Ant_Array[*dev_Ants_Inf * i + 1];// zmian poz y
				}else{
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i] - 1;	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = dev_Ant_Array[*dev_Ants_Inf * i + 1];// zmian poz y
				}
			
				x = dev_Ant_Array[*dev_Ants_Inf * i];
				y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
				dev_Ant_Map[y + x] = dev_Ant_Array[*dev_Ants_Inf * i + 2];

			}else if (dev_Ant_Map[y + x] == 2){ // obrócona w prawo
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 1; // zmian kierunku
				dev_Ant_Map[y + x] = 0;
				if (y == 0){
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i];	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = ((*dev_width) - 1)*(*dev_width);// zmian poz y
				}else {
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i];	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = ((dev_Ant_Array[*dev_Ants_Inf * i + 1] / *dev_width) - 1)*(*dev_width);// zmian poz y

				}
			
				x = dev_Ant_Array[*dev_Ants_Inf * i];
				y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
				dev_Ant_Map[y + x] = dev_Ant_Array[*dev_Ants_Inf * i + 2];

			}else if (dev_Ant_Map[y + x] == 3){ // obrócona w dó³
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 2; // zmian kierunku
				dev_Ant_Map[y + x] = 0;
				if (x == *dev_width - 1){
					dev_Ant_Array[*dev_Ants_Inf * i] = 0;	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = dev_Ant_Array[*dev_Ants_Inf * i + 1];// zmian poz y
				}else{
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i] + 1;	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = dev_Ant_Array[*dev_Ants_Inf * i + 1];// zmian poz y
				}

				x = dev_Ant_Array[*dev_Ants_Inf * i];
				y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
				dev_Ant_Map[y + x] = dev_Ant_Array[*dev_Ants_Inf * i + 2];

			}else if (dev_Ant_Map[y + x] == 4){ // obrócona w lewo
				dev_Ant_Array[*dev_Ants_Inf * i + 2] = 3; // zmian kierunku
				dev_Ant_Map[y + x] = 0;
				if (y/(*dev_width) == *dev_width - 1){
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i];	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = 0;// zmian poz y
				}else{
					dev_Ant_Array[*dev_Ants_Inf * i] = dev_Ant_Array[*dev_Ants_Inf * i];	// zmian poz x
					dev_Ant_Array[*dev_Ants_Inf * i + 1] = ((dev_Ant_Array[*dev_Ants_Inf * i + 1] / *dev_width) + 1)*(*dev_width);// zmian poz y
				}
			
				x = dev_Ant_Array[*dev_Ants_Inf * i];
				y = dev_Ant_Array[*dev_Ants_Inf * i + 1];
				dev_Ant_Map[y + x] = dev_Ant_Array[*dev_Ants_Inf * i + 2];

			}
		}
		// synchronizacja
		__syncthreads();
	//}
}

void clear_Col_Map(int width, int Col_Map[64 * 64]){
	for (int i = 0; i < width*width; i++){
		if (i < width || i%width == 0 || i%width == width - 1 || i > width*(width - 1)){
			//Col_Map[i] = 1;	// tworze ramkê nie do przejœcia
			Col_Map[i] = 0;
		}
		else{
			Col_Map[i] = 0;
		}
	}
};
void clear_Map(int width, int Map[64 * 64]){
	for (int i = 0; i < width*width; i++){
		Map[i] = 0;
	}
};
void createAnts(int width, int Ants_Inf, int Ants_Size, int Ant_Array[100 * 3], int Ant_Map[64 * 64]){
	srand(time(NULL));
	for (int i = 0; i < Ants_Inf*Ants_Size; i += Ants_Inf){
		// pozycja x
		Ant_Array[i] = 1 + rand() % (width - 2);
		// pozycja y
		Ant_Array[i + 1] = width * (1 + rand() % (width - 2));
		// kierunek
		Ant_Array[i + 2] = 1 + rand() % 4;
		//wypisanie
		Ant_Map[Ant_Array[i + 1] + Ant_Array[i]] = Ant_Array[i + 2];
	}
}
void show_Col_Map(int width, int Col_Map[64 * 64], int Ant_Map[64 * 64]){
	printf("\n  ");
	for (int i = 0; i < width*width; i++){
		// wpisywanie numeru wiersza
		if (i%width == 0 && i < width * 10){
			printf("\n   %i ", i / width);
		}
		else if (i%width == 0 && i >= width * 10){
			printf("\n  %i ", i / width);
		}
		// wypisywanie komorek mapy
		if (Ant_Map[i] == 0){
			if (Col_Map[i] == 0){
				printf(" ");
			}
			else{
				printf("#");
			}
		}
		else{
			if (Ant_Map[i] == 1){
				printf("^");
			}
			if (Ant_Map[i] == 2){
				printf(">");
			}
			if (Ant_Map[i] == 3){
				printf("#");
			}
			if (Ant_Map[i] == 4){
				printf("<");
			}
		}

	}
}
void show_2_Map(int width, int Col_Map[64 * 64], int Ant_Map[64 * 64]){
	printf("\n\t Mapa Zycia %i x %i \t\t\t\t\t\t\t\t Mapa Mrowek %i x %i", width, width, width, width);
	printf("\n"); printf("\n     ");
	for (int k = 0; k < width; k++){
		printf(" _");
	}
	printf("\t\t     ");
	for (int k = 0; k < width; k++){
		printf(" _");
	}

	for (int i = 0; i < width*width; i++){
		// wpisywanie numeru wiersza
		if (i%width == 0 && i < width * 10){
			printf("|\n   %i ", i / width);
		}
		else if (i%width == 0 && i >= width * 10){
			printf("|\n  %i ", i / width);
		}
		// wypisywanie komorek mapy
		if (Col_Map[i] == 0){
			printf("|_");
		}
		else{
			printf("|#");
		}
		// wypisywanie mapy mrowek
		if (i%width == width - 1){
			printf("|\t\t");
			if (i%width == (width - 1) && i < width * 10){
				printf("   %i ", (i - width + 1) / width);
			}
			else if (i%width == (width - 1) && i >= width * 10){
				printf("  %i ", (i - width + 1) / width);
			}
			for (int k = 0; k < width; k++){
				if (Ant_Map[(i - width + 1) + k] == 0){
					printf("|_");
				}
				else{
					if (Ant_Map[(i - width + 1) + k] == 1){
						printf("|^");
					}
					if (Ant_Map[(i - width + 1) + k] == 2){
						printf("|>");
					}
					if (Ant_Map[(i - width + 1) + k] == 3){
						printf("|#");
					}
					if (Ant_Map[(i - width + 1) + k] == 4){
						printf("|<");
					}
				}
			}
		}

	}
}
void show_Ants(int width, int Ants_Inf, int Ants_Size, int Ant_Array[100 * 3]){
	printf("\n\n\n\t | Ant\t| x\t| y\t| kier\t|\n");
	printf("\t --------------------------------\n \t ");
	for (int i = 0; i < Ants_Size*Ants_Inf; i += Ants_Inf){
		printf("| %i\t| %i\t| %i\t| %i\t", i / Ants_Inf, Ant_Array[i], Ant_Array[i + 1] / width, Ant_Array[i + 2]);
		printf("|\n\t --------------------------------\n\t ");
	}
}

int main()
{
	const int Ants_Size = 16; // iloœæ mrówek
	const int Ants_Inf = 3; // iloœæ danych o mrówce
	const int width = 64;

	int Col_Map[width*width];
	int Ant_Map[width*width];
	int Ant_Array[Ants_Size*Ants_Inf];

	int *dev_Ants_Size; // iloœæ mrówek
	int *dev_Ants_Inf; // iloœæ danych o mrówce
	int *dev_width;

	int *dev_Col_Map;
	int *dev_Ant_Map;
	int *dev_Ant_Array;

//Alokowanie pamiêci na GPU
	cudaMalloc((void**)&dev_Ants_Size, sizeof(int));
	cudaMalloc((void**)&dev_Ants_Inf, sizeof(int));
	cudaMalloc((void**)&dev_width, sizeof(int));

	cudaMalloc((void**)&dev_Col_Map, width*width * sizeof(int));
	cudaMalloc((void**)&dev_Ant_Map, width*width * sizeof(int));
	cudaMalloc((void**)&dev_Ant_Array, Ants_Size * Ants_Inf * sizeof(int));

//Tworzenie danych poczatkowych
	clear_Col_Map(width, Col_Map);
	clear_Map(width, Ant_Map);
	createAnts(width, Ants_Inf, Ants_Size, Ant_Array, Ant_Map);

//Kopiowanie danych z CPU na GPU
	cudaMemcpy(dev_Ants_Size, &Ants_Size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Ants_Inf, &Ants_Inf, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_width, &width, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_Col_Map, Col_Map, width * width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Ant_Map, Ant_Map, width * width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Ant_Array, Ant_Array, Ants_Size * Ants_Inf * sizeof(int), cudaMemcpyHostToDevice);
	
	int step = 0;
// Nieskoñczoncza pêtla do algorytmu
	while (step < 1000){
		//	system("cls"); // czyszczenie konsoli
			printf("\n\n  Langthon Ant's Algorithm \tStep: %i\n", step);
		// Poszukiwnie b³êdów
			findFailsKernel << < 1, Ants_Size >> >(dev_width, dev_Ants_Inf, dev_Ants_Size, dev_Ant_Array, dev_Ant_Map);
		// Przeszukanie
			searchKrenel	<<< 1, Ants_Size >>>(dev_width, dev_Ants_Inf, dev_Ants_Size, dev_Ant_Array, dev_Ant_Map); //wyszykiwanie zbli¿eñ
		// Poszukiwnie b³êdów
			findFailsKernel << < 1, Ants_Size >> >(dev_width, dev_Ants_Inf, dev_Ants_Size, dev_Ant_Array, dev_Ant_Map);
		// Algorytm
			alghoritmKernel <<< 1, Ants_Size >>>(dev_width, dev_Ants_Inf, dev_Ant_Array, dev_Ant_Map, dev_Col_Map); //poruszenie mrówek
		//Kopiowanie danych z GPU na CPU
			cudaMemcpy(Col_Map, dev_Col_Map, width * width * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(Ant_Map, dev_Ant_Map, width * width * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(Ant_Array, dev_Ant_Array, Ants_Size * Ants_Inf * sizeof(int), cudaMemcpyDeviceToHost);
		//wyrysowanie
			//show_Col_Map(width, Col_Map, Ant_Map);			// wyrysowanie jednej mapy zawierajacej informacje o komórkach i pozycji mrówek
			show_2_Map(width, Col_Map, Ant_Map);				// wyrysowanie mapy ¿ywych/martwych komórek i mapy istniejacych mrówek
			show_Ants(width, Ants_Inf, Ants_Size, Ant_Array);	// wypianie danych o mrówkach

		step++;
	//getchar();  // pauza po kroku
	}

	getchar(); getchar();

	cudaFree(dev_Col_Map);
	cudaFree(dev_Ant_Map);
	cudaFree(dev_Ant_Array);
	return 0;
}

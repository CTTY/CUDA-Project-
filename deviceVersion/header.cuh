#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
#include <string>

#define Fdopp 10
#define d 1
/* Pi: M_PI
e: M_E
Fdopp: (10)
theta/lambda: (Pi/4)
d: 1*/
#define N 2
#define M 2
#define L M*N
#define SIZE N*M*L
/*Cube: NxMxL*/
/*Slice: MxN*/
#define SINTL 1



void createCubeOnHost(cuComplex* h_cube);

void QRsolver(cuComplex* A, int MN,
	cuComplex* B,
	cuComplex* X);

int batchFFT(cufftComplex* inputData,cufftComplex* outputData, int BATCH, int DATASIZE);

__global__ void steeringVector(cuComplex* t);

__global__ void vectorizeSlices(cuComplex* d_cube, cuComplex* d_y);

__global__ void computeS(cuComplex* d_s, cuComplex* d_y);

__global__ void getConj(cuComplex* vector);

void QRhelper(cuComplex* d_s, cuComplex* t_conj, cuComplex* d_u);
	__global__ void initData(cuComplex* d_s, cuComplex** temp_s, cuComplex* d_u, cuComplex** temp_u);
	__global__ void copyData(cuComplex** temp_u, cuComplex* d_u);
__global__ void computeTU(cuComplex* d_t, cuComplex* d_u, cuComplex* tempRes);

__global__ void computeH(cuComplex* d_u, cuComplex* tempRes);

__global__ void computeZ(cuComplex* d_u, cuComplex* d_y, cuComplex* d_z);

void transpose(cuComplex* odata, cuComplex *idata, int m, int n, int lda, int ldc);

void subtraction(cuComplex* C, cuComplex *A, cuComplex* B, int m, int n, int lda, int ldc);

void matrixMulC(cuComplex* d_c, cuComplex* d_y1, cuComplex* d_y2,
	int m, int n, int k, 
	int lda, int ldb, int ldc, 
	cublasOperation_t opA, cublasOperation_t opB);

void batchCholeskySolver();

void choleskySolver(cuComplex* d_X, cuComplex* d_A, cuComplex* d_B,
	int lda, int ldb, int m);

void preprocessing(cuComplex* d_u, cuComplex* d_s, cuComplex* d_y, cuComplex* d_t);

void matrixScal(cuComplex* d_a, cuComplex* d_scaler, int n);

void matrixBatchMul(cuComplex* d_c, cuComplex* d_a, cuComplex* d_b);

void printMatrix(int m, int n, const cuComplex* A, int lda, const char* name);
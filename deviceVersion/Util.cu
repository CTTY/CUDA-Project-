#include "header.cuh"

/*Create a data cube row wisely*/
void createCubeOnHost(cuComplex* h_cube) {
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k < N; k++) {
				h_cube[k + j*N + i*M*N] = make_cuComplex(rand(), rand());
			}
		}
	}

}

/*Create steering vector_3*/
/*NxM*/
__global__ void steeringVector(cuComplex* t) {
	int m = threadIdx.x;
	int n = blockIdx.x;
	if (m < M && n < N) {
		cuComplex Fd = make_cuComplex(cosf(n*Fdopp), -sinf(n*Fdopp));
		cuComplex A = make_cuComplex(cosf(m*d*SINTL), -sinf(m*d*SINTL));
		t[n*M + m] = cuCmulf(Fd, A);
	}
}

/*Vectorize Slices _4  Deprecated, using transposition now*/
/*Lx(MxN)*/
__global__ void vectorizeSlices(cuComplex* d_cube, cuComplex* d_y) {
	int sliceNum = blockIdx.x;
	int m = threadIdx.x;
	int n = threadIdx.y;
	int idx_cube = sliceNum*M*N + m*N + n;
	int idx_slice = sliceNum*M*N + n*M + m;

	d_y[idx_slice] = d_cube[idx_cube];
}

/* Matrix Multiplication for cuComplex */
/* C = opA(A) * opB(B)  */
void matrixMulC(cuComplex* d_c, cuComplex* d_y1, cuComplex* d_y2, int m, int n, int k, int lda, int ldb, int ldc, cublasOperation_t opA, cublasOperation_t opB) {
	//cudaError_t cudaStat; // cudaMalloc status 
	cublasStatus_t stat; // CUBLAS functions status 
	cublasHandle_t handle; // CUBLAS context 

	stat = cublasCreate(&handle); // initialize CUBLAS context
	assert(stat == CUBLAS_STATUS_SUCCESS);
	cuComplex alpha = make_cuComplex(1, 0);
	cuComplex beta = make_cuComplex(0, 0);

	stat = cublasCgemm(handle, opA, opB, m, n, k,
		&alpha, d_y1, lda, d_y2, ldb, &beta, d_c, ldc);
	assert(stat == CUBLAS_STATUS_SUCCESS);

	stat = cublasDestroy(handle); // destroy CUBLAS context
	assert(stat == CUBLAS_STATUS_SUCCESS);
}

void matrixBatchMul(cuComplex* d_c, cuComplex* d_a, cuComplex* d_b) {
	cublasStatus_t stat; // CUBLAS functions status 
	cublasHandle_t handle; // CUBLAS context 

	stat = cublasCreate(&handle); // initialize CUBLAS context
	assert(stat == CUBLAS_STATUS_SUCCESS);

	cuComplex alpha = make_cuComplex(1, 0);
	cuComplex beta = make_cuComplex(1, 0);

	stat = cublasCgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1, M*N, &alpha, d_a, M*N, M*N, d_b, M*N, M*N, &beta, d_c, L, 1, L);
	assert(stat == CUBLAS_STATUS_SUCCESS);

	stat = cublasDestroy(handle); // destroy CUBLAS context
	assert(stat == CUBLAS_STATUS_SUCCESS);

}

void matrixScal(cuComplex* d_a, cuComplex* d_scaler, int n) {
	cublasStatus_t stat; // CUBLAS functions status 
	cublasHandle_t handle; // CUBLAS context 

	stat = cublasCreate(&handle); // initialize CUBLAS context
	assert(stat == CUBLAS_STATUS_SUCCESS);

	stat = cublasCscal(handle, n, d_scaler, d_a, M*N);
	assert(stat == CUBLAS_STATUS_SUCCESS);

	stat = cublasDestroy(handle); // destroy CUBLAS context
	assert(stat == CUBLAS_STATUS_SUCCESS);
}

/*Get conjugate of a vector*/
/*1xMN*/
__global__ void getConj(cuComplex* vector) {
	int i = threadIdx.x;
	vector[i] = make_cuComplex(cuCrealf(vector[i]), -cuCimagf(vector[i]));
}

/*Compute Z _8*/
/*LxMN*/
__global__ void computeTU(cuComplex* d_t, cuComplex* d_u, cuComplex* tempRes) {
	int tid = threadIdx.x;
	int slice = blockIdx.x;
	tempRes[slice] = make_cuComplex(0, 0);
	cuComplex temp = cuCmulf(d_t[slice*M*N + tid], d_u[slice*M*N + tid]);
	tempRes[slice] = cuCaddf(tempRes[slice], temp);
}

/*LxMN*/
__global__ void computeH(cuComplex* d_u, cuComplex* tempRes) {
	int tid = threadIdx.x;
	int sliceNum = blockIdx.x;
	d_u[sliceNum*M*N + tid] = cuCdivf(d_u[sliceNum*M*N + tid], tempRes[sliceNum]); //compute h and store it in d_u
}

/*LxMN*/
__global__ void computeZ(cuComplex* d_u, cuComplex* d_y, cuComplex* d_z) {
	int tid = threadIdx.x;
	int slice = blockIdx.x;
	d_z[slice] = make_cuComplex(0, 0);

	cuComplex temp = cuCmulf(d_u[slice*M*N + tid], d_y[slice*M*N + tid]);
	d_z[slice] = cuCaddf(d_z[slice], temp);
}

void printMatrix(int m, int n, const cuComplex* A, int lda, const char* name)
{
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			cuComplex Areg = A[row + col*lda];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg.x);
		}
	}
}
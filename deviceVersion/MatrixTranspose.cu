#include "header.cuh"

/* C = A^T*/
void transpose(cuComplex* C, cuComplex *A, int m, int n, int lda, int ldc)
{
	cublasStatus_t stat; // CUBLAS functions status 
	cublasHandle_t handle; // CUBLAS context 

	stat = cublasCreate(&handle); // initialize CUBLAS context

	cuComplex alpha = make_cuComplex(1, 0);
	cuComplex beta = make_cuComplex(0, 0);

	cuComplex* temp;
	cudaMalloc((void**)&temp, sizeof(cuComplex)*SIZE);
	int ldb = M*L;
	stat = cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, lda, &beta, temp, ldb, C, ldc);
	assert(stat == CUBLAS_STATUS_SUCCESS);

	cublasDestroy(handle); // destroy CUBLAS context
	if (temp) cudaFree(temp);
}

/*C = A - B*/
void subtraction(cuComplex* C, cuComplex *A, cuComplex* B, int m, int n, int lda, int ldc) { 
	cublasStatus_t stat; // CUBLAS functions status 
	cublasHandle_t handle; // CUBLAS context 

	stat = cublasCreate(&handle); // initialize CUBLAS context

	cuComplex alpha = make_cuComplex(1, 0);
	cuComplex beta = make_cuComplex(-1, 0);
	stat = cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, A, lda, &beta, B, lda, C, ldc);
	assert(stat == CUBLAS_STATUS_SUCCESS);

	cublasDestroy(handle); // destroy CUBLAS context
}

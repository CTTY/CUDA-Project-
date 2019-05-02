#include "header.cuh"


/* Solve linear equation AX = B with Cholesky factorization */
void choleskySolver(cuComplex* d_X, cuComplex* d_A, cuComplex* d_B,
	int lda, int ldb, int m) {
	cusolverDnHandle_t handle = NULL;
	cudaStream_t stream = NULL;
	

	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;

	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	/*const int lda = 3;
	const int ldb = 3;
	const int m = 3;*/
	/* Create handle*/
	status = cusolverDnCreate(&handle);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cudaStat1);

	status = cusolverDnSetStream(handle, stream);
	assert(status == CUSOLVER_STATUS_SUCCESS);
	/*
	Initialize test matrices
	cuComplex A0[lda*m]; // = { 1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0 };
	A0[0] = make_cuComplex(1, 0);
	A0[1] = make_cuComplex(2, 0);
	A0[2] = make_cuComplex(3, 0);
	A0[3] = make_cuComplex(2, 0);
	A0[4] = make_cuComplex(5, 0);
	A0[5] = make_cuComplex(5, 0);
	A0[6] = make_cuComplex(3, 0);
	A0[7] = make_cuComplex(5, 0);
	A0[8] = make_cuComplex(12, 0);

	cuComplex B0[m] = { 1.0, 1.0, 1.0 };
	cuComplex X0[m]; // X0 = A0\B0 

	cuComplex L0[lda*m]; // cholesky factor of A0 

	printf("A0 = (matlab base-1)\n");
	printMatrix(m, m, A0, lda, "A0");
	printf("=====\n");

	printf("B0 = (matlab base-1)\n");
	printMatrix(m, 1, B0, ldb, "B0");
	printf("=====\n");
	*/

	//cuComplex* d_A = NULL;
	//cuComplex* d_B = NULL;
	int* d_info = NULL;

	/* Copy data to device*/
	/*
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(cuComplex)*lda*m);
	cudaStat2 = cudaMalloc((void**)&d_B, sizeof(cuComplex)*m);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);

	cudaStat1 = cudaMemcpy(d_A, A0, sizeof(cuComplex)*lda*m, cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_B, B0, sizeof(cuComplex)*m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	*/

	cudaStat1 = cudaMalloc((void**)&d_info, sizeof(int));
	assert(cudaSuccess == cudaStat1);

	int Lwork = 0;
	/* Calculate the size of workspace*/
	status = cusolverDnCpotrf_bufferSize(handle,
		uplo,
		m,
		d_A,
		lda,
		&Lwork);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);

	cuComplex* workSpace;
	cudaStat1 = cudaMalloc(&workSpace, sizeof(cuComplex)*Lwork);
	assert(cudaSuccess == cudaStat1);

	/* Cholesky factorization*/
	status = cusolverDnCpotrf(handle,
		uplo,
		m,
		d_A,
		lda,
		workSpace,
		Lwork,
		d_info);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);

	
	/*cudaStat2 = cudaMemcpy(L0, d_A, sizeof(cuComplex) * lda * m, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);

	printf("L = (matlab base-1), upper triangle is don't care \n");
	printMatrix(m, m, L0, lda, "L0");
	printf("=====\n"); */

	/* Solve equation*/
	status = cusolverDnCpotrs(handle,
		uplo,
		m,
		1,
		d_A,
		lda,
		d_B,
		ldb,
		d_info);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);


	cudaStat2 = cudaMemcpy(d_X, d_B, sizeof(cuComplex)*m, cudaMemcpyDeviceToDevice);
	
	assert(cudaSuccess == cudaStat2);
	cudaDeviceSynchronize();
	/*
	cuComplex h_X[3];
	cudaStat1 = cudaMemcpy(h_X, d_X, sizeof(cuComplex)*m, cudaMemcpyDeviceToHost);

	printMatrix(m, 1, h_X, ldb, "X");*/


	/*
	printf("X0 = (matlab base-1)\n");
	printMatrix(m, 1, d_X, ldb, "X0");
	printf("=====\n");
	*/
	/* free resources */
	//if (d_Aarray) cudaFree(d_Aarray);
	//if (d_Barray) cudaFree(d_Barray);
	//if (d_infoArray) cudaFree(d_infoArray);

	if (handle) cusolverDnDestroy(handle);
	if (stream) cudaStreamDestroy(stream);

	return;
}
#include "header.cuh"

class TestCreateCubeOnHost_1 {
public: 
	TestCreateCubeOnHost_1(cuComplex* h_cube) {
		createCubeOnHost(h_cube);
	}
};

class TestFFTCube_2 {
public:
	TestFFTCube_2(cufftComplex* inputData, cufftComplex* outputData, int BATCH, int DATASIZE) {
		batchFFT(inputData, outputData, BATCH, DATASIZE);
	}
};

class TestSteeringVector_3 {
public:
	TestSteeringVector_3(cuComplex* d_t) {
		steeringVector << <N, M >> > (d_t);
	}
};

class TestVectorizeSlices_4 {
public:
	TestVectorizeSlices_4(cuComplex* d_cube, cuComplex* d_y) {
		transpose(d_y, d_cube, M*L, N, N, M*L);
		//dim3 threads(M, N);
		//vectorizeSlices << <L, threads >> > (d_cube, d_y);
	}
};

class TestComputeS_5 {
public:
	TestComputeS_5(cuComplex* d_s, cuComplex* d_y) {
		/* Y * Y^H */
		matrixMulC(d_s, d_y, d_y, M*N, M*N, L, M*N, M*N, M*N, CUBLAS_OP_N, CUBLAS_OP_C);
	}
};

class TestConj_6 {
public:
	TestConj_6(cuComplex* d_t) {
		getConj << <1, M*N >> > (d_t);
	}
};

class TestQR_7 {
public:
	TestQR_7(cuComplex* d_s, int MN,
		cuComplex* t_conj,
		cuComplex* d_u) {
		choleskySolver(d_u, d_s, t_conj, M*N, M*N, M*N);
	}
};

class TestComputeTU_8 {
public:
	TestComputeTU_8(cuComplex* d_t, cuComplex* d_u, cuComplex* d_tempRes) {
		matrixMulC(d_tempRes, d_t, d_u, 1, L, M*N, M*N, M*N, 1, CUBLAS_OP_C, CUBLAS_OP_N);

	}
};

int main() {	

	std::ofstream outFile;
	std::string fileName = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(L) + ".txt";
	outFile.open(fileName);
	outFile << "Test begin!\n";
	outFile << "Test with cube size: M: "<< M <<", N: " << N <<", L: "<< L <<std::endl;
	cudaEvent_t start0, stop0;
	float elapsedTime0;
	cudaEventCreate(&start0); //Timing begins
	cudaEventCreate(&stop0);
	cudaEventRecord(start0, 0);

	/* Test 1: Create the cube*/
	cudaEvent_t start, stop;
	float elapsedTime1;
	cudaEventCreate(&start); //Timing begins
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuComplex *h_cube = new cuComplex[SIZE];

	TestCreateCubeOnHost_1* test1 = new TestCreateCubeOnHost_1(h_cube);
	delete test1;
	cuComplex* d_cube;
	cudaMalloc((void**)&d_cube, sizeof(cuComplex)*SIZE);

	cudaEventRecord(stop, 0);             //Timing ends
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime1, start, stop);
	outFile << "Create the cube: " << elapsedTime1 << std::endl;
	/* Test 2: Test FFT*/
	float elapsedTime2;
	cudaEventRecord(start, 0);

	TestFFTCube_2* test2 = new TestFFTCube_2(h_cube, d_cube, M*L,N);
	delete test2;
	delete[] h_cube;

	cudaEventRecord(stop, 0);             //Timing ends
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime2, start, stop);
	outFile << "FFT: " << elapsedTime2 << std::endl;

	/* Test 3: Test steering vector t*/
	float elapsedTime3;
	cudaEventRecord(start, 0);

	cuComplex *d_t;
	cudaMalloc((void**)&d_t, sizeof(cuComplex)*N*M);

	TestSteeringVector_3 *test3= new TestSteeringVector_3(d_t);
	delete test3;

	cudaEventRecord(stop, 0);             //Timing ends
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime3, start, stop);
	outFile << "Create the steering vector: " << elapsedTime3 << std::endl;

	/* Test 4: Test vectorizing slices*/
	float elapsedTime4;
	cudaEventRecord(start, 0);

	cuComplex* d_y;
	cudaMalloc((void**)&d_y, sizeof(cuComplex)*SIZE);

	TestVectorizeSlices_4 *test4 = new TestVectorizeSlices_4(d_cube, d_y);
	delete test4;
	if (d_cube) cudaFree(d_cube);	//free cube

	cudaEventRecord(stop, 0);             //Timing ends
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime4, start, stop);
	outFile << "Vectorizing slices: " << elapsedTime4 << std::endl;

	/* Test 5: Compute convariance matrix for every slice*/
	float elapsedTime5;
	cudaEventRecord(start, 0);

	cuComplex* d_s;
	cudaMalloc((void**)&d_s, sizeof(cuComplex)*M*N*M*N);
	
	
	TestComputeS_5 *test5 = new TestComputeS_5(d_s, d_y);
	delete test5;

	cudaEventRecord(stop, 0);             //Timing ends
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime5, start, stop);
	outFile << "Compute convariance matrix: " << elapsedTime5 << std::endl;

	/* Test 6: Get conjugate of t*/
	float elapsedTime6;
	cudaEventRecord(start, 0);

	TestConj_6 *test6 = new TestConj_6(d_t);
	delete test6;

	cudaEventRecord(stop, 0);             //Timing ends
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime6, start, stop);
	outFile << "Get conjugate of t: " << elapsedTime6 << std::endl;

	/* Test 7: Call Cholesky solver to solve u*/
	float elapsedTime7;
	cudaEventRecord(start, 0);

	cuComplex* d_u;
	cudaMalloc((void**)&d_u, sizeof(cuComplex)*SIZE);

	TestQR_7 *test7 = new TestQR_7(d_s, M*N, d_t, d_u);	
	delete test7;

	cudaEventRecord(stop, 0);             //Timing ends
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime7, start, stop);
	outFile << "Cholesky solver: " << elapsedTime7 << std::endl;

	/* Test 8: Compute t^H * (u*) Note: t is already (t*) in the last step */
	float elapsedTime8;
	//cudaEventCreate(&start); //Timing begins
	//cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuComplex* d_tempRes;
	cudaMalloc((void**)&d_tempRes, sizeof(cuComplex)*L);

	cuComplex* d_z;
	cudaMalloc((void**)&d_z, sizeof(cuComplex)*L);

	TestComputeTU_8 *test8 = new TestComputeTU_8(d_t, d_u, d_tempRes);
	delete test8;

	computeH << <L, M*N >> > (d_u, d_tempRes);	//compute h and store it in d_u
	//matrixScal(d_u, d_tempRes, L);
	//computeZ << <L, M*N >> > (d_u, d_y, d_z);
	matrixBatchMul(d_z, d_u, d_y);
	

	cudaEventRecord(stop, 0);             //Timing ends
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime8, start, stop);
	outFile << "Compute z : " << elapsedTime8 << std::endl;

	/*Copy d_z to h_z*/
	float elapsedTime9;
	cudaEventRecord(start, 0);

	cuComplex* h_z = (cuComplex*)malloc(L * sizeof(cuComplex));
	cudaMemcpy(h_z, d_z, L * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);             //Timing ends
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime9, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	outFile << "Copy z: " << elapsedTime9 << std::endl;

	/*Clean up*/
	if (d_cube) cudaFree(d_cube);
	if (d_t) cudaFree(d_t);
	if (d_y) cudaFree(d_y);
	
	if (d_s) cudaFree(d_s);
	if (d_u) cudaFree(d_u);
	if (d_z) cudaFree(d_z);
	if (h_z) free(h_z);

	cudaEventRecord(stop0, 0);             //Timing ends
	cudaEventSynchronize(stop0);
	cudaEventElapsedTime(&elapsedTime0, start0, stop0);
	cudaEventDestroy(start0);
	cudaEventDestroy(stop0);

	outFile <<"Total: " << elapsedTime0 << std::endl;
	outFile << "Success!" << std::endl;
	
	//system("pause");
	return 0;
}
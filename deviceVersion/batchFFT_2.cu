#include "header.cuh"

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/********/
/* MAIN */
/********/
int batchFFT(cufftComplex* inputData, cufftComplex* outputData,int BATCH, int DATASIZE)
{

	// --- Device side input data allocation and initialization
	cufftComplex *deviceInputData; gpuErrchk(cudaMalloc((void**)&deviceInputData, DATASIZE * BATCH * sizeof(cufftComplex)));
	cudaMemcpy(deviceInputData, inputData, DATASIZE * BATCH * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	// --- Batched 1D FFTs
	cufftHandle handle;
	int rank = 1;                           // --- 1D FFTs
	int n[] = { DATASIZE };                 // --- Size of the Fourier transform
	int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
	int idist = DATASIZE, odist = DATASIZE; // --- Distance between batches
	int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
	int batch = BATCH;                      // --- Number of batched executions
	cufftPlanMany(&handle, rank, n,
		inembed, istride, idist,
		onembed, ostride, odist, CUFFT_C2C, batch);

	//cufftPlan1d(&handle, DATASIZE, CUFFT_R2C, BATCH);
	cufftExecC2C(handle, deviceInputData, outputData, CUFFT_FORWARD);
	cufftDestroy(handle);
	

	return 0;
}
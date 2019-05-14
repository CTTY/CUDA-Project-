#STAP-CUDA version

##Compile Type "nvcc batchFFT_2.cu CholeskySolver.cu Test.cu Util.cu MatrixTranspose.cu -o STAP -lcublas -lcusolver -lcufft" to compile, this will output a executable file named "STAP"

##Parameters The size of cube: M,N,L (L equals MxN by default)
Test times: TESTTIMES Number of bins: BIN Fdopp and other

Can all be modified in the "header.cuh" file.

##Output This program will output a text file recording the test data, the format of outputfile is: "M x N x L.txt"

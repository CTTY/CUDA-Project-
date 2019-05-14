# STAP-CUDA version

## Compile 
Enter "nvcc batchFFT_2.cu CholeskySolver.cu Test.cu Util.cu MatrixTranspose.cu -o STAP -lcublas -lcusolver -lcufft" in command line to compile, and this will output a executable file named "STAP". Enter "./STAP" to execute it.  


## Parameters 
The size of cube: M,N,L (L equals MxN by default)    
Numer of tests for each run: TESTTIMES (Test the same size cube for multiple times)      
Number of bins: BIN    
Fdopp and other   

All of them can be modified in the "header.cuh" file.

## Output 
This program will output a text file recording the test data, 
the format of outputfile is: "M x N x L.txt"

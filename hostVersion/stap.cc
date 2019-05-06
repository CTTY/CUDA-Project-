// stap code 
#include "Array.h"
#include "fftw++.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <valarray>
#include <complex>
#include <cstdio>
#include <vector>
#include <math.h>
#include <limits>
#include <cstring>
#include <cblas.h>
#include <lapacke.h>

// Compile with
// g++ -I .. -fopenmp stap.cc ../fftw++.cc -lfftw3 -lfftw3_omp -llapack -lblas
// Run with ./a.out 
// D'souza, Clive [cd585@cornell.edu] 
#define PI 3.1415926535
using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;
float random_float(float min, float max) {
	return ((float)rand() / RAND_MAX) * (max - min) + min;
}

int main() 
{
  //cout << "3D complex to complex in-place FFT" << endl;

  fftw::maxthreads=get_max_threads();
  
  int nx=32, ny=32, nz=nx*ny;
  size_t align=sizeof(Complex);
  //arrays and floats
  array3<Complex> cube(nx,ny,nz,align);
  array1<Complex> y(nx*ny,align);
  array2<Complex> S(nx*ny,nx*ny,align);
  array2<Complex> s(nx*ny,nx*ny,align);
  array1<Complex> t(nx*ny,align);
  array1<Complex> ts(nx*ny,align);
  array1<Complex> tH(nx*ny,align);
  array1<Complex> F(nx,align);
  array1<Complex> A(1,align);
  array1<Complex> star(1,align);
  array1<Complex> u(ny,align);
  array1<Complex> it(1,align);
  array1<Complex> z(1,align);
  array1<Complex> alpha(1,align);
  array1<Complex> beta(1,align);
  alpha=Complex(1,0);
  beta=alpha;

  float Fdopp = random_float(0,100);
  float theta = random_float(0,90);
  float lambda= random_float(0,10);
  float angle= theta/lambda;
  fft3d Forward(-1,cube);
  star=Complex(1,1); //simple vector to conjugate any other vector 
  //generate random cube and F vector for steering angle
  srand(time(NULL));
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      for(unsigned int k=0; k < nz; k++) 
        cube(i,j,k)=Complex(10*random_float(0.0,1.0),10*random_float(0.0,1.0));
  cout<<"Created the cube"<<endl;

  for(unsigned int i=0;i<nx;i++){
    F(i)=Complex(cos(2*PI*i*Fdopp),-sin(2*PI*i*Fdopp));
  }
  //cout << "\ninput:\n" << cube; 
  //fft start
  Forward.fft(cube);
  //fft end
  //cout << "\noutput:\n" << cube;

  // Record start time
  auto start = std::chrono::high_resolution_clock::now();
  //vectorize slice  and get Covariance start


  for(unsigned int i=0; i < nz; i++) {
    for(unsigned int j=0; j < nx; j++) { 
      for(unsigned int k=0; k < ny; k++) {
        y(k+ny*j)=cube(j,k,i);
      }
    }
    cblas_cdotc_sub(nx*ny,y,1,y,1,s);
    S+=s;
  }
  S-=s; //done to exclude the product of one 'gate under test'. in our case, we take it to be the last slice.
  cout<<"Covariance made"<<endl;    
  //vectorize slice end
  //cout<<"\nCovariance:\n"<< S;
  __complex__ float Smat[nx][ny];
  for(unsigned int i=0; i < nx; i++){
    for(unsigned int j=0; j < ny; j++){
    Smat[i][j]=(real(S(i,j)),imag(S(i,j)));
    }
  }
  cout<<"Covariance copied"<<endl;
  //Steering vector start

  for(unsigned int i=0;i<ny;i++){
    A=Complex(cos(2*PI*i*sin(angle)),-sin(2*PI*i*sin(angle)));
    cblas_cscal(ny,A,F,1);
    for(unsigned int j=0;j<nx;j++){
      t(j+i*nx)=F(j);
    }
  }
  __complex__ float tstar[nx*ny];
  //Steering vector end
cout<<"steering vector made"<<endl;
  //Conjugate steering vector start
  cblas_cscal(nx*ny,star,t,1);
  cblas_ccopy(nx*ny,t,1,tstar,1);
  cblas_ccopy(nx*ny,t,1,ts,1);
  //Conjugate steering vector end

  //Cholesky start
  int info;
  char uplo= 'U';
  int v=min(nx,ny)-1;
  cpbsv_(&uplo,&ny,&v,&ny,*Smat,&nx,tstar,&ny,&info);
  //Cholesky end
  cout<<"Cholesky done"<<endl;
  cblas_ccopy(ny,tstar,1,u,1);    //u
  cblas_cscal(ny,star,tstar,1);   //u*
  //h=u/(tH x u*)
  cblas_cgemm(CblasRowMajor,CblasTrans,CblasNoTrans,1,1,ny,&alpha,tstar,1,ts,1,&beta,it,1);
  cblas_cscal(ny,it,u,1); 
  cblas_cgemm(CblasRowMajor,CblasTrans,CblasNoTrans,1,1,ny,&alpha,u,1,y,1,&beta,z,1);
  cout<<"SUCCESS!"<<endl;

  // Record end time
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count()*1000 << " ms\n";
}

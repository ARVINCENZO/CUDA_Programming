#include <iostream>
#include <cmath>

//initialize matrix dimension 10*10
const int N= 10;

//kernel function which uses 2d block and 2d grid
__global__ void MatrixAddition(int* A, int* B, int* C, int size){
  int i= blockIdx.x * blockDim.x + threadIdx.x; // column index
  int j= blockIdx.y * blockDim.y + threadIdx.y; // row index
 
  if(i< size && j<size){
    int index= i * size + j; // index calculation
    C[index] = A[index] + B[index];
  }
}

//main function 
int main(){

  int matrix_size_bytes= N * N * sizeof(int);

  //initialize host(cpu) variables

  int *h_a = new int[N * N];
  int *h_b = new int[N * N];
  int *h_c = new int[N * N];


  //fill some values in the empty matrices

  for(int i=0; i< N*N ;++i){
    h_a[i]=1;
    h_b[i]=2;
    h_c[i]=0;
  }


  //initialize device(gpu) variables

  int *d_a, *d_b, *d_c;

  //allocate memory to device variables

  cudaMalloc((void**)&d_a, matrix_size_bytes);
  cudaMalloc((void**)&d_b, matrix_size_bytes);
  cudaMalloc((void**)&d_c, matrix_size_bytes);

  //copy host matrices to device for calculation

  cudaMemcpy(d_a, h_a, matrix_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, matrix_size_bytes, cudaMemcpyHostToDevice);


  //grid and block dimension calculation
  dim3 blockDim(32,16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

  // call the kernel function
  MatrixAddition<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

  //synchronize cpu and gpu for calculation
  cudaDeviceSynchronize();

  //copy the result back to host(CPU)
  cudaMemcpy(h_c, d_c, matrix_size_bytes, cudaMemcpyDeviceToHost);


  //print the resultant matrix
  for(int i=0; i< N*N; ++i){
    std::cout<< h_c[i] << ",";
  }
  std::cout << std::endl;

  //verify the results
  for(int i = 0; i < N * N; ++i)
  {
    if(h_c[i] != 3)
    {
        std::cout << "Verification Failed!" << std::endl;
        return 0;
    }
  }


  //free the device variables
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  //free the host memory
  delete[] h_a; 
  delete[] h_b;
  delete[] h_c; 

  return 0;
}
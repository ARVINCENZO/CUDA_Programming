#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// defining the vector size and block size

#define N 1000000
#define BLOCK_SIZE 256


// cpu vector_addition

void vec_add_cpu(float* a, float* b, float* c, int n){
  for(int i=0;i<n;i++){
      c[i]= a[i] + b[i];
  }
}

// CUDA kernel for vector_addition

__global__ void vec_add_gpu(float* a, float*b, float*c, int n){
  int i= threadIdx.x + blockIdx.x * blockDim.x;

  if(i<n){
    c[i]= a[i]+ b[i];
  }
}

// initialize vector with random values

void init_vec(float* vec, int n){
  for(int i=0; i<n;i++){
    vec[i]= (float)rand()/ RAND_MAX;
  }
}

// function to measure execution time

double get_time(){
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
  float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
  float *d_a, *d_b, *d_c;
  size_t size= N* sizeof(float);


  // allocate host memory
  h_a= (float*)malloc(size);
  h_b= (float*)malloc(size);
  h_c_cpu= (float*)malloc(size);
  h_c_gpu= (float*)malloc(size);

  //initialize vectors
  srand(time(NULL));
  init_vec(h_a, N);
  init_vec(h_b,N);

  // allocate device memory

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // copy data to device

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // define grid and block dimensions
  int num_blocks= (N + BLOCK_SIZE-1)/BLOCK_SIZE;

  // warmup runs

  printf("Performing warmup runs...\n");
  for(int i=0; i<4;i++){

    vec_add_cpu(h_a, h_b, h_c_cpu, N);

    vec_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();
  }

  // Benchmark cpu implementation

  printf("Benchmarking cpu implementation...\n");

  double cpu_total_time= 0.0;
  for(int i=0; i<20; i++){

    double cpu_start_time= get_time();

    vec_add_cpu(h_a, h_b, h_c_cpu, N);

    double cpu_end_time= get_time();

    cpu_total_time += cpu_end_time - cpu_start_time;
  }

  double cpu_avg_time= cpu_total_time/ 20.0;



  // benchmarking gpu implementation

  printf("Benchmarking gpu implementation...\n");
  double gpu_total_time= 0.0;
  for(int i=0; i<20; i++){

    double gpu_start_time= get_time();

    vec_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    double gpu_end_time= get_time();

    gpu_total_time += gpu_end_time- gpu_start_time;

  }

  double gpu_avg_time= gpu_total_time/20.0;


  // Print results
  printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
  printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  // verify the results
  cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
  bool correct= true;

  for(int i=0;i<N;i++){
    if(fabs(h_c_gpu[i]- h_c_gpu[i]) > 1e-5){
      correct= false;
      break;
    }
  }
  printf("Results are %s\n ", correct ? "correct" : "incorrect");

  // Free memory
  free(h_a);
  free(h_b);
  free(h_c_cpu);
  free(h_c_gpu);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
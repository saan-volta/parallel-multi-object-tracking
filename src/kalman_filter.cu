#include <stdio.h>
#include <cmath>
#include <iostream>
#include "../include/kalman_filter.h"

#define IDX(r, c, width) ((r) * (width) + (c))
#define X_DIM 4
#define Z_DIM 2
#define P_DIM 16



__global__ void kalman_predict_update(
  float* x,
  float* P,
  const float* z,
  const int N,
  const float dt,
  const float Q_var,
  const float R_var
)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= N) return;

  float x_reg[4];
  float P_reg[4*4];

  #pragma unroll
  for (int i = 0; i < 4; i++) x_reg[i] = x[4*idx +i];
  #pragma unroll
  for (int i = 0; i<4*4; i++) P_reg[i] = P[16*idx + i];

  // x <- A @ x
  x_reg[0] += x_reg[1]*dt;
  x_reg[2] += x_reg[3]*dt;

  // P <- A @ P @ A^T + Q
  for (int c = 0; c < 4; c++) P_reg[IDX(0,c,4)] += P_reg[IDX(1,c,4)]*dt;
  for (int c = 0; c < 4; c++) P_reg[IDX(2,c,4)] += P_reg[IDX(3,c,4)]*dt;
  for (int r = 0; r < 4; r++) P_reg[IDX(r,0,4)] += P_reg[IDX(r,1,4)]*dt;
  for (int r = 0; r < 4; r++) P_reg[IDX(r,2,4)] += P_reg[IDX(r,3,4)]*dt;
  const float q1 = 1/4*pow(dt,4)*Q_var, q2 = 1/2*pow(dt,3)*Q_var, q3 = pow(dt,2)*Q_var;
  P_reg[IDX(0,0,4)] += q1; P_reg[IDX(2,2,4)] += q1;
  P_reg[IDX(0,1,4)] += q2; P_reg[IDX(1,0,4)] += q2; P_reg[IDX(2,3,4)] += q2; P_reg[IDX(3,2,4)] += q2;
  P_reg[IDX(1,1,4)] += q3; P_reg[IDX(3,3,4)] += q3;

  // S <- H @ P @ H^T + R
  // [p11 p13]
  // [p13 p33]
  // R = I*R_var
  float s1 = P_reg[IDX(0,0,4)], s2 = P_reg[IDX(0,2,4)], s4 = P_reg[IDX(2,2,4)];  // s3 = s2
  s1 += R_var; s4 += R_var;

  // K <- P @ H^T @ S^-1
  float Sdet = s1*s4-s2*s2;
  s1 /= Sdet; s2 /= Sdet; s4 /= Sdet;
  float K_mat[4*2];
  for (int i = 0; i < 4; i++) {
    K_mat[IDX(i,0,2)] = P_reg[IDX(i,0,4)]*s4 + P_reg[IDX(i,2,4)]*(-s2); 
    K_mat[IDX(i,1,2)] = P_reg[IDX(i,0,4)]*(-s2) + P_reg[IDX(i,2,4)]*s1; 
  }
  // x <- x + K @ (z - H @ x)
  float zxdiff = z[idx + 0]-x_reg[0]; float zydiff = z[idx + 1] - x_reg[2];
  for (int i = 0; i < 4; i++) {
    x_reg[i] +=  K_mat[IDX(i,0,2)]*zxdiff + K_mat[IDX(i,1,2)]*zydiff;
  }

  // P <- (I - K @ H) @ P = P - K@H@P
  for (int i =0; i<4; i++) 
    for (int j = 0; j<4; j++)
      P[16*idx + IDX(i,j,4)] = P_reg[IDX(i,j,4)] - (K_mat[IDX(i,0,2)]*P_reg[IDX(0,j,4)] + K_mat[IDX(i,1,2)]*P_reg[IDX(2,j,4)] );

  for (int i = 0; i<4; i++) x[4*idx + i] = x_reg[i];
    
}

__global__ void init_KF(
  float* P,
  int N
)
{
  // initialize P-matrices to identity
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= N) return;

  #pragma unroll
  for (int i = 0; i < X_DIM*X_DIM; i++)                         
  {
    if (i%(X_DIM+1)==0)
      P[idx*P_DIM + i] = 1;
    else 
      P[idx*P_DIM +i] = 0;
  }
}


constexpr int ceil_div(int a, int b) {
  return (a+b-1)/b;
}


void kalman_launch(
  // float* x_arr,
  const float* x0_data,
  const float* z_data,
  const int N,
  const int T,
  const float dt,
  const float Q_var,
  const float R_var
)
{
  // allocate space on device

  size_t x_size = sizeof(float)*X_DIM *N;
  size_t P_size = sizeof(float)*X_DIM*X_DIM *N;
  size_t z_size = sizeof(float)*Z_DIM *N;

  float* host_x = (float*)malloc(x_size);
  float* dev_x; float* dev_P; float* dev_z;
  cudaMalloc(&dev_x, x_size ); cudaMalloc(&dev_P, P_size); cudaMalloc(&dev_z, z_size );
  // copy to device
  cudaMemcpy(dev_x, x0_data, x_size, cudaMemcpyHostToDevice);
  int num_t = 128; int num_b = ceil_div(N, num_t );
  init_KF<<<num_t, num_b>>>(dev_P, N);

  // execute
  for (int t = 0; t < T; t++)
  {
    cudaMemcpy(dev_z, z_data+(t*Z_DIM*N), z_size, cudaMemcpyHostToDevice);
    kalman_predict_update<<<num_t, num_b>>>(dev_x, dev_P, dev_z, N, dt, Q_var, R_var);
    cudaMemcpy(host_x, dev_x, x_size, cudaMemcpyDeviceToHost);

    std::cout << "[ " << host_x[0] << "\t" << host_x[1] << "\t" << host_x[2] << "\t" << host_x[3] << " ]\n";
  }
 
  free(host_x);
  cudaFree(dev_x); cudaFree(dev_z); cudaFree(dev_P);
 
}


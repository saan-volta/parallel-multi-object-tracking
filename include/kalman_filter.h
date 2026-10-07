#ifndef KALMAN_FILTER_DRIVER_H
#define KALMAN_FILTER_DRIVER_H

#define X_DIM 4
#define Z_DIM 2
#define P_DIM 16

void kalman_launch(
  // float* x_arr, // store pointer
  const float* x0_data,
  const float* z_data,
  const int N,
  const int T,
  const float dt,
  const float Q_var,
  const float R_var
);


#endif

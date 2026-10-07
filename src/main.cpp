#include <string>
#include <iostream>
#include <fstream>
#include "json.hpp"
#include "../include/kalman_filter.h"


int read_bin(std::string fname , float* arr, int size)
{
  std::ifstream file(fname, std::ios::binary );
  if (!file) return 1;
  file.seekg(0, std::ios::beg );
  file.read(reinterpret_cast<char*>(arr), size*sizeof(float));
  return 0;
}


using json = nlohmann::json;
int main(int arc, char* argv[])
{

  // int data_idx = std::stoi(argv[1]);
  std::string data_path = std::string("data/") + std::string(argv[1]) + std::string("/");
  

  std::ifstream inf( data_path + std::string("data_in.json") );
  json data_dict = json::parse(inf);

  int N = data_dict["N"];
  int T = data_dict["T"];
  float Q_var = data_dict["Q_var"];
  float R_var = data_dict["R_var"];
  float dt = data_dict["dt"];
  std::string x0fname = data_dict["x0fname"];
  std::string zfname = data_dict["zfname"];

  size_t x_size = sizeof(float)*X_DIM *N;
  // size_t P_size = sizeof(float)*X_DIM*X_DIM *N;
  size_t z_size = sizeof(float)*Z_DIM *N;

  // load data
  float x0_array[x_size];
  float z_array[z_size*T];

  read_bin( data_path + x0fname, x0_array, x_size);
  read_bin( data_path + zfname, z_array, z_size*T);

  // std::cout << x0_array[0] << "\t" << z_array[0] << "\n";
  // std::cout << x0_array[1] << "\t" << z_array[1] << "\n";
  // std::cout << x0_array[2] << "\t" << z_array[2] << "\n";
  
  kalman_launch(x0_array, z_array, N, T, dt, Q_var, R_var);


  
}

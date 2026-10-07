#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "json.hpp"
#include "../include/kalman_filter.h"


int read_bin(std::string fname , float* arr, int size)
{
  std::ifstream file(fname, std::ios::binary );
  if (!file) return 1;
  file.seekg(0, std::ios::beg );
  file.read(reinterpret_cast<char*>(arr), size*sizeof(float));
  std::cout << "Read " << fname <<  std::endl;
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
  std::cout << "Read " << "json \t N = " << N << " \n" << std::endl;

  int x_size =X_DIM *N;
  int z_size = Z_DIM *N;

  // load data
  std::vector<float> x0_array(x_size);
  std::cout << "Array x0 allocated\n"; 
  std::vector<float> z_array(z_size*T);
  std::cout << "Array z allocated\n"; 

  read_bin( data_path + x0fname, x0_array.data(), x_size);
  read_bin( data_path + zfname, z_array.data(), z_size*T);

  
  kalman_launch(x0_array.data(), z_array.data(), N, T, dt, Q_var, R_var);


  
}

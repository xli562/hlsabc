#include "cordic.h"


int main(int argc, char* argv[]) {
  if (argc != 3) { return 1; }

  theta_type theta = static_cast<theta_type>(std::atof(argv[1]));
  int num_iter = std::atoi(argv[2]);

  cos_sin_type s, c;
  cordic(theta, s, c, num_iter);

  std::cout << s << "," << c << "\n";
  return 0;
}
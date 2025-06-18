#include "cutlass/util/command_line.h"
#include "tma_copy.h"

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  int M, N, iterations;  
  cmd.get_cmd_line_argument("M", M, 16384);
  cmd.get_cmd_line_argument("N", N, 16384);
  cmd.get_cmd_line_argument("iterations", iterations, 10);

  std::cout << "(M, N): " << M << ", " << N << std::endl;

  copy_host_tma_load_and_store_kernel(M, N, iterations);

  return 0;
}

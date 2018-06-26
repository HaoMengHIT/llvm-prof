#include "Profiling.h"
#include <stdlib.h>

static uint64_t *ArrayStart;
static uint64_t NumElements;
static int *ArrayRankStart;
static int NumRankElements;

static void EdgeRankProfAtExitHandler(void) {
  write_edge_rank_profiling_data_long(EdgeInfo64, ArrayStart, NumElements, ArrayRankStart, NumRankElements);
}

int llvm_start_edge_rank_profiling(int argc, const char** argv,
                                    uint64_t* arrayStart, uint64_t numElements, int* arrayRankStart, int numRankElements)
{
  int Ret = save_arguments(argc, argv);
  ArrayStart = arrayStart;
  ArrayRankStart = arrayRankStart;
  NumElements = numElements;
  NumRankElements = numRankElements;
  atexit(EdgeRankProfAtExitHandler);
  return Ret;
}

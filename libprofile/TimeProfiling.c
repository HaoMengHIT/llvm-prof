#include "Profiling.h"
#include <stdlib.h>

static double *ArrayStart;
static uint64_t NumElements;
static int *ArrayRankStart;
static int NumRankElements;

static void TimeProfAtExitHandler(void) {
  write_time_rank_profiling_data_double(MPITimeInfo, ArrayStart, NumElements, ArrayRankStart, NumRankElements);
}

int llvm_start_time_profiling(int argc, const char** argv,
                                    double* arrayStart, uint64_t numElements, int* arrayRankStart, int numRankElements)
{
  int Ret = save_arguments(argc, argv);
  ArrayStart = arrayStart;
  ArrayRankStart = arrayRankStart;
  NumElements = numElements;
  NumRankElements = numRankElements;
  atexit(TimeProfAtExitHandler);
  return Ret;
}

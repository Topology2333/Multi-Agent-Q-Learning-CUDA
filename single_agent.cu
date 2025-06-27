#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  }

struct curandStateSimple {
  unsigned int seed;
};

__device__ float deviceRand(curandStateSimple *state) {
  state->seed = (1103515245u * state->seed + 12345u) % 2147483648u;
  return (float)state->seed / 2147483648.0f;
}

__device__ __constant__ int d_SIZE;
__device__ __constant__ int d_N_MINES;
__device__ __constant__ int d_FLAG_X;
__device__ __constant__ int d_FLAG_Y;
__device__ __constant__ float d_ALPHA;
__device__ __constant__ float d_GAMMA;
__device__ __constant__ float d_EPSILON;
__device__ __constant__ int d_MAX_STEPS;

__device__ int d_grid[1024 * 1024];
__device__ float d_Q[1024 * 1024 * 4];

__device__ void atomicUpdateQ(int x, int y, int action, float tdTarget,
                              float alpha, int size) {
  int idx = (x * size + y) * 4 + action;
  float oldVal = d_Q[idx];
  float update = oldVal + alpha * (tdTarget - oldVal);
  int *intPtr = (int *)(&d_Q[idx]);
  int oldInt = __float_as_int(oldVal);
  int newInt;
  while (true) {
    newInt = __float_as_int(update);
    int ret = atomicCAS(intPtr, oldInt, newInt);
    if (ret == oldInt)
      break;
    oldVal = __int_as_float(ret);
    update = oldVal + alpha * (tdTarget - oldVal);
    oldInt = ret;
  }
}

__device__ void stepDev(int &x, int &y, int action, int size) {
  if (action == 0) {
    x = (x > 0) ? x - 1 : x;
  } else if (action == 1) {
    x = (x < size - 1) ? x + 1 : x;
  } else if (action == 2) {
    y = (y > 0) ? y - 1 : y;
  } else if (action == 3) {
    y = (y < size - 1) ? y + 1 : y;
  }
}

__device__ int chooseActionDev(int x, int y, float eps,
                               curandStateSimple *randState, int size) {
  float r = deviceRand(randState);
  if (r < eps) {
    return (int)(deviceRand(randState) * 4.0f);
  }
  int base = (x * size + y) * 4;
  float bestVal = d_Q[base];
  int bestAct = 0;
  for (int a = 1; a < 4; a++) {
    float val = d_Q[base + a];
    if (val > bestVal) {
      bestVal = val;
      bestAct = a;
    }
  }
  return bestAct;
}

__global__ void runEpisodesKernel(int totalEpisodes,
                                  curandStateSimple *randStates) {
  int size = d_SIZE;
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int episodesPerThread = (totalEpisodes + (stride - 1)) / stride;
  int startEp = threadId * episodesPerThread;
  int endEp = min(startEp + episodesPerThread, totalEpisodes);

  curandStateSimple localState = randStates[threadId];

  for (int ep = startEp; ep < endEp; ep++) {
    int x = 0, y = 0;
    bool done = false;
    int steps = 0;
    while (!done) { // && steps < d_MAX_STEPS) {
      steps++;
      int action = chooseActionDev(x, y, d_EPSILON, &localState, size);
      int oldX = x;
      int oldY = y;
      stepDev(x, y, action, size);

      int reward = -1;
      bool isDone = false;
      int mineVal = d_grid[x * size + y];
      if (mineVal == -1) {
        reward = -10;
        isDone = true;
      } else if (x == d_FLAG_X && y == d_FLAG_Y) {
        reward = 10;
        isDone = true;
      }

      int baseNext = (x * size + y) * 4;
      float bestNext = d_Q[baseNext];
      for (int a = 1; a < 4; a++) {
        float val = d_Q[baseNext + a];
        if (val > bestNext)
          bestNext = val;
      }
      float tdTarget = (float)reward + (isDone ? 0.0f : (d_GAMMA * bestNext));
      atomicUpdateQ(oldX, oldY, action, tdTarget, d_ALPHA, size);

      done = isDone;
    }
  }
  randStates[threadId] = localState;
}

void generateEnvironment(std::vector<int> &h_grid, int size, int n_mines,
                         int flag_x, int flag_y) {
  std::fill(h_grid.begin(), h_grid.end(), 0);
  int count = 0;
  while (count < n_mines) {
    int rx = rand() % size;
    int ry = rand() % size;
    if ((rx == 0 && ry == 0) || (rx == flag_x && ry == flag_y))
      continue;
    if (h_grid[rx * size + ry] != -1) {
      h_grid[rx * size + ry] = -1;
      count++;
    }
  }
}

void printPolicyCPU(const std::vector<int> &h_grid,
                    const std::vector<float> &h_Q, int size, int flag_x,
                    int flag_y) {
  for (int x = 0; x < size; x++) {
    for (int y = 0; y < size; y++) {
      int idx = x * size + y;
      if (h_grid[idx] == -1) {
        std::cout << "M ";
        continue;
      }
      if (x == flag_x && y == flag_y) {
        std::cout << "F ";
        continue;
      }
      float bestVal = h_Q[idx * 4];
      int bestAct = 0;
      for (int a = 1; a < 4; a++) {
        float val = h_Q[idx * 4 + a];
        if (val > bestVal) {
          bestVal = val;
          bestAct = a;
        }
      }
      if (bestAct == 0)
        std::cout << "^ ";
      else if (bestAct == 1)
        std::cout << "v ";
      else if (bestAct == 2)
        std::cout << "< ";
      else if (bestAct == 3)
        std::cout << "> ";
    }
    std::cout << "\n";
  }
}

int main(int argc, char **argv) {
  int size = 32;
  int n_mines = 40;
  int flag_x = 31;
  int flag_y = 31;
  float alpha = 0.1f;
  float gamma = 0.9f;
  float epsilon = 0.1f;
  int episodes = 20000;
  int max_steps = 1000;
  int blocks = 128;
  int threads_per_block = 128;

  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      size = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--n_mines") == 0 && i + 1 < argc) {
      n_mines = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--flag_x") == 0 && i + 1 < argc) {
      flag_x = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--flag_y") == 0 && i + 1 < argc) {
      flag_y = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
      alpha = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
      gamma = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
      epsilon = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
      episodes = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--max_steps") == 0 && i + 1 < argc) {
      max_steps = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
      blocks = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads_per_block") == 0 &&
               i + 1 < argc) {
      threads_per_block = std::atoi(argv[++i]);
    }
  }

  srand((unsigned)time(NULL));

  std::vector<int> h_grid(size * size, 0);
  generateEnvironment(h_grid, size, n_mines, flag_x, flag_y);

  CHECK_CUDA(cudaMemcpyToSymbol(d_SIZE, &size, sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_N_MINES, &n_mines, sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_FLAG_X, &flag_x, sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_FLAG_Y, &flag_y, sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_ALPHA, &alpha, sizeof(float)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_GAMMA, &gamma, sizeof(float)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_EPSILON, &epsilon, sizeof(float)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_MAX_STEPS, &max_steps, sizeof(int)));

  int gridMem = size * size;
  CHECK_CUDA(cudaMemcpyToSymbol(d_grid, h_grid.data(), sizeof(int) * gridMem));

  int qMem = size * size * 4;
  std::vector<float> h_Q(qMem, 0.0f);
  CHECK_CUDA(cudaMemcpyToSymbol(d_Q, h_Q.data(), sizeof(float) * qMem));

  int totalThreads = blocks * threads_per_block;
  curandStateSimple *d_randStates;
  CHECK_CUDA(
      cudaMalloc(&d_randStates, totalThreads * sizeof(curandStateSimple)));

  std::vector<curandStateSimple> initStates(totalThreads);
  for (int i = 0; i < totalThreads; i++) {
    initStates[i].seed = rand();
  }
  CHECK_CUDA(cudaMemcpy(d_randStates, initStates.data(),
                        totalThreads * sizeof(curandStateSimple),
                        cudaMemcpyHostToDevice));

  runEpisodesKernel<<<blocks, threads_per_block>>>(episodes, d_randStates);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaFree(d_randStates));

  CHECK_CUDA(cudaMemcpyFromSymbol(h_Q.data(), d_Q, sizeof(float) * qMem));

  printPolicyCPU(h_grid, h_Q, size, flag_x, flag_y);

  return 0;
}

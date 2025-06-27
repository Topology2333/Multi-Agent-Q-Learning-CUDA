#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
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

// Simple RNG state
struct SimpleCurand {
  unsigned int seed;
};

__device__ float deviceRand(SimpleCurand *state) {
  state->seed = (1103515245u * state->seed + 12345u) & 0x7fffffff;
  return (float)state->seed / 2147483648.0f;
}

__device__ int *d_grid;
__device__ float *d_Q;
__device__ int *d_active;
__device__ int *d_agentX;
__device__ int *d_agentY;

__device__ int d_SIZE;
__device__ int d_N_MINES;
__device__ int d_FLAG_X;
__device__ int d_FLAG_Y;
__device__ int d_N_AGENTS;
__device__ float d_ALPHA;
__device__ float d_GAMMA;
__device__ float d_EPSILON;
__device__ int d_EPISODES;

// Utility function for atomic Q-value update
__device__ void atomicUpdateQ(int x, int y, int action, float tdTarget,
                              float alpha) {
  int idx = (x * d_SIZE + y) * 4 + action;
  float oldVal = d_Q[idx];
  float newVal = oldVal + alpha * (tdTarget - oldVal);
  int *intPtr = (int *)(&d_Q[idx]);
  int oldInt = __float_as_int(oldVal);
  int newInt;
  while (true) {
    newInt = __float_as_int(newVal);
    int ret = atomicCAS(intPtr, oldInt, newInt);
    if (ret == oldInt)
      break;
    oldVal = __int_as_float(ret);
    newVal = oldVal + alpha * (tdTarget - oldVal);
    oldInt = ret;
  }
}

__device__ bool isMineDev(int x, int y) {
  return (d_grid[x * d_SIZE + y] == -1);
}

// Kernel: reset agents
__global__ void resetAgentsKernel() {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d_N_AGENTS) {
    d_agentX[i] = 0;
    d_agentY[i] = 0;
    d_active[i] = 1;
  }
}

__global__ void stepAgentsKernel(SimpleCurand *randStates, int *d_done) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d_N_AGENTS || d_active[i] == 0)
    return;

  // 当前状态
  int x = d_agentX[i];
  int y = d_agentY[i];

  // 选择当前动作 a（ε-greedy）
  float r = deviceRand(&randStates[i]);
  int action = 0;
  if (r < d_EPSILON) {
    float rr = deviceRand(&randStates[i]) * 4.0f;
    action = (int)rr;
    if (action > 3)
      action = 3;
  } else {
    int base = (x * d_SIZE + y) * 4;
    float bestVal = d_Q[base];
    int bestAct = 0;
    for (int a = 1; a < 4; a++) {
      float val = d_Q[base + a];
      if (val > bestVal) {
        bestVal = val;
        bestAct = a;
      }
    }
    action = bestAct;
  }

  // 执行动作，转移到新状态
  int oldX = x;
  int oldY = y;
  if (action == 0 && x > 0)
    x--;
  else if (action == 1 && x < d_SIZE - 1)
    x++;
  else if (action == 2 && y > 0)
    y--;
  else if (action == 3 && y < d_SIZE - 1)
    y++;

  d_agentX[i] = x;
  d_agentY[i] = y;

  // 奖励函数 & 是否结束
  int reward = -1;
  bool doneLocal = false;
  if (isMineDev(x, y)) {
    reward = -10;
    d_active[i] = 0;
    doneLocal = true;
  } else if (x == d_FLAG_X && y == d_FLAG_Y) {
    reward = 10;
    d_active[i] = 0;
    doneLocal = true;
  }

  // 选择下一个动作 a'（ε-greedy）用于 SARSA 更新
  int nextAction = 0;
  float r2 = deviceRand(&randStates[i]);
  if (r2 < d_EPSILON) {
    float rr = deviceRand(&randStates[i]) * 4.0f;
    nextAction = (int)rr;
    if (nextAction > 3)
      nextAction = 3;
  } else {
    int base = (x * d_SIZE + y) * 4;
    float bestVal = d_Q[base];
    int bestAct = 0;
    for (int a = 1; a < 4; a++) {
      float val = d_Q[base + a];
      if (val > bestVal) {
        bestVal = val;
        bestAct = a;
      }
    }
    nextAction = bestAct;
  }

  // SARSA TD Target
  int nextBase = (x * d_SIZE + y) * 4;
  float qNext = d_Q[nextBase + nextAction];
  float tdTarget = reward + (doneLocal ? 0.0f : d_GAMMA * qNext);

  // Q(s,a) ← Q(s,a) + α (r + γ Q(s',a') - Q(s,a))
  atomicUpdateQ(oldX, oldY, action, tdTarget, d_ALPHA);

  if (doneLocal) {
    atomicExch(&d_done[i], 1);
  }
}

// Kernel: count active agents
__global__ void countActiveKernel(int *d_count) {
  __shared__ int blockCount;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0)
    blockCount = 0;
  __syncthreads();

  int value = 0;
  if (i < d_N_AGENTS && d_active[i] == 1) {
    value = 1;
  }
  atomicAdd(&blockCount, value);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(d_count, blockCount);
  }
}

// Kernel: init RNG states
__global__ void initRandKernel(SimpleCurand *randStates, unsigned int seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d_N_AGENTS) {
    randStates[i].seed = seed ^ (i + 12345);
  }
}

// Host function: place mines
void placeMinesHost(int size, int n_mines, int flag_x, int flag_y,
                    std::vector<int> &h_grid) {
  int placed = 0;
  while (placed < n_mines) {
    int rx = rand() % size;
    int ry = rand() % size;
    if ((rx == 0 && ry == 0) || (rx == flag_x && ry == flag_y)) {
      continue;
    }
    if (h_grid[rx * size + ry] != -1) {
      h_grid[rx * size + ry] = -1;
      placed++;
    }
  }
}

void printPolicyCPU(const std::vector<int> &h_grid,
                    const std::vector<float> &h_Q, int size, int flag_x,
                    int flag_y) {
  for (int x = 0; x < size; x++) {
    for (int y = 0; y < size; y++) {
      if (h_grid[x * size + y] == -1) {
        std::cout << "M ";
        continue;
      }
      if (x == flag_x && y == flag_y) {
        std::cout << "F ";
        continue;
      }
      int base = (x * size + y) * 4;
      float bestVal = h_Q[base];
      int bestAct = 0;
      for (int a = 1; a < 4; a++) {
        float val = h_Q[base + a];
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
  int size = 46;
  int n_mines = 96;
  int flag_x = 45;
  int flag_y = 45;
  int n_agents = 512;
  float alpha = 0.1f;
  float gamma = 0.9f;
  float epsilon = 0.1f;
  int episodes = 1000;
  int max_steps_per_episode = 2000;
  int threads_per_block = 256;

  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      size = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--n_mines") == 0 && i + 1 < argc) {
      n_mines = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--flag_x") == 0 && i + 1 < argc) {
      flag_x = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--flag_y") == 0 && i + 1 < argc) {
      flag_y = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--n_agents") == 0 && i + 1 < argc) {
      n_agents = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
      alpha = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
      gamma = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
      epsilon = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
      episodes = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--max_steps_per_episode") == 0 &&
               i + 1 < argc) {
      max_steps_per_episode = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads_per_block") == 0 &&
               i + 1 < argc) {
      threads_per_block = std::atoi(argv[++i]);
    }
  }

  srand((unsigned)time(NULL));

  std::vector<int> h_grid(size * size, 0);
  placeMinesHost(size, n_mines, flag_x, flag_y, h_grid);

  // Q table
  std::vector<float> h_Q(size * size * 4, 0.0f);

  // Agents
  std::vector<int> h_agentX(n_agents, 0);
  std::vector<int> h_agentY(n_agents, 0);
  std::vector<int> h_active(n_agents, 1);

  // Copy environment to device
  int gridSizeInBytes = size * size * sizeof(int);
  int qSizeInBytes = size * size * 4 * sizeof(float);
  int agentSizeInBytes = n_agents * sizeof(int);

  int *d_gridPtr;
  CHECK_CUDA(cudaMalloc(&d_gridPtr, gridSizeInBytes));
  CHECK_CUDA(cudaMemcpy(d_gridPtr, h_grid.data(), gridSizeInBytes,
                        cudaMemcpyHostToDevice));

  float *d_QPtr;
  CHECK_CUDA(cudaMalloc(&d_QPtr, qSizeInBytes));
  CHECK_CUDA(
      cudaMemcpy(d_QPtr, h_Q.data(), qSizeInBytes, cudaMemcpyHostToDevice));

  int *d_agentXPtr;
  CHECK_CUDA(cudaMalloc(&d_agentXPtr, agentSizeInBytes));
  CHECK_CUDA(cudaMemcpy(d_agentXPtr, h_agentX.data(), agentSizeInBytes,
                        cudaMemcpyHostToDevice));

  int *d_agentYPtr;
  CHECK_CUDA(cudaMalloc(&d_agentYPtr, agentSizeInBytes));
  CHECK_CUDA(cudaMemcpy(d_agentYPtr, h_agentY.data(), agentSizeInBytes,
                        cudaMemcpyHostToDevice));

  int *d_activePtr;
  CHECK_CUDA(cudaMalloc(&d_activePtr, agentSizeInBytes));
  CHECK_CUDA(cudaMemcpy(d_activePtr, h_active.data(), agentSizeInBytes,
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyToSymbol(d_grid, &d_gridPtr, sizeof(int *)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_Q, &d_QPtr, sizeof(float *)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_agentX, &d_agentXPtr, sizeof(int *)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_agentY, &d_agentYPtr, sizeof(int *)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_active, &d_activePtr, sizeof(int *)));

  CHECK_CUDA(cudaMemcpyToSymbol(d_SIZE, &size, sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_N_MINES, &n_mines, sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_FLAG_X, &flag_x, sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_FLAG_Y, &flag_y, sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_N_AGENTS, &n_agents, sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_ALPHA, &alpha, sizeof(float)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_GAMMA, &gamma, sizeof(float)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_EPSILON, &epsilon, sizeof(float)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_EPISODES, &episodes, sizeof(int)));
  // CHECK_CUDA(cudaMemcpyToSymbol(d_MAX_STEPS_PER_EPISODE,
  // &max_steps_per_episode,
  //                            sizeof(int)));

  // Random states
  SimpleCurand *d_randStates;
  CHECK_CUDA(cudaMalloc(&d_randStates, n_agents * sizeof(SimpleCurand)));

  int blocks = (n_agents + threads_per_block - 1) / threads_per_block;
  // init RNG
  auto seed = (unsigned)time(NULL);
  dim3 gridDims(blocks, 1, 1);
  dim3 blockDims(threads_per_block, 1, 1);

  __global__ void initRandKernel(SimpleCurand * randStates, unsigned int seed);
  // redeclare or do it inline? We'll do it inline here to avoid confusion, but
  // let's reuse the same signature: We must define it above or we won't
  // compile. It's above. We'll call it now:

  initRandKernel<<<gridDims, blockDims>>>(d_randStates, seed);
  CHECK_CUDA(cudaDeviceSynchronize());

  int *d_done;
  CHECK_CUDA(cudaMalloc(&d_done, n_agents * sizeof(int)));
  int *d_countActive;
  CHECK_CUDA(cudaMalloc(&d_countActive, sizeof(int)));

  for (int ep = 0; ep < episodes; ep++) {
    resetAgentsKernel<<<gridDims, blockDims>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    int stepCount = 0;
    while (stepCount < max_steps_per_episode) {
      stepCount++;
      cudaMemset(d_done, 0, n_agents * sizeof(int));

      stepAgentsKernel<<<gridDims, blockDims>>>(d_randStates, d_done);
      CHECK_CUDA(cudaDeviceSynchronize());

      CHECK_CUDA(cudaMemset(d_countActive, 0, sizeof(int)));

      countActiveKernel<<<gridDims, blockDims>>>(d_countActive);
      CHECK_CUDA(cudaDeviceSynchronize());

      int h_countActive = 0;
      CHECK_CUDA(cudaMemcpy(&h_countActive, d_countActive, sizeof(int),
                            cudaMemcpyDeviceToHost));

      if (h_countActive == 0)
        break;
      if ((float)h_countActive < 0.2f * (float)n_agents)
        break;
    }
  }

  CHECK_CUDA(cudaFree(d_done));
  CHECK_CUDA(cudaFree(d_countActive));

  CHECK_CUDA(
      cudaMemcpy(h_Q.data(), d_QPtr, qSizeInBytes, cudaMemcpyDeviceToHost));

  printPolicyCPU(h_grid, h_Q, size, flag_x, flag_y);

  CHECK_CUDA(cudaFree(d_randStates));
  CHECK_CUDA(cudaFree(d_agentXPtr));
  CHECK_CUDA(cudaFree(d_agentYPtr));
  CHECK_CUDA(cudaFree(d_activePtr));
  CHECK_CUDA(cudaFree(d_gridPtr));
  CHECK_CUDA(cudaFree(d_QPtr));

  return 0;
}

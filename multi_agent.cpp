#include <cstring>
#include <ctime>
#include <iostream>
#include <tuple>
#include <vector>

// Environment class
class MultiAgentGridWorld {
public:
  MultiAgentGridWorld(int size, int n_mines, std::pair<int, int> flag_pos,
                      int n_agents)
      : size_(size), n_mines_(n_mines), flag_pos_(flag_pos),
        n_agents_(n_agents) {
    grid_.resize(size_, std::vector<int>(size_, 0));
    std::srand((unsigned)std::time(nullptr));
    placeMines();
    agent_states_.resize(n_agents_, std::make_pair(0, 0));
    active_agents_.resize(n_agents_, true);
  }

  std::vector<std::pair<int, int>> reset() {
    for (int i = 0; i < n_agents_; i++) {
      agent_states_[i] = {0, 0};
      active_agents_[i] = true;
    }
    return agent_states_;
  }

  std::tuple<std::pair<int, int>, int, bool> step(int agent_id, int action) {
    if (agent_id >= 0 && !active_agents_[agent_id]) {
      return std::make_tuple(agent_states_[agent_id], 0, true);
    }
    int x = 0, y = 0;
    if (agent_id >= 0) {
      x = agent_states_[agent_id].first;
      y = agent_states_[agent_id].second;
    }
    if (action == 0) {
      x = (x > 0) ? x - 1 : x;
    } else if (action == 1) {
      x = (x < size_ - 1) ? x + 1 : x;
    } else if (action == 2) {
      y = (y > 0) ? y - 1 : y;
    } else if (action == 3) {
      y = (y < size_ - 1) ? y + 1 : y;
    }
    if (agent_id >= 0) {
      agent_states_[agent_id] = {x, y};
    }
    if (agent_id >= 0) {
      if (isMine(x, y)) {
        active_agents_[agent_id] = false;
        return std::make_tuple(agent_states_[agent_id], -10, true);
      } else if (x == flag_pos_.first && y == flag_pos_.second) {
        active_agents_[agent_id] = false;
        return std::make_tuple(agent_states_[agent_id], 10, true);
      } else {
        return std::make_tuple(agent_states_[agent_id], -1, false);
      }
    }
    return std::make_tuple(std::make_pair(x, y), 0, false);
  }

  bool isMine(int x, int y) const { return (grid_[x][y] == -1); }

  int getSize() const { return size_; }

  int getNumAgents() const { return n_agents_; }

  int getActiveCount() const {
    int c = 0;
    for (bool b : active_agents_)
      if (b)
        c++;
    return c;
  }

  std::pair<int, int> getFlagPos() const { return flag_pos_; }

  bool isMineCellPublic(int x, int y) { return isMine(x, y); }

private:
  int size_;
  int n_mines_;
  std::pair<int, int> flag_pos_;
  int n_agents_;
  std::vector<std::vector<int>> grid_;
  std::vector<std::pair<int, int>> agent_states_;
  std::vector<bool> active_agents_;

  void placeMines() {
    int placed = 0;
    while (placed < n_mines_) {
      int rx = std::rand() % size_;
      int ry = std::rand() % size_;
      if ((rx == 0 && ry == 0) ||
          (rx == flag_pos_.first && ry == flag_pos_.second)) {
        continue;
      }
      if (grid_[rx][ry] != -1) {
        grid_[rx][ry] = -1;
        placed++;
      }
    }
  }
};

// Q-learning agent
class MultiAgentQLearningAgent {
public:
  MultiAgentQLearningAgent(MultiAgentGridWorld &env, double alpha = 0.1,
                           double gamma = 0.9, double epsilon = 0.1)
      : env_(env), alpha_(alpha), gamma_(gamma), epsilon_(epsilon) {
    int s = env_.getSize();
    q_table_.resize(
        s, std::vector<std::vector<double>>(s, std::vector<double>(4, 0.0)));
    std::srand((unsigned)std::time(nullptr));
  }

  void train(int episodes) {
    for (int ep = 0; ep < episodes; ep++) {
      auto states = env_.reset();
      std::vector<bool> done_flags(env_.getNumAgents(), false);

      while (true) {
        bool all_done = true;
        int num_active_agents = env_.getActiveCount();
        if (num_active_agents < (int)(0.2 * env_.getNumAgents())) {
          break;
        }
        for (int i = 0; i < (int)done_flags.size(); i++) {
          if (!done_flags[i]) {
            all_done = false;
            int x = states[i].first;
            int y = states[i].second;
            int act = chooseAction(x, y);
            auto stepResult = env_.step(i, act);
            auto nxt = std::get<0>(stepResult);
            int rew = std::get<1>(stepResult);
            bool dn = std::get<2>(stepResult);
            updateQ(x, y, act, rew, nxt);
            states[i] = nxt;
            done_flags[i] = dn;
          }
        }
        if (all_done)
          break;
      }
    }
  }

  void printPolicy() {
    int s = env_.getSize();
    auto flagPos = env_.getFlagPos();
    for (int x = 0; x < s; x++) {
      for (int y = 0; y < s; y++) {
        if (env_.isMineCellPublic(x, y)) {
          std::cout << "M ";
          continue;
        }
        if (x == flagPos.first && y == flagPos.second) {
          std::cout << "F ";
          continue;
        }
        double bestVal = q_table_[x][y][0];
        int bestAct = 0;
        for (int a = 1; a < 4; a++) {
          if (q_table_[x][y][a] > bestVal) {
            bestVal = q_table_[x][y][a];
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

protected:
  MultiAgentGridWorld &env_;
  double alpha_;
  double gamma_;
  double epsilon_;
  std::vector<std::vector<std::vector<double>>> q_table_;

  int chooseAction(int x, int y) {
    double r = (double)std::rand() / RAND_MAX;
    if (r < epsilon_) {
      return std::rand() % 4;
    }
    double bestVal = q_table_[x][y][0];
    int bestAct = 0;
    for (int a = 1; a < 4; a++) {
      if (q_table_[x][y][a] > bestVal) {
        bestVal = q_table_[x][y][a];
        bestAct = a;
      }
    }
    return bestAct;
  }

  void updateQ(int x, int y, int action, int reward,
               std::pair<int, int> nextState) {
    int nx = nextState.first;
    int ny = nextState.second;
    double bestNext = q_table_[nx][ny][0];
    for (int a = 1; a < 4; a++) {
      if (q_table_[nx][ny][a] > bestNext) {
        bestNext = q_table_[nx][ny][a];
      }
    }
    double oldVal = q_table_[x][y][action];
    double tdTarget = reward + gamma_ * bestNext;
    q_table_[x][y][action] = oldVal + alpha_ * (tdTarget - oldVal);
  }

  int bestAction(int x, int y) {
    double bestVal = q_table_[x][y][0];
    int bestAct = 0;
    for (int a = 1; a < 4; a++) {
      if (q_table_[x][y][a] > bestVal) {
        bestVal = q_table_[x][y][a];
        bestAct = a;
      }
    }
    return bestAct;
  }
};

int main(int argc, char **argv) {
  int size = 46;
  int n_mines = 96;
  int flag_x = 45;
  int flag_y = 45;
  int n_agents = 512;
  int episodes = 1000;
  double alpha = 0.1;
  double gamma = 0.9;
  double epsilon = 0.1;
  int maxSteps = 200;

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
    } else if (std::strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
      episodes = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
      alpha = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
      gamma = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
      epsilon = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--max_steps") == 0 && i + 1 < argc) {
      maxSteps = std::atoi(argv[++i]);
    }
  }

  MultiAgentGridWorld env(size, n_mines, {flag_x, flag_y}, n_agents);
  MultiAgentQLearningAgent agent(env, alpha, gamma, epsilon);
  agent.train(episodes);
  agent.printPolicy();

  return 0;
}

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iomanip>
#include <tuple>
#include <Magick++.h>
#include <string>
#include <cstring>

class GridWorld {
public:
    GridWorld(int size, int n_mines, std::pair<int,int> flag_pos)
        : size(size), n_mines(n_mines), flag_pos(flag_pos), state({0,0}) {
        grid.resize(size, std::vector<int>(size, 0));
        std::srand((unsigned)std::time(nullptr));
        placeMines();
    }

    void placeMines() {
        mines.clear();
        while ((int)mines.size() < n_mines) {
            int x = std::rand() % size;
            int y = std::rand() % size;
            std::pair<int,int> pos = {x, y};
            if (std::find(mines.begin(), mines.end(), pos) == mines.end() &&
                (x != 0 || y != 0) &&
                (x != flag_pos.first || y != flag_pos.second)) {
                mines.push_back(pos);
                grid[x][y] = -1;
            }
        }
    }

    std::pair<int,int> reset() {
        state = {0, 0};
        return state;
    }

    std::tuple<std::pair<int,int>, int, bool> step(int action) {
        int x = state.first;
        int y = state.second;
        if (action == 0) {
            x = (x > 0) ? x - 1 : x;
        } else if (action == 1) {
            x = (x < size - 1) ? x + 1 : x;
        } else if (action == 2) {
            y = (y > 0) ? y - 1 : y;
        } else if (action == 3) {
            y = (y < size - 1) ? y + 1 : y;
        }
        state = {x, y};
        for (auto &m : mines) {
            if (m.first == x && m.second == y) {
                return std::make_tuple(state, -10, true);
            }
        }
        if (x == flag_pos.first && y == flag_pos.second) {
            return std::make_tuple(state, 10, true);
        }
        return std::make_tuple(state, -1, false);
    }

    int getSize() const {
        return size;
    }

    bool isMine(int x, int y) const {
        for (auto &m : mines) {
            if (m.first == x && m.second == y) return true;
        }
        return false;
    }

    bool isFlag(int x, int y) const {
        return (x == flag_pos.first && y == flag_pos.second);
    }

private:
    int size;
    int n_mines;
    std::pair<int,int> flag_pos;
    std::vector<std::vector<int>> grid;
    std::vector<std::pair<int,int>> mines;
    std::pair<int,int> state;
};

class QLearningAgent {
public:
    QLearningAgent(GridWorld &env, double alpha=0.1, double gamma=0.9, double epsilon=0.1)
        : env(env), alpha(alpha), gamma(gamma), epsilon(epsilon) {
        int s = env.getSize();
        Q.resize(s, std::vector<std::vector<double>>(s, std::vector<double>(4, 0.0)));
        std::srand((unsigned)std::time(nullptr));
    }

    int chooseAction(const std::pair<int,int> &state) {
        double r = (double)std::rand() / RAND_MAX;
        if (r < epsilon) {
            return std::rand() % 4;
        }
        int x = state.first;
        int y = state.second;
        double bestValue = Q[x][y][0];
        int bestAction = 0;
        for (int a = 1; a < 4; a++) {
            if (Q[x][y][a] > bestValue) {
                bestValue = Q[x][y][a];
                bestAction = a;
            }
        }
        return bestAction;
    }

    void update(const std::pair<int,int> &state, int action, int reward, const std::pair<int,int> &next_state) {
        int x = state.first;
        int y = state.second;
        int nx = next_state.first;
        int ny = next_state.second;
        double bestNext = Q[nx][ny][0];
        for (int a = 1; a < 4; a++) {
            if (Q[nx][ny][a] > bestNext) {
                bestNext = Q[nx][ny][a];
            }
        }
        double tdTarget = reward + gamma * bestNext;
        Q[x][y][action] += alpha * (tdTarget - Q[x][y][action]);
    }

    void train(int episodes) {
        for (int ep = 0; ep < episodes; ep++) {
            auto state = env.reset();
            bool done = false;
            while (!done) {
                int action = chooseAction(state);
                auto stepResult = env.step(action);
                auto next_state = std::get<0>(stepResult);
                int reward = std::get<1>(stepResult);
                done = std::get<2>(stepResult);
                update(state, action, reward, next_state);
                state = next_state;
            }
        }
    }

    int bestAction(int x, int y) {
        double bestVal = Q[x][y][0];
        int bestAct = 0;
        for (int a = 1; a < 4; a++) {
            if (Q[x][y][a] > bestVal) {
                bestVal = Q[x][y][a];
                bestAct = a;
            }
        }
        return bestAct;
    }

    void printPolicy() {
        int s = env.getSize();
        for (int x = 0; x < s; x++) {
            for (int y = 0; y < s; y++) {
                if (env.isMine(x, y)) {
                    std::cout << "M ";
                    continue;
                }
                if (env.isFlag(x, y)) {
                    std::cout << "F ";
                    continue;
                }
                int act = bestAction(x, y);
                if (act == 0) std::cout << "^ ";
                else if (act == 1) std::cout << "v ";
                else if (act == 2) std::cout << "< ";
                else if (act == 3) std::cout << "> ";
            }
            std::cout << std::endl;
        }
    }

    void visualizePolicyGIF(const std::string &filename, int maxSteps=200, int cellSize=20) {
        if(filename.empty()) return;
        Magick::InitializeMagick(nullptr);
        std::vector<Magick::Image> frames;
        auto s = env.getSize();

        auto state = env.reset();
        bool done = false;

        for(int stepCount = 0; stepCount < maxSteps && !done; stepCount++) {
            Magick::Image frame(Magick::Geometry(s * cellSize, s * cellSize), "white");
            frame.magick("GIF");

            for(int x = 0; x < s; x++) {
                for(int y = 0; y < s; y++) {
                    if(env.isMine(x,y)) {
                        for(int px = 0; px < cellSize; px++) {
                            for(int py = 0; py < cellSize; py++) {
                                frame.pixelColor(y*cellSize+py, x*cellSize+px, Magick::Color("red"));
                            }
                        }
                    } else if(env.isFlag(x,y)) {
                        for(int px = 0; px < cellSize; px++) {
                            for(int py = 0; py < cellSize; py++) {
                                frame.pixelColor(y*cellSize+py, x*cellSize+px, Magick::Color("green"));
                            }
                        }
                    } else {
                        for(int px = 0; px < cellSize; px++) {
                            for(int py = 0; py < cellSize; py++) {
                                frame.pixelColor(y*cellSize+py, x*cellSize+px, Magick::Color("white"));
                            }
                        }
                    }
                }
            }

            int ax = state.first;
            int ay = state.second;
            for(int px = 0; px < cellSize; px++) {
                for(int py = 0; py < cellSize; py++) {
                    frame.pixelColor(ay*cellSize+py, ax*cellSize+px, Magick::Color("blue"));
                }
            }

            frame.animationDelay(5);
            frames.push_back(frame);

            int act = bestAction(ax, ay);
            auto result = env.step(act);
            state = std::get<0>(result);
            done = std::get<2>(result);
        }

        Magick::writeImages(frames.begin(), frames.end(), filename);
    }

private:
    GridWorld &env;
    double alpha;
    double gamma;
    double epsilon;
    std::vector<std::vector<std::vector<double>>> Q;
};

int main(int argc, char** argv) {
    int size = 32;
    int n_mines = 40;
    int flag_x = 31;
    int flag_y = 31;
    int episodes = 20000;
    int maxSteps = 200;
    int cellSize = 20;
    std::string gifPath;
    double alpha = 0.1;
    double gamma = 0.9;
    double epsilon = 0.1;

    for(int i = 1; i < argc; i++) {
        if(std::strcmp(argv[i], "--size") == 0 && i+1 < argc) {
            size = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--n_mines") == 0 && i+1 < argc) {
            n_mines = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--flag_x") == 0 && i+1 < argc) {
            flag_x = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--flag_y") == 0 && i+1 < argc) {
            flag_y = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--episodes") == 0 && i+1 < argc) {
            episodes = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--max_steps") == 0 && i+1 < argc) {
            maxSteps = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--cell_size") == 0 && i+1 < argc) {
            cellSize = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--gif_path") == 0 && i+1 < argc) {
            gifPath = argv[++i];
        } else if(std::strcmp(argv[i], "--alpha") == 0 && i+1 < argc) {
            alpha = std::atof(argv[++i]);
        } else if(std::strcmp(argv[i], "--gamma") == 0 && i+1 < argc) {
            gamma = std::atof(argv[++i]);
        } else if(std::strcmp(argv[i], "--epsilon") == 0 && i+1 < argc) {
            epsilon = std::atof(argv[++i]);
        }
    }

    GridWorld env(size, n_mines, {flag_x, flag_y});
    QLearningAgent agent(env, alpha, gamma, epsilon);
    agent.train(episodes);
    agent.printPolicy();
    agent.visualizePolicyGIF(gifPath, maxSteps, cellSize);

    return 0;
}

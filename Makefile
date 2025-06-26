CXX = clang++
NVCC = nvcc
CXXFLAGS = -O3

sc:
	$(CXX) $(CXXFLAGS) ./single_agent.cpp -o single_agent

sg:
	$(NVCC) $(CXXFLAGS) ./single_agent.cu -o single_agent_cu

mc:
	$(CXX) $(CXXFLAGS) ./multi_agent.cpp -o multi_agent

mg:
	$(NVCC) $(CXXFLAGS) ./multi_agent.cu -o multi_agent_cu

clean:
	rm single_agent single_agent_cu multi_agent multi_agent_cu

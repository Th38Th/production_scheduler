PRODUCTION PLANNING USING REINFORCEMENT LEARNING
================================================

The project implements a graph-based deep-Q learning neural network for the purpose of allocating jobs from a
potentially infinite stream to a finite number of machines in the most efficient way possible, ensuring minimal
idle time and maximizing the degree of utilized production capacity.

The jobs and machines are represented as a bipartite graph and the problem the network solves at every given moment
is which edge is the best one to remove (and edge J-M corresponding to allocation of job J to machine M, removal
of the edge meaning the allocation has been added to the production schedule).

The neural network itself is composed of two consecutive FullMessagePassing layers (which process both edge and node
data at the same time), with an embedded layer of type GraphMLP.
The network is plugged into a deep-Q-learning model which uses Experience Replay and delayed
target-network updates.

The custom training environment is built using the OpenAI Gym package and is capable of generating production
scheduling problems for the agent to solve. The action space is discrete (equal to the number of possible allocations
+ 1 for the idle action - doing nothing). Choosing the idle action is penalized so as to discourage it, while choosing
already scheduled allocations is managed by using a mask (which leads to the idle action being chosen, incurring the
same penalty).

The most important research is included in the research/ folder

Packages: tensorflow (+keras), spektral, gym
# Simple Graph
This Gymnasium package implements a simple graph environment
described in Adam Zsolt Wagner's paper on using cross entropy method
for reinforcement learning on (simple) graphs.

### Environments
- `SimpleGraphEnv`: 
         Each action is either 0 or 1, 
         representing the adjacency matrix entry for the edge that is to be described next.
         Each observation consists of:
           a) the upper half of the adjacency matrix, followed by
           b) the one-hot encoding of the edge that is to be described next.
         Intermediate rewards are all zeros,
         with the final reward equal to one, when the complete graph is described.
         Reward wrapper is then needed to compute the actual graph invariant of interest.

### Wrappers
- `GraphInvariant`: 
         A `RewardWrapper` that computes the graph invariant of interest,
         when the simple graph is fully described.

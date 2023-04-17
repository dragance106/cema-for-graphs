# Cross entropy method for constructing graphs
The current project presents a fresh implementation of
Adam Zsolt Wagner's approach for applying 
cross entropy method for constructing graph with specified properties
(usually as counterexamples to conjectures).

`cema_train_simple_graph.py` provides methods: 
- for constructing a new generation of graphs 
based on the currently trained network,
- for learning from selected best graphs, and
- for promoting even narrower selection of best graphs to the next generation.

`training_runner.py` contains the code for computing rewards for constructed graphs
through a jpype connection to graph6java.jar,
a library of java methods for computing with graphs
(usually 3-5 faster than a combination of numpy and networkx).

More details to follow in a forthcoming paper.

Joint work with Salem Al-Yakoub, Ali Kanso and Mohammad Ghebleh.
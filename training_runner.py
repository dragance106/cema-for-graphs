"""
This file provides examples of methods for computing graph rewards
which are then used to train cross entropy method agent for their maximisation.

The expectation is that the maximised graphs will provide new insight
into the structure of actual extremal graphs, and
perhaps lead to new or disprove existing conjectures.

The methods for computing rewards use a combination of jpype and graph6java.jar
to do the actual graph computations in java (mostly spectral invariants),
as that combination is some 3-5 times faster than the numpy/networkx combo.
"""


#################################################################
# JPYPE starts the java virtual machine and makes it possible   #
# to use java methods from graph6java.jar directly from python. #
#################################################################
# import jpype
import jpype.imports
from jpype.types import *

# startJVM method scans the provided class path for jar files and loads them all at once.
# hence you have to specify here the path to graph6java.jar
# if it is in the same directory as this file, then classpath=['*'] is enough.
jpype.startJVM(classpath=['*'], convertStrings=False)
from graph6java import Graph


#################################################################
# Examples of computing rewards by using graph6java.jar methods #
#################################################################
import math
INF = 1000000                   # a very large number used by compute_reward to signify unwanted graphs

def lambda1_plus_mu(n, A):
    """
    Computes the reward by calling the auxiliary methods from graph6java.jar,
    which is 3-5 times quicker than by using corresponding methods from networkx or numpy.

    :param n: the number of graph vertices
    :param A: the adjacency matrix
    :return:  the value of the user-defined graph invariant
    """
    g = Graph(JInt[:,:](A))

    # we do not like disconnected graphs at all...
    if g.numberComponents() > 1:
        return -INF

    # compute the reward
    lambda1 = g.Aspectrum()[n-1]
    mu = g.matchingNumber()
    reward = math.sqrt(n - 1) + 1 - lambda1 - mu

    return reward


############################################################
# Training the cross entropy method agent on above rewards #
############################################################
from cema_train_simple_graph import train

# train(compute_reward=lambda1_plus_mu)         # all optional arguments have default values
# train(compute_reward=lambda1_plus_mu, n=29)     # let's try higher number of vertices
r, A = train(compute_reward=lambda1_plus_mu, output_best_graph_rate=5, num_generations=15)
print(f'reward={r}\nadj.mat=\n{A}')

# when jpype is no longer needed...
jpype.shutdownJVM()

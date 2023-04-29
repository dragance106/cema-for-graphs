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


def energy_minus_2musqrtdelta(n, A):
    """
    Computes the difference between the energy and
    twice the product of the matching number and the square root of the maximum degree.
    """
    g = Graph(JInt[:,:](A))

    # we do not like disconnected graphs...
    if g.numberComponents() > 1:
        return -INF

    # compute the reward
    energy = g.Aenergy()
    mu = g.matchingNumber()
    Delta = max(g.degrees())
    reward = energy - 2*mu*math.sqrt(Delta)

    return reward


def lambda1_nonregular(n, A):
    """
    Computes the difference between Delta and lambda1 of a nonregular graph and
    compares it to the same differences for paths,
    in respect of Oboudi's conjecture in Algebra Discrete Math. 24 (2017), 302-307.
    Adds a term that favors graphs with smaller Delta
    """
    g = Graph(JInt[:,:](A))

    # we do not like disconnected graphs...
    if g.numberComponents() > 1:
        return -INF

    # we also do not like regular graphs...
    degrees = g.degrees()
    Delta = max(degrees)
    if Delta == min(degrees):
        return -INF

    lambda1 = max(g.Aspectrum())
    reward = 4*math.sin(math.pi/(2*n+2))**2 - Delta + lambda1 - (Delta-3)**2

    return reward


def aouchiche_ng(n,A):
    """
    Computes the Aouchiche et al.'s Nordhaus-Gaddum bound.
    """
    g = Graph(JInt[:,:](A))
    h = g.complement()

    lambda1 = g.Aspectrum()[n-1]
    lambda1cpl = h.Aspectrum()[n-1]

    reward = lambda1 + lambda1cpl - (4*n-5)/3
    if n%3 == 0:
        reward += (3*n-1 - math.sqrt(9*n*n - 6*n + 9))/6
    elif n%3 == 1:
        reward += (3*n-2 - math.sqrt(9*n*n - 12*n + 12))/6

    return reward


def ramsey_5_6(n, A):
    """
    Computes the total number of 5-cliques and 5-cocliques in the graph
    and returns its negative value (as CEMA wants to maximise stuff).
    """
    g = Graph(JInt[:,:](A))
    c = g.num5cliques()
    s = g.num6cocliques()
    return -c-s


############################################################
# Training the cross entropy method agent on above rewards #
############################################################
from cema_train_simple_graph import train

# r, A = train(compute_reward=lambda1_plus_mu)         # all optional arguments have default values
# r, A = train(compute_reward=lambda1_plus_mu, n=29)     # let's try higher number of vertices
# r, A = train(compute_reward=energy_minus_2musqrtdelta, n=29)
# r, A = train(compute_reward=ramsey_5_6,
#              n=58,
#              batch_size=200,
#              num_generations=10000,
#              percent_learn=92,
#              percent_survive=94,
#              learning_rate=0.0015,
#              neurons=[192,16],
#              act_rndness_init=0.0005,
#              act_rndness_max=0.0015)
r, A = train(compute_reward=lambda1_nonregular,
             n=40,
             batch_size=400,
             num_generations=5000,
             percent_learn=95,
             percent_survive=97.5,
             neurons=[100,15])
print(f'reward={r}\nadj.mat=\n{A}')

# when jpype is no longer needed...
jpype.shutdownJVM()

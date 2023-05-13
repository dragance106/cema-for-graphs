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
import numpy as np
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
    lambda1 = max(g.Aspectrum())
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

    lambda1 = max(g.Aspectrum())
    lambda1cpl = max(h.Aspectrum())

    reward = lambda1 + lambda1cpl - (4*n-5)/3
    if n%3 == 0:
        reward += (3*n-1 - math.sqrt(9*n*n - 6*n + 9))/6
    elif n%3 == 1:
        reward += (3*n-2 - math.sqrt(9*n*n - 12*n + 12))/6

    return reward


def skrek_popivoda(n, A):
    """
    Computes the expression from AG-GA1 conjecture from The List.
    """
    g = Graph(JInt[:,:](A))

    # we do not like disconnected graphs...
    if g.numberComponents() > 1:
        return -INF

    # otherwise, return precomputed reward
    return g.skrek_popivoda()


def ga_lambda1(n, A):
    """
    Computes 2sqrt(n-1)/n - GA/lambda1^2 for GA-lambda1 conjecture from The List.
    """
    g = Graph(JInt[:,:](A))

    # we still do not like disconnected graphs
    if g.numberComponents() > 1:
        return -INF

    ga = g.ga()
    lambda1 = max(g.Aspectrum())

    return 2.0*math.sqrt(n-1)/n - ga/(lambda1**2)


def ga_lambda1_ra(n, A):
    """
    Computes GA/lambda1^2 - Ra/2 for GA-lambda1-Ra conjecture from The List.
    """
    g = Graph(JInt[:,:](A))

    if g.numberComponents()>1:
        return -INF

    ga = g.ga()
    lambda1 = max(g.Aspectrum())
    ra = g.randic()

    return ga/lambda1**2 - ra/2.0


def frustrating_energy(n, A):
    """
    Computes Delta + delta - energy, but wants to see only non-singular graphs.
    """
    g = Graph(JInt[:,:](A))

    if g.numberComponents()>1:
        return -INF

    # how do I check that the graph is non-singular?
    if g.Asingular():
        return -INF

    degrees = g.degrees()
    Delta = max(degrees)
    delta = min(degrees)
    energy = g.Aenergy()

    return Delta+delta-energy


def ramsey_5_6(n, A):
    """
    Computes the total number of 5-cliques and 5-cocliques in the graph
    and returns its negative value (as CEMA wants to maximise stuff).
    """
    g = Graph(JInt[:,:](A))
    c = g.num5cliques()
    s = g.num6cocliques()
    return -c-s


def ti_graphs(n, A):
    """
    Computes the number of distinct vertex transmissions - n.
    Value of 0 means we have a transmission-irregular graph.
    """
    g = Graph(JInt[:,:](A))

    if g.numberComponents()>1:
        return -INF

    t = g.transmissions()
    return -n + len(np.unique(t))


def iti_graphs(n, A):
    """
    What reward can we make so that we have 0 for an ITI graph
    and a negative value if it is not an ITI graph?
    It is an ITI graph
    if the number of distinct vertex transmission is n
    and
    if the difference between the maximum and the minimum vertex transmission is n-1.
    The number of distinct vertex transmissions is always at most n,
    so one always nonnegative term can be
    (n - the number of distinct vertex transmissions).
    The difference between the maximum and the minimum vertex transmission
    can be either larger or smaller than n-1,
    so another always nonnegative factor can be
    |max transmission - min transmission - n + 1|,
    where |...| denotes the absolute value.
    Hence the negative of the product of these two factors could be our reward,
    if the condition would have been if ... or ...,
    but our condition is actually if ... and ...,
    so we will take the negative of the sum of these two terms.
    """
    g = Graph(JInt[:,:](A))

    if g.numberComponents()>1:
        return -INF

    t = g.transmissions()
    return -(n-len(np.unique(t)))-abs(max(t)-min(t)-n+1)


def soltes_sum(n, A):
    """
    Computes the negative of the sum of absolute values of
    the differences between the Wiener index of the graph and those of its vertex-deleted subgraphs.
    """
    g = Graph(JInt[:,:](A))

    if g.numberComponents()>1:
        return -INF

    return -g.soltesSum()


def auto_lapla_1(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(4*deg*deg*deg/avd))


def auto_lapla_2(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(2*avd*avd/deg)


def auto_lapla_3(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(avd*avd/deg + avd)


def auto_lapla_4(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(2*deg*deg/avd)


def auto_lapla_5(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(deg*deg/avd + avd)


def auto_lapla_6(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(avd*avd + 3*deg*deg))


def auto_lapla_7(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(deg*deg/avd + deg)


def auto_lapla_8(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(deg*avd + 3*deg*deg))


def auto_lapla_9(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max((avd + 3.0*deg)/2.0)


def auto_lapla_10(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(deg*deg + 3*deg*avd))


def auto_lapla_11(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(2.0*avd*avd*avd/(deg*deg))


def auto_lapla_12(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(2.0*deg*deg + 2.0*avd*avd))


def auto_lapla_13(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(2.0*avd*avd*avd*avd/(deg*deg*deg))


def auto_lapla_14(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(2.0*deg*deg*deg/(avd*avd))


def auto_lapla_15(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(4*avd*avd*avd/deg))


def auto_lapla_16(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(2.0*deg*deg*deg*deg/(avd*avd*avd))


def auto_lapla_17(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(np.sqrt(5.0*deg*deg*deg*deg + 11.0*avd*avd*avd*avd)))


def auto_lapla_18(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(2.0*avd*avd*avd/deg + 2.0*deg*deg))


def auto_lapla_19(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(np.sqrt(4.0*deg*deg*deg*deg + 12.0*deg*avd*avd*avd)))


def auto_lapla_20(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(0.5*np.sqrt(7.0*deg*deg + 9.0*avd*avd))


def auto_lapla_21(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(deg*deg*deg/avd + 3.0*avd*avd))


def auto_lapla_22(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(np.sqrt(2.0*deg*deg*deg*deg + 14.0*deg*deg*avd*avd)))


def auto_lapla_23(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(deg*deg + 3.0*deg*avd))


def auto_lapla_24(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(np.sqrt(6.0*deg*deg*deg*deg + 10.0*avd*avd*avd*avd)))


def auto_lapla_25(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(np.sqrt(3.0*deg*deg*deg*deg + 13.0*deg*deg*avd*avd)))


def auto_lapla_26(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(0.5*np.sqrt(5.0*deg*deg + 11.0*deg*avd))


def auto_lapla_27(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(1.5*deg*deg + 2.5*deg*avd))


def auto_lapla_28(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(2.0*avd*avd*avd*avd/(deg*deg) + 2.0*deg*avd))


def auto_lapla_29(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(avd*avd + 3.0*avd*avd*avd/deg))


def auto_lapla_30(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(avd*avd*avd/(deg*deg) + deg*deg/avd)


def auto_lapla_31(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(4.0*avd*avd/(avd+deg))


def auto_lapla_32(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())

    return mu - max(np.sqrt(avd*avd*avd*(avd + 3.0*deg))/deg)


def auto_lapla_33(n, A):
    """
    Computes rewards for edge versions of automated Laplacian bounds.
    """
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2.0*(degrow + degcol) - (avdrow + avdcol)))


def auto_lapla_34(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2.0*(degrow**2 + degcol**2)/(degrow + degcol)))


def auto_lapla_35(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2.0*(degrow**2 + degcol**2)/(avdrow + avdcol)))


def auto_lapla_36(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2.0*(avdrow**2 + avdcol**2)/(degrow + degcol)))


def auto_lapla_37(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    # avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*np.sqrt(2.0*(degrow**2 + degcol**2)))


def auto_lapla_38(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    # avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2.0 + np.sqrt(2.0*(degrow**2+degcol**2)-4.0*(degrow+degcol)+4.0)))


def auto_lapla_39(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 2.0*(degrow**2+degcol**2)-4.0*(avdrow+avdcol)+4.0)
    return mu - np.amax(A*(2.0 + np.sqrt(under)))


def auto_lapla_40(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2+np.sqrt(2*(avdrow-1)**2+2*(avdcol-1)**2+degrow**2+degcol**2-degrow*avdrow-degcol*avdcol)))


def auto_lapla_41(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 2*(degrow**2+degcol**2)-4*(avdrow+avdcol)+4)
    return mu - np.amax(A*(2+avdrow+avdcol-(degrow+degcol)+np.sqrt(under)))


def auto_lapla_42(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*np.sqrt(A*(degrow**2+degcol**2+2*avdrow*avdcol)))


def auto_lapla_43(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 3*(avdrow**2+avdcol**2)-2*avdrow*avdcol-4*(degrow+degcol)+4)
    return mu - np.amax(A*(2+np.sqrt(under)))


def auto_lapla_44(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2+np.sqrt(A*(2*((degrow-1)**2+(degcol-1)**2+avdrow*avdcol-degrow*degcol)))))


def auto_lapla_45(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2+np.sqrt(A*((degrow-degcol)**2+2*degrow*avdrow+2*degcol*avdcol-4*(avdrow+avdcol)+4))))


def auto_lapla_46(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2+np.sqrt(A*(2*(degrow**2+degcol**2)-16*degrow*degcol/(avdrow+avdcol)+4))))


def auto_lapla_47(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*((2*degrow**2+2*degcol**2-(avdrow-avdcol)**2)/(degrow+degcol)))


def auto_lapla_48(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 2*degrow**2+2*degcol**2-4*avdrow-4*avdcol+4.0)
    return mu - np.amax(A*((2*degrow**2+2*degcol**2)/(2+np.sqrt(under))))


def auto_lapla_49(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 2*avdrow**2+2*avdcol**2+(degrow-degcol)**2-4*(degrow+degcol)+4)
    return mu - np.amax(A*(2+np.sqrt(A*under)))


def auto_lapla_50(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2*(degrow**2+degcol**2+avdrow*avdcol-degrow*degcol)/(degrow+degcol)))


def auto_lapla_51(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(2*(avdrow+avdcol)-4*avdrow*avdcol/(degrow+degcol)))


def auto_lapla_52(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 8*avdrow**4+8*avdcol**4-8*degrow**2-8*degcol**2+4)
    under2 = np.maximum(zeros, np.sqrt(under)-4*(degrow+degcol)+6)
    return mu - np.amax(A*(2+np.sqrt(under2)))


def auto_lapla_53(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 8*avdrow**4+8*avdcol**4-8*degrow*avdrow-8*degcol*avdcol+4)
    under2 = np.maximum(zeros, np.sqrt(under)-4*(degrow+degcol)+6)
    return mu - np.amax(A*(2+np.sqrt(under2)))


def auto_lapla_54(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 2*avdrow**2+2*avdcol**2+degrow*avdrow+degcol*avdcol-degrow**2-degcol**2-4*(degrow+degcol)+4)
    return mu - np.amax(A*(2+np.sqrt(under)))


def auto_lapla_55(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 3*avdrow**2+3*avdcol**2-degrow**2-degcol**2-4*avdrow-4*avdcol+4)
    return mu - np.amax(A*(2+np.sqrt(under)))


def auto_lapla_56(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*((degrow**2+degcol**2)*(avdrow+avdcol)/(2*degrow*degcol)))


def auto_lapla_57(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 2*avdrow**2+2*avdcol**2-8*(degrow**2+degcol**2)/(avdrow+avdcol)+4)
    return mu - np.amax(A*(2+np.sqrt(under)))


def auto_lapla_58(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 2*avdrow**2+2*avdrow*avdcol+2*avdcol**2-degrow*avdrow-degcol*avdcol-4*(degrow+degcol)+4)
    return mu - np.amax(A*(2+np.sqrt(under)))


def auto_lapla_59(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*((2*avdrow**2+2*avdrow*avdcol+2*avdcol**2-degrow**2-degcol**2)/(avdrow+avdcol)))


def auto_lapla_60(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, 2*avdrow**2+2*avdrow*avdcol+2*avdcol**2-degrow**2-degcol**2-4*degrow-4*degcol+4)
    return mu - np.amax(A*(2+np.sqrt(under)))


def auto_lapla_61(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*((2*avdrow**2+2*avdcol**2)/(2+np.sqrt(A*(2*(degrow-1)**2+2*(degcol-1)**2)))))


def auto_lapla_62(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, avdrow**2+4*avdrow*avdcol+avdcol**2-2*degrow*degcol-4*(degrow+degcol)+4)
    return mu - np.amax(A*(2+np.sqrt(under)))


def auto_lapla_63(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(degrow+degcol+avdrow+avdcol-4*degrow*degcol/(avdrow+avdcol)))


def auto_lapla_64(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*(avdrow*avdcol*(degrow+degcol)/(degrow*degcol)))


def auto_lapla_65(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*((avdrow+avdcol)*(degrow*avdrow+degcol*avdcol)/(2*avdrow*avdcol)))


def auto_lapla_66(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*((avdrow**2+4*avdrow*avdcol+avdcol**2-degrow*avdrow-degcol*avdcol)/(degrow+degcol)))


def auto_lapla_67(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    return mu - np.amax(A*((avdrow+avdcol)*(degrow*avdrow+degcol*avdcol)/(2*degrow*degcol)))


def auto_lapla_68(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    mu = max(g.Lspectrum())
    deg = np.array(g.degrees())
    avd = np.array(g.averageDegrees())
    degrow, degcol = np.meshgrid(deg, deg, indexing='ij')
    avdrow, avdcol = np.meshgrid(avd, avd, indexing='ij')

    zeros = np.zeros((n,n))
    under = np.maximum(zeros, (avdrow-avdcol)**2+4*degrow*degcol-4*(avdrow+avdcol)+4)
    return mu - np.amax(A*(2+np.sqrt(under)))


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
# r, A = train(compute_reward=lambda1_nonregular,
#              n=40,
#              batch_size=400,
#              num_generations=5000,
#              percent_learn=95,
#              percent_survive=97.5,
#              neurons=[100,15])
# r, A = train(compute_reward=skrek_popivoda, n=35, num_generations=10000)
# r, A = train(compute_reward=ga_lambda1_ra, n=30, num_generations=5000)
# r, A = train(compute_reward=ga_lambda1,
#              n=60,
#              percent_learn=50,
#              percent_survive=99,
#              num_generations=2000)
# r, A = train(compute_reward=frustrating_energy, n=39, num_generations=10000)
# r, A = train(compute_reward=ti_graphs)
# r, A = train(compute_reward=iti_graphs, n=25)
# r, A = train(compute_reward=soltes_sum)
# r, A = train(compute_reward=auto_lapla_6,
#              n=21,
#              percent_learn=90,
#              percent_survive=98,
#              batch_size=500,
#              act_rndness_max=0.1,
#              num_generations=5000)
r, A = train(compute_reward=auto_lapla_68)
print(f'reward={r}\nadj.mat=\n{A}')

# when jpype is no longer needed...
jpype.shutdownJVM()

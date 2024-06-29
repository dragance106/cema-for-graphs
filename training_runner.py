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
import networkx as nx
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

def triangle_conflict(n, A):
    B = np.matmul(A, A)
    C = np.matmul(A, B)
    r = 0

    for i in range(n):
        for j in range(i+1, n):
            if B[i,i]!=B[j,j]:
                r += 3
            if C[i,i]==C[j,j]:
                r += 1

    return -r


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

    IMPORTANT TO NOTE: Akbari and Hosseinzadeh have a new strengthened conjecture
    that the energy of a non-singular graph is at least n - 1 + 2m/n,
    except for two counterexamples of order 4.
    CHECK OUT: Li, Xueliang; Wang, Zhiqian
    Validity of Akbari’s energy conjecture for threshold graphs. (English) Zbl 1487.05165
    Bull. Malays. Math. Sci. Soc. (2) 45, No. 3, 991-1002 (2022).
    AND TEST IT OUT HERE AS WELL!
    """
    g = Graph(JInt[:,:](A))

    if g.numberComponents()>1:
        return -INF

    if g.Asingular():
        return -INF

    # additionally, we want to avoid all the corollaries that imply the conjecture
    m = g.m()
    degrees = g.degrees()
    Delta = max(degrees)
    delta = min(degrees)
    energy = g.Aenergy()
    lambda1 = max(g.Aspectrum())
    detA = np.prod(np.array(g.Aspectrum()))

    if n>=Delta+delta:
        return -INF

    if abs(detA)>=lambda1:
        return -INF

    if 2*m+n*(n-1)>=(Delta+delta)**2:
        return -INF

    if lambda1 - math.log(lambda1)>=delta:
        return -INF

    if Delta<=math.pow(n-1, 1-1.0/n):
        return -INF

    return Delta+delta-energy


def frustrating_energy2(n, A):
    """
    Computes n-1 + 2m/n - energy, but wants to see only non-singular graphs.

    IMPORTANT TO NOTE: Akbari and Hosseinzadeh have a new strengthened conjecture
    that the energy of a non-singular graph is at least n - 1 + 2m/n,
    except for two counterexamples of order 4.
    CHECK OUT: Li, Xueliang; Wang, Zhiqian
    Validity of Akbari’s energy conjecture for threshold graphs. (English) Zbl 1487.05165
    Bull. Malays. Math. Sci. Soc. (2) 45, No. 3, 991-1002 (2022).
    """
    g = Graph(JInt[:,:](A))

    if g.numberComponents()>1:
        return -INF

    if g.Asingular():
        return -INF

    # additionally, we want to avoid all the corollaries that imply the conjecture
    m = g.m()
    energy = g.Aenergy()

    return n-1 + (2.0*m)/n - energy


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

def brouwer(n, A):
    """
    Learning with default parameters very quickly leads to complete graphs.
    Slower learning with smaller percent_learn and larger percent_survive
    leads to complete split graphs as the extremal graphs with zero reward.
    """
    g = Graph(JInt[:,:](A))
    m = g.m()
    mu = np.flip(np.array(g.Lspectrum()))
    br = np.zeros(n)
    br[0] = mu[0] - m - 1
    for k in range(1,n):
        br[k] = br[k-1] + mu[k] - (k+1)
    return np.max(br)

def elphick(n, A):
    """
    Elphick-Farber-Goldberg-Wocjan conjecture states that
    the sum of squares of positive A-eigenvalues is at least n-1 in connected graphs,
    and the same for the sum of squares of negative A-eigenvalues.

    splus is the sum of squares of positive A-eigenvalues
    sminus is the sum of squares of negative A-eigenvalues

    Returns the reward n-1 - min(splus, sminus).

    It appears that the maximal reward (0) will be attained for paths, actually!
    """
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    Aeig = g.Aspectrum()
    splus = 0
    sminus = 0
    for eig in Aeig:
        if eig>0:
            splus = splus + eig*eig
        if eig<0:
            sminus = sminus + eig*eig

    return n-1-min(splus, sminus)


import networkx as nx
def lambda2clique(n, A):
    """
    Bollobas-Nikiforov conjectured that lambda1**2 + lambda2**2 <= 2(1 - 1/clique)*|E|.
    Elphick-Linz-Wocjan strengthened it to sum_{i=1}^L lambda_i**2 <= 2(1 - 1/clique)*|E|,
    where L = min(clique number, num of positive eigenvalues).

    Returns the reward sum_{i=1}^L lambda_i**2 - 2(1-1/clique)|E|,
    where the clique number is computed by networkX

    Identifies complete graphs as the ones with the maximum reward
    """
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    gnx = nx.from_numpy_array(A)
    _, cliq = nx.max_weight_clique(gnx, weight=None)

    Aeig = np.flip(np.array(g.Aspectrum()))
    pos_eig = 0
    for eig in Aeig:
        if eig>0:
            pos_eig = pos_eig + 1

    L = min(cliq, pos_eig)
    splus = 0
    for i in range(L):
        splus = splus + Aeig[i]**2

    return splus - 2*g.m()*(cliq-1)/cliq

def energy_independence(n, A):
    """
    Graffiti made a conjecture that energy/2 >= n - independence_number,

    Returns the reward 2*(n - independence_number) - energy.

    It appears that two copies of K_(n/2) connected by an edge will be extremal,
    with the reward tending to zero.
    """
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    energy = g.Aenergy()

    gnx = nx.complement(nx.from_numpy_array(A))
    _, ind = nx.max_weight_clique(gnx, weight=None)

    return 2*(n - ind) - energy

def spectral_gap(n, A):
    """
    Stanic conjectured that among connected graphs on n vertices,
    the minimum spectral gap is attained by some double kite graph
    (two complete graphs of the same size connected by a path).

    Returns reward lambda_2 - lambda_1 since RL is a maximizer.
    """
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    Aeig = np.flip(np.array(g.Aspectrum()))
    return Aeig[1] - Aeig[0]

def powers3(n, A):
    """
    Powers conjectured that lambda_3 <= floor(n/3).

    Returns the reward lambda_3 - floor(n/3).

    It appears cyclically connected copies of complete graphs will have maximum reward.
    """
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    Aeig = np.flip(np.array(g.Aspectrum()))
    return Aeig[2] - math.floor(n/3.0)

def powers4(n, A):
    g = Graph(JInt[:,:](A))
    if g.numberComponents()>1:
        return -INF

    Aeig = np.flip(np.array(g.Aspectrum()))
    return Aeig[3] - math.floor(n/4.0)

def sombor_nordhaus_gaddum(n, A):
    g = Graph(JInt[:,:](A))
    h = g.complement()

    return -(g.sombor()+h.sombor())/(n*(n-1)*(n-1))


############################################################
# Training the cross entropy method agent on above rewards #
############################################################
if __name__=="__main__":
    from cema_train_simple_graph import train

    for n in range(21, 25):
        r, A = train(compute_reward=sombor_nordhaus_gaddum,
                     n=n,
                     num_generations=10000)

    # r, A = train(compute_reward=brouwer,
    #              n=29,
    #              percent_learn=75,
    #              percent_survive=97.5,
    #              # act_rndness_wait=5,
    #              # act_rndness_max=0.05,
    #              num_generations=10000)

#    r, _ = train(compute_reward=triangle_conflict,
#                 num_generations=10000)

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

# r, A = train(compute_reward=ti_graphs)
# r, A = train(compute_reward=iti_graphs, n=23)
# r, A = train(compute_reward=soltes_sum)

# when jpype is no longer needed...
jpype.shutdownJVM()

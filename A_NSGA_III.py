#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 09:08
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : A_NSGA_III.py
# @Statement : Adaptive approach to NSGA-III
# @Reference : Jain H, Deb K. An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II: Handling constraints and extending to an adaptive approach[J]. IEEE Transactions on Evolutionary Computation, 2013, 18(4): 602-622.
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from scipy.linalg import LinAlgError
from scipy.spatial.distance import cdist


def cal_obj(pop, nobj=3):
    # DTLZ5
    g = np.sum((pop[:, nobj - 1:] - 0.5) ** 2, axis=1)
    temp = np.tile(g.reshape((g.shape[0], 1)), (1, nobj - 2))
    pop[:, 1: nobj - 1] = (1 + 2 * temp * pop[:, 1: nobj - 1]) / (2 + 2 * temp)
    temp1 = np.concatenate((np.ones((g.shape[0], 1)), np.cos(pop[:, : nobj - 1] * np.pi / 2)), axis=1)
    temp2 = np.concatenate((np.ones((g.shape[0], 1)), np.sin(pop[:, np.arange(nobj - 2, -1, -1)] * np.pi / 2)), axis=1)
    return np.tile((1 + g).reshape(g.shape[0], 1), (1, nobj)) * np.fliplr(np.cumprod(temp1, axis=1)) * temp2


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from an n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, nvar):
    # calculate approximately npop uniformly distributed reference points on nvar dimensions
    h1 = 0
    while combination(h1 + nvar, nvar - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < nvar:
        h2 = 0
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * nvar)
            points = np.concatenate((points, temp_points), axis=0)
    return points


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    return rank


def selection(pop, npop, nvar):
    # select the mating pool
    ind = np.random.randint(0, pop.shape[0], npop)
    mating_pool = pop[ind]
    if npop % 2 == 1:
        mating_pool = np.concatenate((mating_pool, mating_pool[0].reshape(1, nvar)), axis=0)
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > 1, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < 1 / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def GAoperation(pop, npop, nvar, nobj, lb, ub, eta_c, eta_m):
    # genetic algorithm (GA) operation
    mating_pool = selection(pop, npop, nvar)
    off = crossover(mating_pool, lb, ub, eta_c)
    off = mutation(off, lb, ub, eta_m)
    off_objs = cal_obj(off, nobj)
    return off, off_objs


def environmental_selection(pop, objs, zmin, num, V):
    # NSGA-III environmental selection
    # Step 1. ND sort
    (nref, nobj) = V.shape
    rank = nd_sort(objs)
    ind = 1
    selected = np.full(pop.shape[0], False)
    while np.sum(selected) + np.sum(rank == ind) <= num:
        selected[rank == ind] = True
        ind += 1
    K = num - np.sum(selected)
    if K == 0:
        return pop[selected], objs[selected]

    # Step 2. select the remaining K solutions
    objs1 = objs[selected]
    objs2 = objs[rank == ind]
    npop1 = objs1.shape[0]
    npop2 = objs2.shape[0]
    t_objs = np.concatenate((objs1, objs2), axis=0) - zmin

    # Step 3. Find extreme points
    extreme = np.zeros(nobj)
    w = 1e-6 + np.eye(nobj)
    for i in range(nobj):
        extreme[i] = np.argmin(np.max(t_objs / w[i], axis=1))

    # Step 4. Calculate intercepts
    try:
        hyperplane = np.matmul(np.linalg.inv(t_objs[extreme.astype(int)]), np.ones((nobj, 1)))
        if np.any(hyperplane == 0):
            a = np.max(t_objs, axis=0)
        else:
            a = 1 / hyperplane
    except LinAlgError:
        a = np.max(t_objs, axis=0)
    t_objs /= a.reshape((1, nobj))

    # Step 5. Association
    cosine = 1 - cdist(t_objs, V, 'cosine')
    distance = np.sqrt(np.sum(t_objs ** 2, axis=1).reshape(npop1 + npop2, 1)) * np.sqrt(1 - cosine ** 2)
    dis = np.min(distance, axis=1)
    association = np.argmin(distance, axis=1)
    temp_rho = dict(Counter(association[: npop1]))
    rho = np.zeros(nref)
    for key in temp_rho.keys():
        rho[key] = temp_rho[key]

    # Step 6. Selection
    choose = np.full(npop2, False)
    v_choose = np.full(nref, True)
    while np.sum(choose) < K:
        temp = np.where(v_choose)[0]
        jmin = np.where(rho[temp] == np.min(rho[temp]))[0]
        j = temp[np.random.choice(jmin)]
        I = np.where(np.bitwise_and(~choose, association[npop1:] == j))[0]
        if I.size > 0:
            if rho[j] == 0:
                s = np.argmin(dis[npop1 + I])
            else:
                s = np.random.randint(I.size)
            choose[I[s]] = True
            rho[j] += 1
        else:
            v_choose[j] = False
    last = np.where(rank == ind)[0]
    selected[last[choose]] = True
    return pop[selected], objs[selected]


def associate(objs, V):
    # associate each solution with one reference point
    norm = np.sqrt(np.sum(objs ** 2, axis=1)).reshape(objs.shape[0], 1)
    cosine = 1 - cdist(objs, V, 'cosine')
    dis = norm * np.sqrt(1 - cosine ** 2)
    association = np.argmin(dis, axis=1)
    temp_rho = dict(Counter(association))
    rho = np.zeros(V.shape[0])
    for key in temp_rho.keys():
        rho[key] = temp_rho[key]
    return rho


def adapt_ref(objs, V, num, interval):
    # add and delete reference points
    # Addition
    (nref, nobj) = V.shape
    ind = np.arange(nobj)
    rho = associate(objs, V)
    old_V = np.random.random((nref, nobj))
    while np.any(rho >= 2) and not np.array_equal(V, old_V):
        old_V = V.copy()
        for i in np.where(rho >= 2)[0]:
            p = np.tile(V[i], (nobj, 1)) - interval / nobj
            p[ind, ind] += interval
            V = np.concatenate((V, p), axis=0)
        V = np.delete(V, np.any(V < 0, axis=1), axis=0)
        index = np.unique(np.round(V * 1e4) / 1e4, axis=0, return_index=True)[1]
        V = V[np.sort(index)]
        rho = associate(objs, V)

    # Deletion
    temp_ind = np.arange(num, V.shape[0])
    index = np.intersect1d(temp_ind, np.where(rho == 0)[0])
    return np.delete(V, index, axis=0)


def main(npop, iter, lb, ub, nobj=3, eta_c=30, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space (default = 3)
    :param eta_c: spread factor distribution index (default = 30)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    V = reference_points(npop, nobj)  # reference vectors
    V = V[np.argsort(V[:, 0])]
    interval = V[0, -1] - V[1, -1]  # the distance between two consecutive reference points
    npop = V.shape[0]
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives
    zmin = np.min(objs, axis=0)  # ideal points

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 100 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        off, off_objs = GAoperation(pop, npop, nvar, nobj, lb, ub, eta_c, eta_m)
        zmin = np.min((zmin, np.min(off_objs, axis=0)), axis=0)

        # Step 2.2. Environmental selection
        pop, objs = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), zmin, npop, V)

        # Step 2.3. Add and delete reference points
        V = adapt_ref(objs, V, npop, interval)

    # Step 3. Sort the results
    rank = nd_sort(objs)
    pf = objs[rank == 1]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ5')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(105, 500, np.array([0] * 12), np.array([1] * 12))

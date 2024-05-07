# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:43:30 2022

Define generate data functions and read data functions

@author: dongh
"""

import numpy as np
import pandas as pd

def finite_zeta(n, alpha):
    temp = np.arange(1, n+1).astype(float)
    return sum(np.power(temp, -alpha))

def generate_zipf_array(alpha, size, max_size, total_sum, permutate=True, noise=0.1):
    if permutate:
        m = np.random.permutation(range(1, size+1)).astype(float)
    else:
        m = np.arange(1, size+1).astype(float)
    m = np.power(m, -alpha)*(1+np.random.normal(0, noise, size))
    zeta = sum(m)
    if size==max_size:
        return total_sum/zeta*m
    else:
        res = np.zeros(max_size)
        index = np.random.choice(np.arange(max_size), size, replace=False)
        res[index] = total_sum/zeta*m
        return res

def generate_random_array(size, max_size, total_sum, distribution='poisson'):
    if size==max_size:
        if distribution=='uniform':
            t = np.random.uniform(0, 1, size)
        elif distribution == 'poisson':
            t = np.random.poisson(20, size)+np.random.uniform(0, 1, size)
    else:
        if distribution=='uniform':
            t = np.zeros(max_size)
            index = np.random.choice(np.arange(max_size), size, replace=False)
            t[index] = np.random.uniform(0, 1, size)
        elif distribution == 'poisson':
            t = np.zeros(max_size)
            index = np.random.choice(np.arange(max_size), size, replace=False)
            t[index] = np.random.poisson(20, size)+np.random.uniform(0, 1, size)
    s = sum(t)
    return total_sum/s*t

def create_data(source, local_number, distribution='poisson', amp=1, permutate=True, noise=0.1):
    """
    Parameters
    ----------
    source : np array/list: source of global score.
    local_number : int: number of local databases
    distribution : choose in 'poisson, zipf, random poiss, random zipf'. The default is 'poisson'.
    amp: amplify number, the base skewness of individual distribution in case distribution ='zipf'/'random zipf', default=0.7
    permutate: whether the array of local scores of one item is permutated or not
    Returns
    -------
    numpy array: data
    """
    a = []
    count = 0
    for item in source:
        count+=1
        if distribution == 'poisson':
            a.append(generate_random_array(size=local_number, max_size=local_number, total_sum=item))
        elif distribution == 'zipf':
            alpha = max(0.1, np.random.normal(camp(count, amp), camp(count, amp)/4))
            a.append(generate_zipf_array(alpha, size=local_number, max_size=local_number, total_sum=item, noise=noise))
        elif distribution == 'random poiss':
            size = np.random.randint(1, local_number)
            a.append(generate_random_array(size=size, max_size=local_number, total_sum=item))
        elif distribution == 'random zipf':
            alpha = max(0.1, np.random.normal(camp(count, amp), camp(count, amp)/4))
            size = np.random.randint(1, local_number)
            a.append(generate_zipf_array(alpha, size=size, max_size=local_number, total_sum=item, noise=noise))
    return np.array(a).reshape(len(source), local_number)

def correlate_permutation(cor, n):
    """
    """
    c = int(n*cor)
    N = n+2*c
    result = np.arange(N)
    C = np.ones(N, dtype='bool')    
    for i in range(n):
        pos = np.random.randint(i, i+2*c)
        if C[pos]:
            result[pos] = i
            C[pos] = False
        else:
            temp1 = result[i:pos][C[i:pos]]
            temp2 = result[pos:i+2*c][C[pos:i+2*c]]
            if len(temp1)==0:
                left = -np.inf
            else:
                left = temp1[-1]
            if len(temp2) == 0:
                right = np.inf
            else:
                right = temp2[0]
            if pos-left <= right-pos:
                pos=left
                result[pos] = i
                C[pos] = False
            else:
                pos=right
                result[pos] = i
                C[pos] = False
    return result[~C]

def fast_correlate_permutation(cor, n):
    """
    """
    c = int(n*cor)
    result = pd.Series(np.arange(n) + np.random.randint(0, 2*c, n))
    return result.sort_values().index

def camp(t, x):
    return x
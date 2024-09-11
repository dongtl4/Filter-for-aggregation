"""
Created on Sun Aprl 7 18:52:51 2024
For convenient, the index in computing is np.arange(n)

@author: vandong
"""

# from machine import local_nodes as node
import numpy as np
import pandas as pd
from rbloom import Bloom
import scipy
import time
# import gc 

# Filters
# following func will return the list of collected items/samples by specific condition
def value_filter(data, index, value, sort=True):
    """
    data, index: np array
    value: float
    """
    if sort==False:
        outdata = data[data >= value]
        outindex = index[data >= value]
    else:
        l = len(data)
        low = l-np.searchsorted(data[::-1], value)
        outdata = data[:low]
        outindex = index[:low]
    return outindex, outdata

def position_filter(data, index, items):
    """
    data, index: np array
    items: array like
    """
    if len(np.where(np.isin(index, items))[0]) == 0:
        pos = 0
    else:
        pos = max(np.where(np.isin(index, items))[0])
    outindex = index[:pos]
    outdata = data[:pos]
    return outindex, outdata
    
def hash_filter(index, size):
    """
    index: np array
    size: int
    """
    hashindex = np.array([hash(x) for x in index])%size
    outbloom = np.zeros(size, dtype='bool')
    outbloom[hashindex] = True
    return outbloom

def Bernoulli_filter(data, index, lbda):
    """
    data, index: np array
    k: int
    lbda, delta: float in (0, 1)
    """
    sample = np.random.binomial(data.astype(int), lbda)
    return index[sample>0], sample[sample > 0]
    

def Exp_filter(data, index, k, delta=0, tau=1):
    """
    data, index: np array (data > 0)
    delta: float in [0, 1)
    tau: float
    """
    sample = np.random.exponential(1/data)
    if delta==0:
        argmin = np.argpartition(sample, k)[:k]
        return index[argmin], sample[argmin]
    else:
        threshold = -np.log(delta)/tau
        return index[sample < threshold], sample[sample < threshold]

def Pois_filter(data, index, lbda):
    """
    data, index: np array (data > 0)
    delta, lbda: float in (0, 1)
    tau: float
    """
    sample = np.random.poisson(data*lbda)
    return index[sample>0], sample[sample>0]

def solve_lambda(delta, alpha):
    alpha1 = scipy.special.gammaincinv(alpha, 1-delta/2)+1e-5
    a0 = np.ceil(2*alpha1)
    a1 = np.ceil(alpha1/scipy.special.gammaincinv(a0, delta/2)*a0)
    while np.abs(a1-a0) > 1:
        a0=a1
        a1 = np.ceil(alpha1/scipy.special.gammaincinv(a0, delta/2)*a0)
    return max(a0, a1)

def evaluate(alg, nodes, k, output, cct, gs, centernet = 10e9, epsilon=0, refresh=True):
    """
    Evaluating the result
    """
    # accuracy
    er = len(set(gs[:k].index)-set(output[:k].index))
    mse =  max(gs[:k].values/gs[output.index[:k]].values-1)
    # bandwidth
    rbw = sum([node.res_bw for node in nodes])*544/514 + 40
    vbw = (sum([node.val_bw for node in nodes]) + sum(nodes[0].received_message.values()))*544/514
    tbw = rbw + vbw
    # total time
    round_time_comp = {}
    round_time_delay = {}
    round_time_comm = {}
    for i in range(1,nodes[0].nbrounds+1):
        round_time_comp[i] = max([node.compute_time.get(i, 0) for node in nodes]) # compute time of local nodes
        round_time_comm[i] = max([node.received_message.get(i, 0)*544/514/node.downspeed/1e6 for node in nodes]) # receiving messages time
        round_time_comm[i] += max(max([(node.ID_sent.get(i, 0)*8+node.extended.get(i,0)+node.score_sent.get(i, 0)*8)*544/514*8/node.upspeed/1e6 for node in nodes]), sum([(node.ID_sent.get(i, 0)*8+node.extended.get(i,0)+node.score_sent.get(i, 0)*8)*544/514*8 for node in nodes])/centernet) # upload information time
        round_time_delay[i] = max([np.random.poisson(100)/100*node.ping*2/1000 for node in nodes]) # ping
    # To have a simple code and be able to run multiple algorithms parallelly, I using single thread numpy for all calculation of the query initiator
    # The assumption here is that the query center have 8x(CPU & RAM). Thus the 'cct' is divided by 8.
    comp_time = sum(cct.values())/8 + sum(round_time_comp.values())
    comm_time = sum(round_time_comm.values())
    delay_time = sum(round_time_delay.values())
    final = comp_time + comm_time + delay_time
    if refresh:
        for i in nodes:
            i.refresh()
    return [alg, er, mse, rbw, vbw, tbw, comp_time, comm_time, delay_time, final]

def TPUT(nodes, n, k, a=1):
    """
    Three phase Uniform threshold
    """
    # phase 1 (initial message + first response)
    temp_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].res_bw += len(considered[0])*8 + len(considered[0])*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    tau1=np.partition(temp_score, -k)[-k]*a
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 2
    message2 = []
    for i in range(len(nodes)):
        nodes[i].threshold = tau1/len(nodes)
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 8
        start=time.time()
        ind, dat = value_filter(nodes[i].darray[k:], nodes[i].dindex[k:], nodes[i].threshold)
        nodes[i].cursor = k + len(ind)
        end=time.time()
        message2.append([ind, dat])
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].res_bw += len(ind)*8 + len(ind)*8
    miss = np.ones(n)*tau1
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message2[i][0]]+=message2[i][1]
        miss[message[i][0]]-=nodes[i].threshold
        miss[message2[i][0]]-=nodes[i].threshold
    tau2 = np.partition(temp_score, -k)[-k]
    best = temp_score + miss +1e-5
    final = np.where(best>tau2)[0]
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 3
    message3=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
        start=time.time()
        ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final))[0]            
        message3.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].val_bw += len(ind)*8 + len(ind)*8
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
    temp = np.argsort(temp_score)[::-1][:k]
    collected = pd.Series(temp_score[temp], temp)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    return collected, cct        
    
def KLEE(nodes, n, k, a=1): # value + popular filter
    """
    KLEE4
    """
    # phase 1 (initial message + first response)
    temp_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].res_bw += len(considered[0])*8 + len(considered[0])*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    tau1=np.partition(temp_score, -k)[-k]*a
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 2
    message2 = []
    for i in range(len(nodes)):
        nodes[i].threshold = tau1/len(nodes)
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 16
        start=time.time()
        temp, _ = value_filter(nodes[i].darray[k:], nodes[i].dindex[k:], nodes[i].threshold)
        message2.append(len(temp))
        nodes[i].cursor = k + len(temp)
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].extended[nodes[i].nbrounds] = 8
        nodes[i].res_bw += 8
    b = max(message2)
    d = int(b/-np.log(1-0.06))+8
    message2=[]
    for i in range(len(nodes)):
        nodes[i].received_message[nodes[i].nbrounds] += 8
        start=time.time()
        bloom = hash_filter(nodes[i].dindex[k:nodes[i].cursor], d)
        message2.append(bloom)
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] += end-start
        nodes[i].extended[nodes[i].nbrounds] += d/8
        nodes[i].res_bw += d/8
    popular = np.zeros(d, dtype='int')
    start=time.time()
    for i in range(len(nodes)):
        popular += message2[i]
    R = np.partition(popular, -k)[-k]
    popular = popular >= int(R*0.8)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 3
    message3 = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = d/8
        start=time.time()
        check = np.array([hash(x) for x in nodes[i].dindex[k:nodes[i].cursor]])%d
        ind = nodes[i].dindex[k:nodes[i].cursor][popular[check] == True]
        dat = nodes[i].darray[k:nodes[i].cursor][popular[check] == True]
        message3.append([ind, dat])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].val_bw += len(ind)*8 + len(ind)*8
    miss = np.ones(n)*tau1
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
        miss[message[i][0]]-=nodes[i].threshold
        miss[message3[i][0]]-=nodes[i].threshold
    tau2 = np.partition(temp_score, -k)[-k]
    best = temp_score + miss +1e-5
    final = np.where(best>tau2)[0]
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    message3=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
        start=time.time()
        ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final))[0]            
        message3.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].val_bw += len(ind)*8 + len(ind)*8
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
    temp = np.argsort(temp_score)[::-1][:k]
    collected = pd.Series(temp_score[temp], temp)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    return collected, cct
        
def TPOR(nodes, n, k):
    """
    TPOR
    """
    # phase 1 (initial message + first response)
    temp_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].res_bw += len(considered[0])*8 + len(considered[0])*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    initial_cand = np.argpartition(temp_score, -k)[-k:]
    end=time.time()
    cct[nodes[0].nbrounds]=end-start  
    
    # phase 2
    message2 = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = k*8
        start = time.time()
        ind, dat = position_filter(nodes[i].darray[k:], nodes[i].dindex[k:], initial_cand)
        nodes[i].cursor = k+len(ind)
        end=time.time()
        message2.append([ind, dat])
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].res_bw += len(ind)*8 + len(ind)*8
    start=time.time()
    miss = np.ones(n, dtype='float')*sum([x.darray[x.cursor] for x in nodes])
    for i in range(len(nodes)):
        temp_score[message2[i][0]]+=message2[i][1]
        miss[message[i][0]]-=nodes[i].darray[nodes[i].cursor]
        miss[message2[i][0]]-=nodes[i].darray[nodes[i].cursor]
    tau2 = np.partition(temp_score, -k)[-k]
    best = temp_score + miss +1e-5
    final = np.where(best>tau2)[0]
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    # phase 3
    message3=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
        start=time.time()
        ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final))[0]            
        message3.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].val_bw += len(ind)*8 + len(ind)*8
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
    temp = np.argsort(temp_score)[::-1][:k]
    collected = pd.Series(temp_score[temp], temp)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    return collected, cct   

def BSA(nodes, n, k, strategy = 'exponential'):
    temp_score = np.zeros(n)
    cct = {}
    start_curs = 0
    end_curs = k
    loopcont = True
    while loopcont:
        message = []
        threshold = 0
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = 1
            start = time.time()
            ind = nodes[i].dindex[start_curs:end_curs]
            dat = nodes[i].darray[start_curs:end_curs]
            try:
                threshold+=nodes[i].darray[end_curs-1]
            except IndexError:
                threshold+=0
            end=time.time()
            message.append([ind, dat])
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].res_bw += len(ind)*8 + len(ind)*8
        start = time.time()
        for i in range(len(nodes)):
            temp_score[message[i][0]] += message[i][1]
        current_topk = np.partition(temp_score, -k)[-k]
        loopcont = (current_topk < threshold) and (end_curs < n)
        start_curs = end_curs
        if strategy == 'linear':
            end_curs += k
        if strategy == 'exponential':
            end_curs = min(n, end_curs*2)
        end = time.time()
        cct[nodes[0].nbrounds]=end-start
    start = time.time()
    final = np.where(temp_score >= threshold)[0]
    end = time.time()
    cct[nodes[0].nbrounds]+=end-start
    
    # phase 3
    message3=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
        start=time.time()
        ind = np.where(np.isin(nodes[i].dindex[start_curs:], final))[0]            
        message3.append([nodes[i].dindex[start_curs:][ind], nodes[i].darray[start_curs:][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].val_bw += len(ind)*8 + len(ind)*8
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
    temp = np.argsort(temp_score)[::-1][:k]
    collected = pd.Series(temp_score[temp], temp)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    return collected, cct
       
def BF(nodes, n, k, delta = 0.1):
    # phase 1
    temp_score = np.zeros(n)
    estimate_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].res_bw += len(considered[0])*8 + len(considered[0])*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    tau=np.partition(temp_score, -k)[-k]
    # lbda (rho) = 4/x_k*ln(k/delta) as described in Hubschle-Schneider et. al. 2016
    lbda = 4/tau*np.log(k/delta) 
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 2
    message2 = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 8
        start=time.time()
        ind, samp = Bernoulli_filter(nodes[i].darray*50, nodes[i].dindex, lbda/50)
        end=time.time()
        message2.append([ind, samp])
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].extended[nodes[i].nbrounds] = len(ind)*4
        nodes[i].res_bw += len(ind)*8 + len(ind)*4
    start = time.time()
    for i in range(len(nodes)):
        estimate_score[message2[i][0]] += message2[i][1]
    rk = np.partition(estimate_score, -k)[-k]
    # alpha (k*) = r_k - np.sqrt(2r_kln(k/delta))
    alpha = int(rk - np.sqrt(2*rk*np.log(k/delta)))
    final = np.where(estimate_score > alpha)[0]
    end = time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 3
    message3=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
        start=time.time()
        ind = np.where(np.isin(nodes[i].dindex[k:], final))[0]            
        message3.append([nodes[i].dindex[k:][ind], nodes[i].darray[k:][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].val_bw += len(ind)*8 + len(ind)*8
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
    temp = np.argsort(temp_score)[::-1][:k]
    collected = pd.Series(temp_score[temp], temp)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    return collected, cct  
    
def EF(nodes, n, k):
    minsample = {}
    cct ={}
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 8
        start=time.time()
        ind, samp = Exp_filter(nodes[i].darray, nodes[i].dindex, k)
        end=time.time()
        message.append([ind, samp])
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].extended[nodes[i].nbrounds] = len(ind)*4
        nodes[i].res_bw += len(ind)*8 + len(ind)*4
    start = time.time()
    for i in range(len(nodes)):
        for j in range(len(message[i][0])):
            if minsample.get(message[i][0][j], 0) == 0:
                minsample[message[i][0][j]] = message[i][1][j]
            else:
                minsample[message[i][0][j]] = min(minsample[message[i][0][j]], message[i][1][j])
    
    # k' = 10k
    sorted_items = sorted(minsample.items(), key=lambda x: x[1])[:10*k]
    final = [x[0] for x in sorted_items]
    end = time.time()
    cct[nodes[0].nbrounds]=end-start
    
    temp_score = np.zeros(n)
    message3=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
        start=time.time()
        ind = np.where(np.isin(nodes[i].dindex, final))[0]            
        message3.append([nodes[i].dindex[ind], nodes[i].darray[ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].val_bw += len(ind)*8 + len(ind)*8
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
    temp = np.argsort(temp_score)[::-1][:k]
    collected = pd.Series(temp_score[temp], temp)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    return collected, cct 

def PF(nodes, n, k, a=1, delta = 0.1):
    
    # phase 1
    temp_score = np.zeros(n)
    estimate_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].res_bw += len(considered[0])*8 + len(considered[0])*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    tau1=np.partition(temp_score, -k)[-k]
    lbda = (solve_lambda(delta, 1)+1)/tau1
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 2
    message2 = []
    for i in range(len(nodes)):
        nodes[i].threshold = tau1*a/len(nodes)
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 8
        start=time.time()
        ind, samp = Pois_filter(nodes[i].darray, nodes[i].dindex, lbda)
        end=time.time()
        message2.append([ind, samp])
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].extended[nodes[i].nbrounds] = len(ind)*4
        nodes[i].res_bw += len(ind)*8 + len(ind)*4
    start = time.time()
    for i in range(len(nodes)):
        estimate_score[message2[i][0]] += message2[i][1]
    rk = np.partition(estimate_score, -k)[-k]
    alpha1 = scipy.special.gammaincinv(rk, delta/2)
    alpha2 = scipy.stats.poisson.ppf(delta/2, alpha1)
    alpha = max(alpha2, scipy.stats.poisson.ppf(delta, lbda*tau1))
    final = np.where(estimate_score > alpha)[0]
    end = time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 3
    message3=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
        start=time.time()
        ind = np.where(np.isin(nodes[i].dindex[k:], final))[0]            
        message3.append([nodes[i].dindex[k:][ind], nodes[i].darray[k:][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].val_bw += len(ind)*8 + len(ind)*8
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
    temp = np.argsort(temp_score)[::-1][:k]
    collected = pd.Series(temp_score[temp], temp)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    return collected, cct 

def VEF(nodes, n, k, a=1, delta = 0.1):
    # phase 1 (initial message + first response)
    temp_score = np.zeros(n)
    cct = {} # center computing time
    C = set()
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].res_bw += len(considered[0])*8 + len(considered[0])*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    tau1=np.partition(temp_score, -k)[-k]
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 2
    message2 = []
    for i in range(len(nodes)):
        nodes[i].threshold = tau1*a/len(nodes)
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 8
        start=time.time()
        temp1, temp2 = value_filter(nodes[i].darray, nodes[i].dindex, nodes[i].threshold)
        nodes[i].cursor = max(k, len(temp1))
        ind, _ = Exp_filter(temp2, temp1, k, delta, tau1)
        end=time.time()
        message2.append(ind)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].res_bw += len(ind)*8
    start=time.time()
    for i in range(len(nodes)):
        for j in message2[i]:
            C.add(j)
    B = Bloom(len(C)+10, 0.01)
    B.update(C)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 3
    message3 = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = B.size_in_bits/8
        start=time.time()
        ind = np.array([x in B for x in nodes[i].dindex[k:nodes[i].cursor]])
        message3.append([nodes[i].dindex[k:nodes[i].cursor][ind], nodes[i].darray[k:nodes[i].cursor][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(nodes[i].dindex[k:nodes[i].cursor][ind])
        nodes[i].score_sent[nodes[i].nbrounds] = len(nodes[i].dindex[k:nodes[i].cursor][ind])
        nodes[i].val_bw += len(nodes[i].dindex[k:nodes[i].cursor][ind])*8 + len(nodes[i].dindex[k:nodes[i].cursor][ind])*8
    start = time.time()
    miss = np.ones(n)*tau1
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
        miss[message[i][0]]-=nodes[i].threshold
        miss[message3[i][0]]-=nodes[i].threshold
    tau2 = np.partition(temp_score, -k)[-k]
    best = temp_score + miss +1e-5
    final = np.where(best>tau2)[0]
    end = time.time()
    cct[nodes[0].nbrounds]=end-start
    
    message3=[]
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
        start=time.time()
        ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final))[0]            
        message3.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
        end=time.time()
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].val_bw += len(ind)*8 + len(ind)*8
    start=time.time()
    for i in range(len(nodes)):
        temp_score[message3[i][0]]+=message3[i][1]
    temp = np.argsort(temp_score)[::-1][:k]
    collected = pd.Series(temp_score[temp], temp)
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    return collected, cct 

def PPF(nodes, n, k, delta = 0.1, strategy = 'exponential'):
    # phase 1
    temp_score = np.zeros(n)
    estimate_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].res_bw += len(considered[0])*8 + len(considered[0])*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    tau1=np.partition(temp_score, -k)[-k]
    lbda = (solve_lambda(delta, 1)+1)/tau1
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 2
    start_curs = 0
    end_curs = 2*k
    loopcont = True
    while loopcont:
        message2 = []
        threshold = 0
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = 1
            start = time.time()
            ind, samp = Pois_filter(nodes[i].darray[start_curs:end_curs], nodes[i].dindex[start_curs:end_curs], lbda)
            try:
                threshold+=nodes[i].darray[end_curs-1]
            except IndexError:
                threshold+=0
            end=time.time()
            message2.append([ind, samp])
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].extended[nodes[i].nbrounds] = len(ind)*4
            nodes[i].res_bw += len(ind)*8 + len(ind)*4
        start = time.time()
        for i in range(len(nodes)):
            estimate_score[message2[i][0]] += message2[i][1]
        rk = np.partition(estimate_score, -k)[-k]
        alpha1 = scipy.special.gammaincinv(rk, delta/2)
        loopcont = (alpha1/lbda < threshold) and (end_curs < n)
        start_curs = end_curs
        if strategy == 'linear':
            end_curs += k
        if strategy == 'exponential':
            end_curs = min(n, end_curs*2)
        end = time.time()
        cct[nodes[0].nbrounds]=end-start
    start = time.time()
    alpha2 = scipy.stats.poisson.ppf(delta/2, alpha1)
    alpha = max(alpha2, scipy.stats.poisson.ppf(delta, lbda*tau1))
    final = np.where(estimate_score > alpha)[0]
    end = time.time()
    cct[nodes[0].nbrounds]+=end-start
    
    # phase 3
    if(len(final) < n/100):
        message3=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[k:], final))[0]            
            message3.append([nodes[i].dindex[k:][ind], nodes[i].darray[k:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].val_bw += len(ind)*8 + len(ind)*8
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message3[i][0]]+=message3[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
    else:
        message3=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[k:start_curs], final))[0]            
            message3.append([nodes[i].dindex[k:start_curs][ind], nodes[i].darray[k:start_curs][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].val_bw += len(ind)*8 + len(ind)*8
        miss = np.ones(n)*tau1
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message3[i][0]]+=message3[i][1]
            miss[message[i][0]]-=nodes[i].darray[start_curs]
            miss[message3[i][0]]-=nodes[i].darray[start_curs]
        tau2 = np.partition(temp_score, -k)[-k]
        best = temp_score + miss +1e-5
        final2 = np.where(best>tau2)[0]
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
        
        message4=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final2)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[start_curs:], final2))[0]            
            message4.append([nodes[i].dindex[start_curs:][ind], nodes[i].darray[start_curs:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].val_bw += len(ind)*8 + len(ind)*8
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message4[i][0]]+=message4[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
    
    return collected, cct

def VPF(nodes, n, k, a=1, delta = 0.1):
    # phase 1
    temp_score = np.zeros(n)
    estimate_score = np.zeros(n)
    cct = {} # center computing time
    message = []
    for i in range(len(nodes)):
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 2 # broadcast k and a flag indicate the query started
        start = time.time()
        considered = [nodes[i].dindex[:k], nodes[i].darray[:k]]
        end=time.time()
        message.append(considered)
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].score_sent[nodes[i].nbrounds] = len(considered[0])
        nodes[i].res_bw += len(considered[0])*8 + len(considered[0])*8
    start = time.time()
    for i in range(len(nodes)):
        temp_score[message[i][0]]+=message[i][1]
    tau1=np.partition(temp_score, -k)[-k]
    lbda = (solve_lambda(delta, 1)+1)/tau1
    end=time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 2
    message2 = []
    for i in range(len(nodes)):
        nodes[i].threshold = tau1*a/len(nodes)
        nodes[i].nbrounds += 1
        nodes[i].received_message[nodes[i].nbrounds] = 8
        start=time.time()
        temp1, temp2 = value_filter(nodes[i].darray, nodes[i].dindex, nodes[i].threshold)
        nodes[i].cursor = max(k, len(temp1))
        ind, samp = Pois_filter(temp2, temp1, lbda)
        end=time.time()
        message2.append([ind, samp])
        nodes[i].compute_time[nodes[i].nbrounds] = end-start
        nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
        nodes[i].extended[nodes[i].nbrounds] = len(ind)*4
        nodes[i].res_bw += len(ind)*8 + len(ind)*4
    start = time.time()
    for i in range(len(nodes)):
        estimate_score[message2[i][0]] += message2[i][1]
    rk = np.partition(estimate_score, -k)[-k]
    alpha1 = scipy.special.gammaincinv(rk, delta/2)
    alpha2 = scipy.stats.poisson.ppf(delta/2, alpha1)
    alpha = max(alpha2, scipy.stats.poisson.ppf(delta, lbda*tau1))
    final = np.where(estimate_score > alpha)[0]
    end = time.time()
    cct[nodes[0].nbrounds]=end-start
    
    # phase 3
    if(len(final) < n/100):
        message3=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[k:], final))[0]            
            message3.append([nodes[i].dindex[k:][ind], nodes[i].darray[k:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].val_bw += len(ind)*8 + len(ind)*8
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message3[i][0]]+=message3[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
    else:
        message3=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[k:nodes[i].cursor], final))[0]            
            message3.append([nodes[i].dindex[k:nodes[i].cursor][ind], nodes[i].darray[k:nodes[i].cursor][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].val_bw += len(ind)*8 + len(ind)*8
        miss = np.ones(n)*tau1
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message3[i][0]]+=message3[i][1]
            miss[message[i][0]]-=nodes[i].threshold
            miss[message3[i][0]]-=nodes[i].threshold
        tau2 = np.partition(temp_score, -k)[-k]
        best = temp_score + miss +1e-5
        final2 = np.where(best>tau2)[0]
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
        
        message4=[]
        for i in range(len(nodes)):
            nodes[i].nbrounds += 1
            nodes[i].received_message[nodes[i].nbrounds] = len(final2)*8
            start=time.time()
            ind = np.where(np.isin(nodes[i].dindex[nodes[i].cursor:], final2))[0]            
            message4.append([nodes[i].dindex[nodes[i].cursor:][ind], nodes[i].darray[nodes[i].cursor:][ind]])
            end=time.time()
            nodes[i].compute_time[nodes[i].nbrounds] = end-start
            nodes[i].ID_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].score_sent[nodes[i].nbrounds] = len(ind)
            nodes[i].val_bw += len(ind)*8 + len(ind)*8
        start=time.time()
        for i in range(len(nodes)):
            temp_score[message4[i][0]]+=message4[i][1]
        temp = np.argsort(temp_score)[::-1][:k]
        collected = pd.Series(temp_score[temp], temp)
        end=time.time()
        cct[nodes[0].nbrounds]=end-start
        
    return collected, cct 
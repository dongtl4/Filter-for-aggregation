import multiprocessing
from datetime import datetime
import time
from machine import local_nodes as node
from utils import common as cm
from utils import method as mt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import gc

def preprocess(i):
    i.data = i.data.sort_values(ascending=False)
    i.data = i.data[i.data>0]
    i.dup_data()
    return i
        
def run_test(func):
    # Helper function to run a test function
    return func()
        
if __name__ == "__main__":
    result = []
    # data = pd.read_csv('data/HIGGS.csv', header=None, dtype='float')
    data = pd.read_csv('data/HEPMASS.csv', header=0, dtype='float')
    for amp in [10]:
        # print('loop' + str(count))
        rdata = data.drop(["# label", "mass"], axis=1) # for hepmass
        # rdata = data.drop([0], axis=1) # for higgs
        rdata.columns = np.arange(0, rdata.shape[1])
        n=rdata.shape[0]
        m=rdata.shape[1]*10
        nodes=[]
        num_workers = 40
        chunk = int(n/num_workers)
        
        # partitioning scores to nodes (for varying m, amp is amplification ratio)
        print('gen')
        for k in range(rdata.shape[1]):
            source = rdata[k].values
            with multiprocessing.Pool(processes=num_workers) as pool:
                datai = pool.starmap(
                    cm.create_data, 
                    [(source[i*chunk:(i+1)*chunk], amp, 'zipf', np.random.randint(7,17)/10,) 
                     for i in range(num_workers)])
            temp = np.concatenate(datai)
            for j in range(10):
                nodes.append(node.node(temp[:,j], rdata.index))
        del rdata
        gc.collect()
        
        print('cal')
        gs = pd.Series([0]*n)
        nodes[0].fullindex = gs.index      
        for i in nodes:
            i.data = pd.Series(data=np.power(i.data, 2), index = i.data.index)
            gs+=i.data
        gs.sort_values(ascending=False, inplace=True)
          
        print('sort')
        with multiprocessing.Pool(processes=num_workers) as pool:
            nodes = pool.map(preprocess, nodes)
            
        for i in nodes:
            i.refresh()
        
        gc.collect()
        
        print('main')
        def ktest1(k):
            output, cct = mt.TPUT(nodes, n, k)
            gc.collect()
            return mt.evaluate('TPUT '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest2(k):
            output, cct = mt.BSA(nodes, n, k)
            gc.collect()
            return mt.evaluate('BDBPA '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest3(k):
            output, cct = mt.KLEE(nodes, n, k)
            gc.collect()
            return mt.evaluate('KLEE '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest4(k):
            output, cct = mt.BF(nodes, n, k)
            gc.collect()
            return mt.evaluate('BF '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest5(k):
            output, cct = mt.VPF(nodes, n, k)
            gc.collect()
            return mt.evaluate('VPF '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest6(k):
            output, cct = mt.PF(nodes, n, k)
            gc.collect()
            return mt.evaluate('PFT '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest7(k):
            output, cct = mt.PPF(nodes, n, k)
            gc.collect()
            return mt.evaluate('PPF ' +str(k), nodes, k, output, cct, gs, epsilon=0)
        
        K = [20, 40, 60, 80, 100]
        
        for k in K:
            result.append(ktest1(k))
            result.append(ktest2(k))
            for i in range(10):
                result.append(ktest3(k))
                result.append(ktest4(k))
                result.append(ktest5(k))
                result.append(ktest6(k))
                result.append(ktest7(k))
        
        k=100
        def dtest5(d):
            output, cct = mt.VPF(nodes, n, k, a=1, delta=d)
            gc.collect()
            return mt.evaluate('VPF '+str(d), nodes, k, output, cct, gs, epsilon=0)
        
        def dtest6(d):
            output, cct = mt.PF(nodes, n, k, a=1, delta=d)
            gc.collect()
            return mt.evaluate('PFT '+str(d), nodes, k, output, cct, gs, epsilon=0)
        
        def dtest7(d):
            output, cct = mt.PPF(nodes, n, k, delta=d)
            gc.collect()
            return mt.evaluate('PPF ' +str(d), nodes, k, output, cct, gs, epsilon=0)
        
        D = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        for delta in D:
            for i in range(10):
                result.append(dtest5(delta))
                result.append(dtest6(delta))
                result.append(dtest7(delta))

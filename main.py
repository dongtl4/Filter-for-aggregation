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
    data = pd.read_csv('data/HIGGS.csv', header=None, dtype='float')
    for count in range(1):
        print('loop' + str(count))
        # rdata = data.drop(["# label", "mass"], axis=1)
        rdata = data.drop([0], axis=1)
        rdata.columns = np.arange(0, rdata.shape[1])
        n=rdata.shape[0]
        m=rdata.shape[1]*10
        nodes=[]
        num_workers = 40
        chunk = int(n/num_workers)
        # with multiprocessing.Pool(processes=num_workers) as pool:
        #     temp = pool.starmap(to_positive, [(rdata[i],) for i in range(rdata.shape[1])])
        # rdata = pd.concat(temp, axis=1)
        # del temp
        # gc.collect()
        print('gen')
        for k in range(rdata.shape[1]):
            source = rdata[k].values
            with multiprocessing.Pool(processes=num_workers) as pool:
                datai = pool.starmap(
                    cm.create_data, 
                    [(source[i*chunk:(i+1)*chunk], 10, 'zipf', np.random.randint(7, 17)/10,) 
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
            output, cct = mt.TPOR(nodes, n, k)
            gc.collect()
            return mt.evaluate('TPOR '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest3(k):
            output, cct = mt.KLEE(nodes, n, k)
            gc.collect()
            return mt.evaluate('KLEE '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest4(k):
            output, cct = mt.BF(nodes, n, k)
            gc.collect()
            return mt.evaluate('BF '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest5(k):
            output, cct = mt.EF(nodes, n, k)
            gc.collect()
            return mt.evaluate('EF '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest6(k):
            output, cct = mt.VEF(nodes, n, k)
            gc.collect()
            return mt.evaluate('VEF '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest7(k):
            output, cct = mt.VPF(nodes, n, k)
            gc.collect()
            return mt.evaluate('VPF '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        def ktest8(k):
            output, cct = mt.PF(nodes, n, k)
            gc.collect()
            return mt.evaluate('PF '+str(k), nodes, k, output, cct, gs, epsilon=0)
        
        
        K = [20, 40, 60, 80, 100]
        for count in range(30):
            with multiprocessing.Pool(processes=num_workers) as pool:
                result1 = pool.starmap(ktest1, [(k,) for k in K])
                result2 = pool.starmap(ktest2, [(k,) for k in K])
                result3 = pool.starmap(ktest3, [(k,) for k in K])
                result4 = pool.starmap(ktest4, [(k,) for k in K])
                result5 = pool.starmap(ktest5, [(k,) for k in K])
                result6 = pool.starmap(ktest6, [(k,) for k in K])
                result7 = pool.starmap(ktest7, [(k,) for k in K])
        
            result=result+result1+result2+result3+result4+result5+result6+result7
            gc.collect()

            
       


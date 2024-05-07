# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:43:01 2022

local databases

@author: dongh
"""

import numpy as np
from rbloom import Bloom
# import csv
import pandas as pd
# import time
# from machine import analysis_center

class node:    
    def __init__(self, scores, indexes, mean_ping=20, upload=45, download=45, stored=False, node_name=''):
        """
        create database with new data and generate random variables
        """
        self.data = pd.Series(data=scores, index=indexes)
        self.darray = []
        self.dindex = []
        self.sent = np.zeros(len(self.data), dtype='bool') # only used when needed to save the sent position
        self.random = pd.Series([], name='random sample', dtype = 'float64')
        self.threshold = 0
        self.prevthres = np.inf
        self.fullindex = 0
        # performance tracking
        self.ID_sent = {} # number of item's ID sent
        self.score_sent = {} # number of local score sent
        self.rd_value_sent = {} # number of random value sent
        self.compute_time = {}
        self.extended = {}
        self.nbrounds = 0 # number of round-trip communications
        self.received_message = {}
        self.ping = mean_ping #ms
        self.upspeed = upload #Mbps
        self.downspeed = download #Mbps
        self.cursor = 0 #for sorted access
        self.res_bw = 0 #communication cost of response step
        self.val_bw = 0 #communication cost of validation step
        
        
    def sort(self, t=1):
        if t==1:
            self.data = self.data.sort_values(ascending=False)
        elif t==2:
            self.random = self.random.sort_values(ascending=False)
        
    def dup_data(self):
        self.darray = np.array(self.data.values)
        self.dindex = np.array(self.data.index)
        self.data = pd.Series(dtype='float64')
        
    def sub_data(self):
        self.data = pd.Series(data=self.darray, index=self.dindex)
        self.darray = []
        self.dindex = []
        
    # clear all except data
    def refresh(self):
        self.threshold = 0
        self.prevthres = np.inf
        self.random = pd.Series([], name='random sample', dtype = 'float64')
        self.sent = np.zeros(max(len(self.data), len(self.darray)), dtype='bool')
        # performance tracking
        self.compute_time = {}
        self.ID_sent = {} # number of item's ID sent
        self.score_sent = {} # number of local score sent
        self.rd_value_sent = {} # number of random value sent
        self.extended = {}
        self.nbrounds = 0 # number of round-trip communications
        self.received_message = {}
        self.res_bw = 0 #communication cost of response step
        self.val_bw = 0
    
        
        
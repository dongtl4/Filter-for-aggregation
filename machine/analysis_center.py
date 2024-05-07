# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:40:04 2022

Analysis center

@author: dongh
"""
# from machine import local_databases
import numpy as np
import pandas as pd

class query_initiator:
    
    def __init__(self, S, p):
        self.rm = []
    
    # boardcast message to local databases
    def boardcast(self, message, nbrs, local_list):
        for node in local_list:
            node.received_message[nbrs] = message
    
    # drop all message received
    def empty_message(self):
        self.rm = []
        
    
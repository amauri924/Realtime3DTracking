
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:26:48 2020

@author: antoine
"""

import torch


with open('test.txt','w') as f:
    f.write(str(torch.cuda.is_available()))
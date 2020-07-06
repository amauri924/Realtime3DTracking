
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:20:48 2020

@author: antoine
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--result_file', type=str, default='result_1.1_nul.txt', help='cfg file path')
opt = parser.parse_args()



with open(opt.result_file,'r') as f:
    lines=f.readlines()


epoch=[]
loss_xy=[]
lobj=[]
lcls=[]
lcent=[]
loss=[]


for line in lines:
    splitted_elements=line.split(' ')
    line=[elmt.split('\n')[0] for elmt in splitted_elements if elmt!='']


    epoch.append(int(line[0].split('/')[0])+1)
    loss_xy.append(float(line[2]))
    lobj.append(float(line[4]))
    lcls.append(float(line[5]))
    lcent.append(float(line[6]))
    loss.append(float(line[7]))
    
    
    
            

plt.figure(1)
plt.ylim(0, 1)
plt.plot(epoch,loss,label='total')
plt.plot(epoch,lcent,label='lcent')
plt.plot(epoch,lcls,label='lcls')
plt.plot(epoch,loss_xy,label='loss_xy')
plt.legend()
plt.show()



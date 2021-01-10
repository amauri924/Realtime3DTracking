
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 10:04:33 2020

@author: antoine
"""

import matplotlib.pyplot as plt
import os
import numpy as np

result_file="result_15.16.txt"

with open(result_file,'r') as f:
    result_lines=[[value for value in line.split('\n')[0].split('|||')[1].split(' ') if value != ''] for line in f.readlines()]
    

with open("log_15.16.txt",'r') as f:
    log_lines=[[value for value in line.split('\n')[0].split(' ') if value != ''] for line in f.readlines() if len([value for value in line.split('\n')[0].split(' ') if value != '']) == 15]
    

it_per_epoch=int(log_lines[0][2].split('/')[1])

loss=[float(item[11]) for item in log_lines]
lr=[float(item[14]) for item in log_lines]

OS=[float(item[7]) for item in result_lines]
mAP=[float(item[3]) for item in result_lines]
depth_rel_err=[float(1-float(item[5])) for item in result_lines]
dim_rel_err=[float(1-float(item[6])) for item in result_lines]


fitness=[(float(1-float(result_lines[i][7]))+float(1-float(result_lines[i][3]))+float(result_lines[i][5]) + float(result_lines[i][6]))/4.  for i in range(len(result_lines))]

num_it=np.linspace(0,len(loss)-1,len(loss),dtype=np.uint32)

plt.figure()
plt.plot(num_it,loss)
# plt.xscale('log')
plt.show()

plt.figure()
plt.plot(mAP,color="b")
plt.plot(depth_rel_err,color="r")
plt.plot(dim_rel_err,color="g")
plt.plot(OS,color="y")
plt.show()


plt.figure()
plt.plot(fitness)
plt.show()
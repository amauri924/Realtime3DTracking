
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 07:49:52 2020

@author: antoine
"""

import matplotlib.pyplot as plt
import numpy as np
import os


training_root_folder="/home/antoine/remote_criann/barebone_depth/CosineAnnealing/"

training_folders=[os.path.join(training_root_folder,folder) for folder in os.listdir(training_root_folder) if os.path.isdir(os.path.join(training_root_folder,folder))]

all_val=[]
all_train_err=[]
all_lr=[]
for folder in training_folders:
    val=np.zeros(500)
    with open(os.path.join(folder,"val_results.txt"),"r") as f:
        val=np.array([float(num.split('\n')[0].split('(')[-1].split(')')[0]) for num in f.readlines()])
#        val[:len(values)]=values
    all_val.append(val)
    training_rel_err=np.zeros(500)
    with open(os.path.join(folder,"log_rel_err.txt"),"r") as f:
        training_rel_err=np.array([float(num.split('\n')[0].split('(')[-1].split(')')[0]) for num in f.readlines()])
   
#    with open(os.path.join(folder,"lr.txt"),"r") as f:
#        lr=np.array([float(num.split('\n')[0].split('[')[-1].split(']')[0]) for num in f.readlines()])
#    all_lr.append(lr)
    all_train_err.append(training_rel_err)
    
    
    

for i,folder in enumerate(training_folders):
    
    method_name=folder.split('/')[-1]
    
    
    plt.plot(all_val[i],'+',label=method_name)

plt.ylim(ymax = 0.2, ymin = 0.1)
plt.legend()
plt.title('Validation with Cosine Annealing')
plt.savefig("validation.png",dpi=200)
plt.show()



for i,folder in enumerate(training_folders):
    
    method_name=folder.split('/')[-1]
    
    
    plt.plot(all_train_err[i],'+',label=method_name)

plt.ylim(ymax = 0.2, ymin = 0.1)
plt.legend()
plt.title('Training Relative error with Cosine Annealing')
plt.savefig("training err.png",dpi=200)
plt.show()


#for i,folder in enumerate(training_folders):
#    
#    method_name=folder.split('/')[-1]
#    
#    
#    plt.plot(all_lr[i],'+',label=method_name)
#
#plt.legend()
#plt.title('LR')
#plt.savefig("lr.png",dpi=200)
#plt.show()
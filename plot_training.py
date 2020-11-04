
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 07:49:52 2020

@author: antoine
"""

import matplotlib.pyplot as plt
import numpy as np
import os


training_root_folder="/home/antoine/remote_criann/barebone_depth/AdaBound/Single_GPU_train/with_decay/"

training_folders=[os.path.join(training_root_folder,folder) for folder in os.listdir(training_root_folder) if os.path.isdir(os.path.join(training_root_folder,folder))]

all_val=[]
all_train_err=[]
all_lr=[]
for folder in training_folders:
    val=np.zeros(500)
    with open(os.path.join(folder,"val_results.txt"),"r") as f:
#        val=np.array([float(num.split('(')[-1].split(')')[0]) for num in f.readlines()[0].split('t')[1:]])
        val=np.array([float(num.split('\n')[0].split('(')[-1].split(')')[0]) for num in f.readlines()])
#        val[:len(values)]=values
    all_val.append(val)
    training_rel_err=np.zeros(500)
    with open(os.path.join(folder,"log_rel_err.txt"),"r") as f:
        training_rel_err=np.array([float(num.split('\n')[0].split('(')[-1].split(')')[0]) for num in f.readlines()])
   
    with open(os.path.join(folder,"lr.txt"),"r") as f:
        lr=np.array([float(num.split('\n')[0].split('[')[-1].split(']')[0]) for num in f.readlines()])
    all_lr.append(lr)
    all_train_err.append(training_rel_err)
    
    
    

for i,folder in enumerate(training_folders):
    
    method_name=folder.split('/')[-1]
    
    
    plt.plot(all_val[i],'+',label=method_name)

plt.ylim(ymax = 0.3, ymin = 0)
plt.legend()
plt.title('Validation')
plt.savefig("validation.png",dpi=200)
plt.show()



for i,folder in enumerate(training_folders):
    
    method_name=folder.split('/')[-1]
    
    
    plt.plot(all_train_err[i],'+',label=method_name)

plt.ylim(ymax = 0.3, ymin = 0)
plt.legend()
plt.title('Training Relative error')
plt.savefig("training err.png",dpi=200)
plt.show()

with open("best_errors.txt",'w') as f:
    f.write("Backbone"+"     "+"Validation"+"     "+"Training"+'\n')
    for i,folder in enumerate(training_folders):
        method_name=folder.split('/')[-1]
        min_val=str(np.nanmin(all_val[i]))
        min_train=str(np.nanmin(all_train_err[i]))
        f.write(method_name+':      '+min_val+'       '+min_train+'\n')

for i,folder in enumerate(training_folders):
    
    method_name=folder.split('/')[-1]
    
    
    plt.plot(all_lr[i],'+',label=method_name)

plt.legend()
plt.title('LR')
plt.savefig("lr.png",dpi=200)
#plt.show()
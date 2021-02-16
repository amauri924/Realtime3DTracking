
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:20:48 2020

@author: antoine
"""

import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--result_file', type=str, default='5bin_2fc_results.txt', help='cfg file path')
opt = parser.parse_args()

result_file_list=["5bin_2fc_results.txt","5bin_results.txt","10bin_results.txt","20bin_results.txt","result_ADAM_lre-3.txt"]

for result_file in result_file_list:

    with open(result_file,'r') as f:
        lines=f.readlines()
    
    
    
    loss=[]
    abs_err=[]
    epoch=[]
    
    for line in lines:
        splitted_elements=line.split(' ')
        line=[elmt.split('\n')[0] for elmt in splitted_elements if elmt!='']
    
        if result_file=="result_ADAM_lre-3.txt":
            epoch.append(int(line[0].split('/')[0])+1)
            loss.append(float(line[8]))
            abs_err.append(float(line[18]))
        else:
            epoch.append(int(line[0].split('/')[0])+1)
            loss.append(float(line[9]))
            abs_err.append(float(line[21]))
        
        
        
                
    
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    offset = 60
    par1 = host.twinx()
#    par2 = host.twinx()
#    
#    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
#    par2.axis["right"] = new_fixed_axis(loc="right",
#                                        axes=par2,
#                                        offset=(offset, 0))
    
    par1.axis["right"].toggle(all=True)
    
    
    
    
    host.set_xlim(200, 600)
    host.set_ylim(0, 20)
    
    host.set_xlabel("Epoch")
    host.set_ylabel("Training loss")
    par1.set_ylabel("Test abs error")
    par1.set_ylim(0, 1)
    p1, = host.plot(epoch, loss, label="Training loss")
    p2, = par1.plot(epoch, abs_err, label="Test abs error")
    
    

    
    host.legend()
    
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    
    plt.draw()
    plt.savefig(result_file.split('.')[0]+".png",dpi=300)
    plt.show()
    



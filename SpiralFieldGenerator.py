# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

from HeliopodTools import positions
from HeliopodTools import get_x_y_co
from HeliopodTools import heliopod
from HeliopodTools import line_between_2
"""
Author: T McKecnie
Stellenbosch University
8 September 2019

Spiral Heliood field layout generator
"""
''' note this tool is currenlty only applicable for positive y values in field dimensions and a tower at (0,0)'''
#%% Polar to cart function
def polar_to_cart(angle,radius):
    quad = (math.degrees(angle) % 360) // 90
    if quad == 0: # first quadrant
#        print('First quad')
        x = radius*math.cos(angle)
        y = radius*math.sin(angle)
    elif quad == 1: # second quadrant
#        print('Second quad')
        x = -radius*math.cos(np.pi-angle)
        y = radius*math.sin(np.pi-angle)
    elif quad == 2: # third quadrant
#        print('Third quad')
        x = -radius*math.cos(angle-np.pi)
        y = -radius*math.sin(angle-np.pi)
    else: # fourth quadrant
#        print('Fourth quad')
        x = radius*math.cos(2*np.pi-angle)
        y = -radius*math.sin(2*np.pi-angle)
    return x,y
#%% field size and heliopod parameters
l = 6
num_pods = 600;
x_t = 0
y_t = 0
r_t = 15


#%% Spiral pattern equations

alpha = np.empty(num_pods,dtype=float)
r = np.empty(num_pods,dtype=float)

psi = (1+np.sqrt(5))/2
a = 0.75
b = 0.85

for k in range(num_pods): # this loop calcs polar co-ords of heliopods centres
    alpha[k] = 2*np.pi*(psi**(-2))*(k+1)
    r[k] = a*(k+1)**b

#%% Pod centres from polar to cart
centres = np.zeros((num_pods,2), dtype=float)

for k in range(num_pods): # loop to get pod centres in cartesian
    centres[k] = polar_to_cart(alpha[k],r[k])
     
#%%  Remove pods near tower
new = np.full(num_pods,0)    
centres = np.column_stack((centres,new))
   
for k in range(num_pods): 
    if ((centres[k,0]-x_t)**2 + (centres[k,1]-y_t)**2)**0.5 < r_t*2:
        centres[k,2] = 1   
      
while  centres[:,2].sum() > 0:
    print(centres[:,2].sum())
    if centres[0,2] == 1:
        centres = np.delete(centres,0,axis = 0)
#%% Remove pods south of the tower
for k in range(len(centres)):
    if centres[k,1] < 0:
        centres[k,2] = 1
centres = centres[centres[:,2].argsort()[::-1]]

while  centres[:,2].sum() > 0:
    print(centres[:,2].sum())
    if centres[0,2] == 1:
        centres = np.delete(centres,0,axis = 0)                
#%% Pods to individual heliostats        
for k in range(len(centres)):
    if k == 0:
        field = heliopod(centres[k,0],centres[k,1],l, x_t, y_t,0) # field contains the locations of each heliostat
    else:
        field = np.append(field, heliopod(centres[k,0],centres[k,1],l, x_t, y_t,0), axis = 0) 
#$$ Plot heliostat bounding circles            
for k in range(len(field)):
        temp = get_x_y_co([field[k,0],field[k,1],math.sqrt((1.83)**2+(1.22)**2)/2])
        plt.plot(temp[:,0],temp[:,1],'b.',markersize = 0.1)          
#%% Check for collision
for k in range(len(centres)):
    for i in range(len(centres)):
        if (((centres[k,0] - centres[i,0])**2 + (centres[k,1] - centres[i,1])**2)**0.5 < 2*(l/math.sqrt(3)+0.5*(1.83**2+1.22**2)**0.5)) and i != k:
            print('Collision pods: ', k, 'and', i)

#%% Plot layout    
       
for k in range(len(centres)): # loop to plot pod bounding circles   
    circles = get_x_y_co([centres[k,0],centres[k,1],(l/math.sqrt(3)+0.5*(1.83**2+1.22**2)**0.5)])
    plt.plot(circles[:,0],circles[:,1],'.',linewidth=0.06) 
plt.plot(field[:,0],field[:,1],'y*')   
plt.plot(centres[:,0],centres[:,1],'g*',markersize=1)
plt.grid(True)
plt.axis('equal')
plt.show()

#%% writes text file
new = np.full((len(centres))*6, 1.5)
field = np.column_stack((field,new))
temp = np.copy(field[:,1])
field[:,1] = field[:,2]
field[:,2] = temp
np.savetxt('test.txt',field, fmt = "%.3f",delimiter="\t")
    
    
    
    
    
    
    
    
    
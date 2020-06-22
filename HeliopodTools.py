# -*- coding: utf-8 -*-
"""
Python Module

Author: T McKechnie
Stellenbosch University
10 September 2019

"""

import numpy as np
import matplotlib.pyplot as plt
import math

""" This module constains the class positions which stores heliopod centres, 
get_x_y_co functions which returns the x and y co-ordinates of a circle given the centre and radius,
heliopod function, which returns the positions for individual heliostats given the pod centre and tower location,
here the pod points on of its vertices towards the tower
line_between_2 function which plots a line bewteen the tower and heliopod centre. """

#%% create a class of heliopod positions
class positions: # centres of pods
    # attrinutes
    def __init__(self, x, y):
        self.x = x # x position of pod center
        self.y = y # y position of pod center
    def print(self):
        print('\nx =', self.x,', y =',self.y)
        
#%%        
def get_x_y_co(circle): # return x and y values for a circle defined by its radius and centre
    x_c = circle[0] #x-co of circle (center)
    y_c = circle[1] #y-co of circle (center)
    r = circle[2] #radius of circle
    arr = np.ones(shape=(360,2))
    for i in range(360):
        y = y_c + r*math.sin(i)
        x = x_c + r*math.cos(i)
        #Create array with all the x-co and y-co of the circle
        arr[i][0] = x
        arr[i][1] = y
    return arr         
#%%
def heliopod(x_c,y_c,l,x_t,y_t,orientation): # this class determines the locations of individual heliostats given a pod center point and the tower location
   
    r = l/math.sqrt(3) # pod radius
    e = math.sqrt(3)/(2)*l # triangle height   
     
    centre = np.array([x_t,y_t]) # Tower centre
    
    # Parameters that define first heliostat position on circle, vertex points to tower
    # if x_c- centre[0] < 0.001:
    #     theta = -np.pi/2
    # else:
    theta = math.atan((y_c-centre[1])/(x_c-centre[0]))
    if x_c-centre[0] < 0.1:
        print(x_c, centre[0])
    if orientation == 0: # vertex toward tower
        line_length = ((x_c-centre[0])**2 + (y_c-centre[1])**2)**0.5 - r
    elif orientation > 0:
        line_length = ((x_c-centre[0])**2 + (y_c-centre[1])**2)**0.5 + r
    point = np.array([[line_length*math.cos(theta), line_length*math.sin(theta)]])
    
    # theta_2 = math.atan((point[0,1]-y_t)/(point[0,0]-x_t)) # sanity check, angle for centre and vertex must be equal
    
    
    if x_c < 0:
        point =  point * - 1
    
    if orientation == 0: # vertex toward tower
        gamma = theta - np.pi/2  # pod rotation angle
        print('**')
    elif orientation > 0:
        gamma = theta + np.pi/2  # pod rotation angle
        print('******')
 
    
    if x_c < 0:
        gamma = gamma + np.pi
    
    # other heliostat positions given first vertex
    x_2 = point[0,0] + 0.25*l*math.cos(gamma) - 0.5*e*math.sin(gamma)
    y_2 = point[0,1] + 0.5*e*math.cos(gamma) + 0.25*l*math.sin(gamma)
    
    x_3 = point[0,0] - 0.25*l*math.cos(gamma) - 0.5*e*math.sin(gamma)
    y_3 = point[0,1] + 0.5*e*math.cos(gamma) - 0.25*l*math.sin(gamma)
    
    x_4 = point[0,0] + 0.5*l*math.cos(gamma) - e*math.sin(gamma)
    y_4 = point[0,1] + e*math.cos(gamma) + 0.5*l*math.sin(gamma)
    
    x_5 = point[0,0]  - e*math.sin(gamma)
    y_5 = point[0,1] + e*math.cos(gamma) 
    
    x_6 = point[0,0] - 0.5*l*math.cos(gamma) - e*math.sin(gamma)
    y_6 = point[0,1] + e*math.cos(gamma) - 0.5*l*math.sin(gamma)
    
    # append array to include all heliostat positions of a single heliopod
    point = np.append(point, [[x_2, y_2]],axis=0)
    point = np.append(point, [[x_3, y_3]],axis=0)
    point = np.append(point, [[x_4, y_4]],axis=0)
    point = np.append(point, [[x_5, y_5]],axis=0)
    point = np.append(point, [[x_6, y_6]],axis=0)
    
    return point
#%% cornfield heliopod positions
def heliopod_cornfield(x_c,y_c,l,x_t,y_t,orientation): # this class determines the locations of individual heliostats given a pod center point
   
    r = l/math.sqrt(3) # pod radius
    e = math.sqrt(3)/(2)*l # triangle height   
     
    centre = np.array([x_t,y_t]) # Tower centre
    
    # Parameters that define first heliostat position on circle, vertex points to tower
    if orientation == 0 and x_c >= 0 and y_c >= 0:# vertex upward ie quadrant 1
        point = np.array([[x_c, y_c-r]])
    elif orientation > 0 and x_c >= 0 and y_c >= 0: # vertext down ie quadrant 1
        point = np.array([[x_c, y_c+r]]) # this is what lines up the pod stacked in line
    elif orientation == 0 and x_c < 0 and y_c >= 0: # vertex upward quadtrant 2
        point = np.array([[x_c, y_c-r]])
    elif orientation > 0 and x_c < 0 and y_c >= 0: # vertex down quadrant 2
        point = np.array([[x_c,y_c +r]])
        
    if x_c < 0:
        point =  point * - 1
    
    if orientation == 0: # vertex toward tower
        gamma = 0  # pod rotation angle
        print('**')
    elif orientation > 0:
        gamma = np.pi  # pod rotation angle
        print('******')
 
    
    if x_c < 0:
        gamma = gamma + np.pi
    
    # other heliostat positions given first vertex
    x_2 = point[0,0] + 0.25*l*math.cos(gamma) - 0.5*e*math.sin(gamma)
    y_2 = point[0,1] + 0.5*e*math.cos(gamma) + 0.25*l*math.sin(gamma)
    
    x_3 = point[0,0] - 0.25*l*math.cos(gamma) - 0.5*e*math.sin(gamma)
    y_3 = point[0,1] + 0.5*e*math.cos(gamma) - 0.25*l*math.sin(gamma)
    
    x_4 = point[0,0] + 0.5*l*math.cos(gamma) - e*math.sin(gamma)
    y_4 = point[0,1] + e*math.cos(gamma) + 0.5*l*math.sin(gamma)
    
    x_5 = point[0,0]  - e*math.sin(gamma)
    y_5 = point[0,1] + e*math.cos(gamma) 
    
    x_6 = point[0,0] - 0.5*l*math.cos(gamma) - e*math.sin(gamma)
    y_6 = point[0,1] + e*math.cos(gamma) - 0.5*l*math.sin(gamma)
    
    # append array to include all heliostat positions of a single heliopod
    point = np.append(point, [[x_2, y_2]],axis=0)
    point = np.append(point, [[x_3, y_3]],axis=0)
    point = np.append(point, [[x_4, y_4]],axis=0)
    point = np.append(point, [[x_5, y_5]],axis=0)
    point = np.append(point, [[x_6, y_6]],axis=0)
    
    return point
#%%
def line_between_2(x_1,x_2,y_1,y_2):
    theta = math.atan((y_1-y_2)/(x_1-x_2))
    length = ((x_1-x_2)**2 + (y_1-y_2)**2)**0.5
    
    r = np.linspace(start = 0, stop = length, num = 100)
    
    
    if x_1 > 0:
        x = x_2 + r*math.cos(theta)
        y = y_2 + r*math.sin(theta)
    else:
        x = x_2 - r*math.cos(-theta)
        y = y_2 + r*math.sin(-theta)
    plt.plot(x,y,'g--',linewidth=0.1)
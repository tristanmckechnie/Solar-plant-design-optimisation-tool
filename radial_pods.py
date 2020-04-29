# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:05:08 2020

@author: Tristan mckechnie

Radial layout of pods for field design optimization
"""
import numpy as np
import matplotlib.pyplot as plt
from HeliopodTools import get_x_y_co
from HeliopodTools import heliopod
#%% # of pods on a radius

r = 40 # radius on which to place pod
l_pod = 4.6 # pods side legnth
d_pod = (l_pod/np.sqrt(3))*2 # diameter of circle passing through pods vertices

theta_pod = 2*2*np.arctan((d_pod/2)/r) # angle from tower to tangent of pod circles, ie how much azimuthal spacing a pods circles needs.

# find centres on a radius

allowed_azimuth = 45*np.pi/180 # degrees
pod_centres = [[0,r]] # list of list of pod centres
pods = [] # list of list of heliostats on each pod
count = 0

# heliostats for first (0,radius) pod
heliopod_coords = heliopod(0, r, l_pod, 0, 0, 1)
pods.append(heliopod_coords)

# loop to find pods in allowed angle
while (len(pod_centres)-1)*theta_pod <= allowed_azimuth: # do while pod centres are postive in y 
# for i in range(4):    
    centre_x = r*np.cos(np.pi/2 - theta_pod*(count+1)) # temp x co-ord for pod centres
    centre_y = r*np.sin(np.pi/2 - theta_pod*(count+1)) # temp y co-ords for pod centres
    pod_centres.append([centre_x,centre_y]) # append pod centre co-ords to list
    
    # if count % 2 == 0:
    #     orientation = 1
    # else:
    #     orientation = 0
    
    heliopod_coords = heliopod(centre_x, centre_y, l_pod, 0, 0, 1)
    pods.append(heliopod_coords)
    
    count += 1
    
# mirror around y axis
num_pods = len(pod_centres)      
for i in range(num_pods):  
    if pod_centres[i][0] != 0:
        new_x = pod_centres[i][0]*-1
        new_y = pod_centres[i][1]
        pod_centres.append([new_x, new_y]) # append pod centre co-ords to list        
        
        # if count % 2 == 0:
        #     orientation = 1
        # else:
        #     orientation = 0
        new_heliopod_x = pods[i][:,0] * -1
        new_heliopod_y = pods[i][:,1] 
        
        new_heliopod = np.vstack((new_heliopod_x,new_heliopod_y))
        new_heliopod = np.transpose(new_heliopod)

        pods.append(new_heliopod)
    
# plt.figure(figsize=(10,10))
for i in range(len(pod_centres)):
    plt.plot(pod_centres[i][0],pod_centres[i][1],'go',ms=5)
    circle1 = get_x_y_co([pod_centres[i][0],pod_centres[i][1],d_pod/2])    
    plt.plot(circle1[:,0],circle1[:,1],'b.')
    plt.plot(pods[i][:,0],pods[i][:,1],'ks-')
    
circle2 = get_x_y_co([0,0,r])    
plt.plot(circle2[:,0],circle2[:,1],'r.')
plt.grid(True)
plt.ylim([-50,50])
plt.xlim([-50,50])
plt.axis('equal')
plt.show()

#%% turn above into a function

def pods_on_radius(radius,side_lengths, allowed_angle): # angle must be in degrees
    r = radius # radius on which to place pod
    l_pod = side_lengths # pods side legnth
    d_pod = (l_pod/np.sqrt(3))*2 # diameter of circle passing through pods vertices
    
    theta_pod = 2*np.arctan((d_pod/2)/r) # angle from tower to tangent of pod circles, ie how much azimuthal spacing a pods circles needs.
    
    # find centres on a radius

    allowed_azimuth = allowed_angle*np.pi/180 # degrees
    pod_centres = [[0,r]] # list of list of pod centres
    pods = [] # list of list of heliostats on each pod
    count = 0
    
    # heliostats for first (0,radius) pod
    heliopod_coords = heliopod(0, r, l_pod, 0, 0, 0)
    pods.append(heliopod_coords)
    
    # loop to find pods in allowed angle
    while (len(pod_centres))*theta_pod <= allowed_azimuth: # do while pod centres are postive in y 
 
        if count % 2 == 1:
            centre_x = r*np.cos(np.pi/2 - theta_pod*(count+1)) # temp x co-ord for pod centres
            centre_y = r*np.sin(np.pi/2 - theta_pod*(count+1)) # temp y co-ords for pod centres
        else:
            centre_x = (r-0.25*d_pod)*np.cos(np.pi/2 - theta_pod*(count+1)) # temp x co-ord for pod centres
            centre_y = (r-0.25*d_pod)*np.sin(np.pi/2 - theta_pod*(count+1)) # temp y co-ords for pod centres            
        
        pod_centres.append([centre_x,centre_y]) # append pod centre co-ords to list
        
        if count % 2 == 0:
            orientation = 1
        else:
            orientation = 0
        
        heliopod_coords = heliopod(centre_x, centre_y, l_pod, 0, 0, orientation)
        pods.append(heliopod_coords)
        
        count += 1
        
    # mirror around y axis
    num_pods = len(pod_centres)      
    for i in range(num_pods):  
        if pod_centres[i][0] != 0:
            new_x = pod_centres[i][0]*-1
            new_y = pod_centres[i][1]
            pod_centres.append([new_x, new_y]) # append pod centre co-ords to list        
            
            # if count % 2 == 0:
            #     orientation = 1
            # else:
            #     orientation = 0
            new_heliopod_x = pods[i][:,0] * -1
            new_heliopod_y = pods[i][:,1] 
            
            new_heliopod = np.vstack((new_heliopod_x,new_heliopod_y))
            new_heliopod = np.transpose(new_heliopod)
    
            pods.append(new_heliopod)
        
    # plt.figure(figsize=(10,10))
    for i in range(len(pod_centres)):
        plt.plot(pod_centres[i][0],pod_centres[i][1],'go',ms=5)
        circle1 = get_x_y_co([pod_centres[i][0],pod_centres[i][1],d_pod/2])    
        # plt.plot(circle1[:,0],circle1[:,1],'b.')
        
        pod_array = np.zeros((7,2)) # empty array to plot pods nicely
    
        pod_array[0,:] = pods[i][0,:]
        pod_array[1,:] = pods[i][1,:]
        pod_array[2,:] = pods[i][3,:]
        pod_array[3,:] = pods[i][4,:]
        pod_array[4,:] = pods[i][5,:]
        pod_array[5,:] = pods[i][2,:]
        pod_array[6,:] = pods[i][0,:]
            
        plt.plot(pod_array[:,0],pod_array[:,1],'ks-',ms=4)
        
    circle2 = get_x_y_co([0,0,r])    
    # plt.plot(circle2[:,0],circle2[:,1],'r.')
    plt.grid(True)
    plt.ylim([-50,50])
    plt.xlim([-50,50])
    plt.axis('equal')
    plt.show()
    
    return pods


#%% test function

side1 = 4.6
side2 = 5 
side3 = 6
side4 = 7
side5 = 8 

temp = pods_on_radius(15,side1,90)
radius1 = max(temp[0][:,1]) + (side1/np.sqrt(3)) + (side2/np.sqrt(3))

temp1 = pods_on_radius(radius1,side2,80)
radius2 = max(temp1[0][:,1])  + (side2/np.sqrt(3)) + (side3/np.sqrt(3))

temp2 = pods_on_radius(radius2,side3,70)
radius3 = max(temp2[0][:,1]) + (side3/np.sqrt(3)) + (side4/np.sqrt(3))

temp3 = pods_on_radius(radius3,side4,60)
radius4 = max(temp3[0][:,1]) + (side4/np.sqrt(3)) + (side5/np.sqrt(3))

temp4 = pods_on_radius(radius4,side5,50)

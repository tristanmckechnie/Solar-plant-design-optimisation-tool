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
    # for i in range(len(pod_centres)):
    #     plt.plot(pod_centres[i][0],pod_centres[i][1],'go',ms=5)
    #     circle1 = get_x_y_co([pod_centres[i][0],pod_centres[i][1],d_pod/2])    
    #     # plt.plot(circle1[:,0],circle1[:,1],'b.')
        
    #     pod_array = np.zeros((7,2)) # empty array to plot pods nicely
    
    #     pod_array[0,:] = pods[i][0,:]
    #     pod_array[1,:] = pods[i][1,:]
    #     pod_array[2,:] = pods[i][3,:]
    #     pod_array[3,:] = pods[i][4,:]
    #     pod_array[4,:] = pods[i][5,:]
    #     pod_array[5,:] = pods[i][2,:]
    #     pod_array[6,:] = pods[i][0,:]
            
    #     plt.plot(pod_array[:,0],pod_array[:,1],'ks-',ms=4)
        
    # circle2 = get_x_y_co([0,0,r])    
    # # plt.plot(circle2[:,0],circle2[:,1],'r.')
    # plt.grid(True)
    # plt.ylim([-50,50])
    # plt.xlim([-50,50])
    # plt.axis('equal')
    # plt.show()
    
    return pods


#%% test function

def radial_layout(x):
    
    # assign variables

    angle1 = x[0]
    angle2 = x[1]
    angle3 = x[2]
    angle4 = x[3]
    angle5 = x[4]
    angle6 = x[5]
    angle7 = x[6]
    angle8 = x[7]
    angle9 = x[8]
    angle10 = x[9]
    angle11 = x[10]
    angle12 = x[11]
    
    side1 = x[12]
    side2 = x[13] 
    side3 = x[14]
    side4 = x[15]
    side5 = x[16]
    side6 = x[17]
    side7 = x[18] 
    side8 = x[19]
    side9 = x[20]
    side10 = x[21] 
    side11 = x[22]
    side12 = x[23] 
    
    # generate zones
    
    temp = pods_on_radius(15,side1,angle1)
    radius1 = max(temp[0][:,1]) + (side1/np.sqrt(3)) + (side2/np.sqrt(3))
    
    temp1 = pods_on_radius(radius1,side2,angle2)
    radius2 = max(temp1[0][:,1])  + (side2/np.sqrt(3)) + (side3/np.sqrt(3))
    
    temp2 = pods_on_radius(radius2,side3,angle3)
    radius3 = max(temp2[0][:,1]) + (side3/np.sqrt(3)) + (side4/np.sqrt(3))
    
    temp3 = pods_on_radius(radius3,side4,angle4)
    radius4 = max(temp3[0][:,1]) + (side4/np.sqrt(3)) + (side5/np.sqrt(3))
    
    temp4 = pods_on_radius(radius4,side5,angle5)
    radius5 = max(temp4[0][:,1]) + (side5/np.sqrt(3)) + (side6/np.sqrt(3))
    
    temp5 = pods_on_radius(radius5,side6,angle6)
    radius6 = max(temp5[0][:,1])  + (side6/np.sqrt(3)) + (side7/np.sqrt(3))
    
    temp6 = pods_on_radius(radius6,side7,angle7)
    radius7 = max(temp6[0][:,1]) + (side7/np.sqrt(3)) + (side8/np.sqrt(3))
    
    temp7 = pods_on_radius(radius7,side8,angle8)
    radius8 = max(temp7[0][:,1]) + (side8/np.sqrt(3)) + (side9/np.sqrt(3))
    
    temp8 = pods_on_radius(radius8,side9,angle9)
    radius9 = max(temp8[0][:,1])  + (side9/np.sqrt(3)) + (side10/np.sqrt(3))
    
    temp9 = pods_on_radius(radius9,side10,angle10)
    radius10 = max(temp9[0][:,1]) + (side10/np.sqrt(3)) + (side11/np.sqrt(3))
    
    temp10 = pods_on_radius(radius10,side11,angle11)
    radius11 = max(temp10[0][:,1]) + (side11/np.sqrt(3)) + (side12/np.sqrt(3))
    
    temp11 = pods_on_radius(radius11,side12,angle12)
    
    # create one field list and plot
    
    field = [] # each entry in this list is a zone of heliopod / ring of heliopods
    field.append(temp)
    field.append(temp1)
    field.append(temp2)
    field.append(temp3)
    field.append(temp4)
    field.append(temp5)
    field.append(temp6)
    field.append(temp7)
    field.append(temp8)
    field.append(temp9)
    field.append(temp10)
    field.append(temp11)
    
    # plot field
    
    plt.figure(figsize=(10,10))
    pod_count = 0
    heliostat_field = np.empty((1,2),dtype=float)
    for k in range(len(field)):# plot heliostats on a pod
        for i in range(len(field[k])):
            pod_array = np.empty((7,2),dtype=float)
            
            pod_array[0,:] = field[k][i][0,:]
            pod_array[1,:] = field[k][i][1,:]
            pod_array[2,:] = field[k][i][3,:]
            pod_array[3,:] = field[k][i][4,:]
            pod_array[4,:] = field[k][i][5,:]
            pod_array[5,:] = field[k][i][2,:]
            pod_array[6,:] = field[k][i][0,:]
        
            
            plt.plot(pod_array[:,0],pod_array[:,1]*-1,'ro-',markersize=4)
            
            pod_array = np.delete(pod_array,6,axis=0)
            heliostat_field = np.append(heliostat_field,pod_array,axis=0)
        pod_count = pod_count + i +1
        
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    heliostat_field = np.delete(heliostat_field,0,axis=0) # delete initial empty row
    zeros = np.zeros((len(heliostat_field[:,0]),1))
    heliostat_field = np.hstack((heliostat_field,zeros,zeros))
    heliostat_field[:,1] = heliostat_field[:,1] * -1 # reflect across the x axis
    
    np.savetxt('../data/my_field_tests/positions.csv',heliostat_field,delimiter=",")
    return heliostat_field
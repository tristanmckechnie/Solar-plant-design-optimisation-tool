'''Matti's field layout tool '''

import numpy as np
import matplotlib.pyplot as plt
from HeliopodTools import get_x_y_co
#%%
def matti_dense(radius,vis):
    #boundary conditions: 
    #Field  maximum width and length:
    w_f = 200      #width of the field from edge to edge [m]
    h_f = 300      #length of the field from tower to rear end [m]
    #shape circular, bottom of circle at tower.
    R_shape = radius       #radius of circle [m] #65.8 --> 6000 m2 58 m --> 4594 m2
    n_focal = 8        #number of focal length permitted
    
    #tower coordinates::
    t_x = 0
    t_y = 0
    t_y_c = 1*R_shape    #distance from tower to field centre point
    
    #Pod dimensions:
    w_p = 4.6        #width of a pod [m]
    
    
    #Pod spacing:
    d_p_x = 2.5      #distance between pods in x-direction [m]
    #d_p_y = 2.0 + 0.5*w_p*(3/4)**(0.5)      #distance between pods in y-direction [m]
    d_p_y = 2.0 + 0.5*w_p*(3/4)**(0.5)      #distance between pods in y-direction [m]
    
    
    #linear increased row spacing:
    d_p_lin = 1.00                         #percentage of row spacing depending on distance to tower
    #tower block radius
    r_block = 5                           #radius around tower within no pod centers are permitted
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #generation of field
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    #zero variables
    #number of pods per row, centered
    x_0 = 0
    x = x_0
    y_0 = -h_f
    y = y_0
    i = 1          #running variable for storage matrix
    j = 1
    i_h = 6         #
    
    
    #field centre point coordinates:
    field_x = 0
    field_y = t_y_c
    pod_field = [] # empty list onto which to append individual pods
    row_positions = [] # empty list to keep track of row y postions
    
    while (y) <= (h_f):  #<= (h_f + y_0)
        while x <= (w_f/2):
            #shift x for alternating rows
            row_test = j % 2
            if row_test == 0:
                x_shift = (w_p + d_p_x)/2
    
            else:
                x_shift = 0 
            
            #identify if Pod is within defined circle, pod centre coordinates
            POD_R_x = x + x_shift
            POD_R_y = y + 0.5*w_p*(3/4)**(0.5)
            #compute distance of pod centre to field centre
            R_c = ((field_x - POD_R_x)**2 + (((field_y - POD_R_y)**2)**(0.5))**2)**(0.5)
            
            if R_c < R_shape:
                row_positions.append(y)
                #determine pod focal length
                R_pod = (POD_R_x**2 + POD_R_y**2)**(0.5)   #radial distance of pod centre to receiver centre
                #cutout around tower
                if R_pod >= r_block:     
                    #positioning heliostats upwards pointin
                    #x
                    POD_h = np.empty((6,2),dtype=float) # individual pods
                        
                    POD_h[0,0] = POD_R_x
                    POD_h[1,0] = x - w_p/2 + x_shift
                    POD_h[2,0] = x + w_p/2 + x_shift
                    POD_h[3,0] = x - w_p/4 + x_shift
                    POD_h[4,0] = x + w_p/4 + x_shift
                    POD_h[5,0] = x + x_shift
                    #y
                    POD_h[0,1] = y
                    POD_h[1,1] = y
                    POD_h[2,1] = y
                    POD_h[3,1] = y + 0.5*w_p*(3/4)**(0.5)
                    POD_h[4,1] = y + 0.5*w_p*(3/4)**(0.5)
                    POD_h[5,1] = y + 1.0*w_p*(3/4)**(0.5)
                    
                    # print('POD_h',POD_h)
                    pod_field.append(POD_h)
                else:
                    i = i -1
                
                #keep counting i
                i = i + 1
            
            
            #positioning heliostats downwards pointing
            x_d = x + (w_p + d_p_x)/2 + x_shift  #shift x
            #identify if Pod is within defined circle
            POD_R_x = x_d
            POD_R_y = y + 1.0*w_p*(3/4)*(0.5) + 0.5*w_p*(3/4)**(0.5) - 0.5*w_p*(3/4)**(0.5)
            #compute distance of pod centre
            R_c = ((field_x - POD_R_x)**2 + (((field_y - POD_R_y)**2)**(0.5))**2)**(0.5)
            if R_c < R_shape:
                row_positions.append(y)
                #determine pod focal length
                R_pod = (POD_R_x**2 + POD_R_y**2)**(0.5)   #radial distance of pod centre to receiver 
                # cutout around tower
                if R_pod >= r_block:
              
                    POD_h = np.empty((6,2),dtype=float) # individual pods
                    # x
                    POD_h[0,0] = POD_R_x
                    POD_h[1,0] = x_d - w_p/2
                    POD_h[2,0] = x_d + w_p/2
                    POD_h[3,0] = x_d - w_p/4
                    POD_h[4,0] = x_d + w_p/4
                    POD_h[5,0] = x_d
                    #     y
                    POD_h[0,1] = y + 1.0*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
                    POD_h[1,1] = y + 1.0*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
                    POD_h[2,1] = y + 1.0*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
                    POD_h[3,1] = y + 0.5*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
                    POD_h[4,1] = y + 0.5*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
                    POD_h[5,1] = y + 0.0*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)  
                    
                    pod_field.append(POD_h)                     
                else: 
                    i = i - 1
    
                # keep counting i
                i = i + 1
              
            #update x
            x = x + (w_p + d_p_x)
        
        #reset x-coordinate
        x = x_0
        #update runner
        j = j + 1
        #update y
        y = y*d_p_lin + (w_p + d_p_y)
        # print('Y value:',y)
    
    ###########################################################################
    # add missing centre pods due to Matti's shift
    ###########################################################################
    # add centre pods
    # check row y cords:
    temp = set(row_positions) # extract unique set from list
    temp2 = list(temp) # transform set into list
    temp2.sort() # sort list into ascending order
    
    row_centres = np.arange(2,len(temp2),2)   
    
    for i in row_centres:
    
        POD_R_x = 0
        POD_R_y = temp2[i]
        #compute distance of pod centre to field centre
        R_c = ((field_x - POD_R_x)**2 + (((field_y - POD_R_y)**2)**(0.5))**2)**(0.5)
        R_pod = (POD_R_x**2 + POD_R_y**2)**(0.5)   #radial distance of pod centre to receiver centre
        if R_c < R_shape and R_pod >= r_block :
            POD_h = np.empty((6,2),dtype=float) # individual pods
            #x
            POD_h[0,0] = 0
            POD_h[1,0] = 0 - w_p/2
            POD_h[2,0] = 0 + w_p/2
            POD_h[3,0] = 0 - w_p/4
            POD_h[4,0] = 0 + w_p/4
            POD_h[5,0] = 0
            #y
            POD_h[0,1] = temp2[i] + 1.0*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
            POD_h[1,1] = temp2[i] + 1.0*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
            POD_h[2,1] = temp2[i] + 1.0*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
            POD_h[3,1] = temp2[i] + 0.5*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
            POD_h[4,1] = temp2[i] + 0.5*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
            POD_h[5,1] = temp2[i] + 0.0*w_p*(3/4)**(0.5) + 0.5*w_p*(3/4)**(0.5)
        
            pod_field.append(POD_h)
            # print('Added centre pod')
    ###########################################################################
    # mirror pods across y axis   
    ###########################################################################
    pod_array = np.zeros((6,2),dtype=float)    
    temp = 1
    mirror = [] # empty list to append negative x field to
    for i in range(len(pod_field)):
        
        for k in range(6):
            if pod_field[i][k,0] == 0:
                temp = 0 # dont mirror if pod is on centre
        if temp > 0:
            for m in range(6):
                pod_array[m,0] = pod_field[i][m,0]*-1
            pod_array[:,1] = pod_field[i][:,1]
            mirror.append(pod_array)
            pod_array = np.zeros((6,2),dtype=float)
        temp = 1
    
    for i in range(len(mirror)):
        pod_field.append(mirror[i])
    ###########################################################################
    #% plot mattis field
    ###########################################################################    
    if vis == 1:
        plt.figure()
        for i in range(len(pod_field)):
        # plt.plot(pod_field[i][:,0],pod_field[i][:,1],'r*',markersize=4)
            pod_array = np.empty((7,2),dtype=float)
            
            pod_array[0,:] = pod_field[i][1,:]
            pod_array[1,:] = pod_field[i][0,:]
            pod_array[2,:] = pod_field[i][2,:]
            pod_array[3,:] = pod_field[i][4,:]
            pod_array[4,:] = pod_field[i][5,:]
            pod_array[5,:] = pod_field[i][3,:]
            pod_array[6,:] = pod_field[i][1,:]
            
            plt.plot(pod_array[:,0],pod_array[:,1],'ro-')
            
        bounding_circle = get_x_y_co([field_x,field_y,R_shape])
        plt.plot(bounding_circle[:,0],bounding_circle[:,1],'g.')
        
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        
    ###########################################################################
    #% create positions for sunflower ray tracer
    ###########################################################################         
    heliostat_field = np.zeros((1,2),dtype=float)
    for i in range(len(pod_field)):
        heliostat_field = np.append(heliostat_field,pod_field[i],axis=0)
    heliostat_field = np.delete(heliostat_field,0,axis=0)
    zeros = np.zeros((len(heliostat_field[:,0]),1))
    heliostat_field = np.hstack((heliostat_field,zeros,zeros))
    heliostat_field[:,1] = heliostat_field[:,1] * -1 # reflect across the x axis
    
    np.savetxt('../data/my_plant_config/positions.csv',heliostat_field,delimiter=",")    
    return heliostat_field,pod_field
#%% 




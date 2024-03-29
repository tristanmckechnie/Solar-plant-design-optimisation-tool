import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

from HeliopodTools import get_x_y_co
from HeliopodTools import heliopod_cornfield
import optical_model_class as opt
"""

Author: T McKechnie
Stellenbosch University
1 April 2020

Densely packed,staggered field layout with zones.

field_x : contains the x co-ords of pod centres
field_y : contains the y co-rods of pod centres
field_heliostats : contains the x and y co-ords of heliostats

"""

class dense_zone:
    
    def __init__(self,l,number_rows,width,r_min,tower_x,tower_y,field_y):
        self.l = l                              # pod side length
        self.r = self.l / np.sqrt(3)            # radius of circle into which pod fits        
        self.d_row = (3/4)*np.sqrt(3)*self.r*2  # distance between pod centres in x
        self.d_col = 1.25*self.r                # distance between pod centres in y
        self.width = width                      # allowed width of zone
        self.max_rows = number_rows             # maximum allows rows
        self.r_min = r_min                      # tower exclusion zone
        self.tower_x = tower_x
        self.tower_y = tower_y
        self.field_y_c = field_y       
        self.heliostat_field = []         
        
# =============================================================================        
    # method to generate heliostat positions from pod centres
# =============================================================================
    def pod_from_centre(self,x_c,y_c,l,orientation):
            
        r = l/np.sqrt(3) 
        pod = np.empty((6,2),dtype=float)
        
        if orientation == 'up':
        
            pod[0,0] = x_c
            pod[0,1] = y_c + r
            pod[1,0] = x_c - (np.sqrt(3)/4)*r
            pod[1,1] = y_c + 0.25*r
            pod[2,0] = x_c + (np.sqrt(3)/4)*r
            pod[2,1] = y_c + 0.25*r
            pod[3,0] = x_c - l/2
            pod[3,1] = y_c - 0.5*r
            pod[4,0] = x_c
            pod[4,1] = y_c - 0.5*r
            pod[5,0] = x_c + l/2
            pod[5,1] = y_c - 0.5*r
            
        elif orientation == 'down':
            
            pod[0,0] = x_c
            pod[0,1] = y_c - r
            pod[1,0] = x_c + (np.sqrt(3)/4)*r
            pod[1,1] = y_c - 0.25*r
            pod[2,0] = x_c - (np.sqrt(3)/4)*r
            pod[2,1] = y_c - 0.25*r
            pod[5,0] = x_c - l/2
            pod[5,1] = y_c + 0.5*r
            pod[4,0] = x_c
            pod[4,1] = y_c + 0.5*r
            pod[3,0] = x_c + l/2
            pod[3,1] = y_c + 0.5*r
        
        return pod
        
# =============================================================================        
    # method to place pods in densly packed pattern
# =============================================================================        
    def zone_pattern(self):
        
        field_x = [] # list containing the x co-ords of pod centres
        field_y = [] # list containt the y co-ords of pod centres
        
        row_number = 1 # row counter
        
        # iterate over rows and place and stagger pod centres
        while row_number <= self.max_rows: 
            
            # Determine y co-ord of row
            if row_number % 2 == 0: # even rowed pods
                pod_y = pod_y + self.d_col 
            elif row_number % 2 > 0 and row_number > 1: # odd rowed pods
                pod_y = pod_y + self.d_col + 1.5 # every odd rows pod has extra 1.5 m spacing 
            elif row_number == 1:
                pod_y = self.d_col + self.field_y_c
            
            # Determine first x postion of new row ie to alternate or not
            if row_number % 2 == 0:
                pod_x = np.array([0],dtype=float) # first pod on centre line
            else:
                pod_x = np.array([self.d_row/2],dtype=float) # first pod off centre line 
            
            # re calc first pod on a row if it falls in tower exclusion zone
            while ((pod_x[0]-self.tower_x)**2 + (pod_y-self.tower_y)**2)**0.5 < self.r_min:
                pod_x[0] = pod_x[0] + self.d_row
            
            # Fill a certain row with pods
            while pod_x[-1] < self.width/2 - self.d_row: # place pods while last pod centre x co-ords is less than allowed zone width
                if row_number % 2 == 0:
                    pod_x = np.append(pod_x,pod_x[-1] + self.d_row) 
                else:
                    pod_x = np.append(pod_x,pod_x[-1] + self.d_row)
            
            
            # Save row pod centres to field        
            if row_number == 1:
                field_x.append(pod_x) 
                temp_y = np.full(len(pod_x),pod_y)
                field_y.append(temp_y)
            elif row_number > 1:
                field_x.append(pod_x)
                temp_y = np.full(len(pod_x),pod_y)
                field_y.append(temp_y)        
                
            row_number += 1 # increment row number
            
            # Save negative x pods, ie reflect over y axis
            field_neg_x = []
            field_neg_y= []
            
            for i in range(len(field_x)):
                temp_neg_row = []
                temp_neg_col = []
                for k in range(len(field_x[i])):
                    if field_x[i][k] > 0:
                        temp_neg_row.append(-field_x[i][k])
                        temp_neg_col.append(field_y[i][k])
                field_neg_x.append(temp_neg_row)
                field_neg_y.append(temp_neg_col)

        # determine heliostat positons for each pod
            
        field_heliostats = []
        
        for i in range(len(field_x)):
            for k in range(len(field_x[i])):
                if i % 2 == 0: 
                    field_heliostats.append(self.pod_from_centre(field_x[i][k],field_y[i][k],self.l,'up'))
                else:
                    field_heliostats.append(self.pod_from_centre(field_x[i][k],field_y[i][k],self.l,'down'))
                    
        for i in range(len(field_neg_x)):
            for k in range(len(field_neg_x[i])):
                if i % 2 == 0: 
                    field_heliostats.append(self.pod_from_centre(field_neg_x[i][k],field_neg_y[i][k],self.l,'up'))
                else:
                    field_heliostats.append(self.pod_from_centre(field_neg_x[i][k],field_neg_y[i][k],self.l,'down'))
           
        self.heliostat_field = field_heliostats
        
        return field_x, field_y, field_neg_x, field_neg_y, field_heliostats

# =============================================================================        
    # method to generate plot field layout
# =============================================================================        
    def plot(self):
        
        field_x, field_y, field_neg_x, field_neg_y, field_heliostats = self.zone_pattern() # call zone pattern method
        
        plt.figure()
        for i in range(len(field_x)): # plot pod centres
            plt.plot(field_x[i],field_y[i],'b*')
            
        for i in range(len(field_neg_x)): # plot negative x pod centres
            plt.plot(field_neg_x[i],field_neg_y[i],'bs')
            
        for i in range(len(field_heliostats)):# plot heliostats on a pod
            pod_array = np.empty((7,2),dtype=float)
            
            pod_array[0,:] = field_heliostats[i][0,:]
            pod_array[1,:] = field_heliostats[i][1,:]
            pod_array[2,:] = field_heliostats[i][3,:]
            pod_array[3,:] = field_heliostats[i][4,:]
            pod_array[4,:] = field_heliostats[i][5,:]
            pod_array[5,:] = field_heliostats[i][2,:]
            pod_array[6,:] = field_heliostats[i][0,:]
            
            plt.plot(pod_array[:,0],pod_array[:,1],'ro-')
        
        # plot bounding circles for each heliostats
        radius = ((1.83**2 + 1.22**2)**0.5)/2
        
        for i in range(len(field_heliostats)):
            for k in range(6):
                heliostat_bound_circle = get_x_y_co([field_heliostats[i][k,0],field_heliostats[i][k,1],radius])
                plt.plot(heliostat_bound_circle[:,0],heliostat_bound_circle[:,1],'k.',markersize=0.5)
            
        
        plt.grid(True)
        plt.show()
        plt.axis('equal')

#%% test class
def field_layout(widths):
    zone_1 = dense_zone(4.6, 4, widths[0],8,0,0,0) # initialize class instance
    zone_1.zone_pattern()
    
    x_start_2 = zone_1.d_col*4 + 1.5*2 
    
    zone_2 = dense_zone(4.6, 4, widths[1],8,0,0,x_start_2) 
    zone_2.zone_pattern()
    
    x_start_3 = zone_1.d_col*4 + zone_2.d_col*4 + 1.5*4 
    
    zone_3 = dense_zone(4.6, 4, widths[2],8,0,0,x_start_3) 
    zone_3.zone_pattern()
    
    x_start_4 = zone_1.d_col*4 + zone_2.d_col*4 + zone_3.d_col*4 + 1.5*6
    
    zone_4 = dense_zone(4.6, 4, widths[3],8,0,0,x_start_4)
    zone_4.zone_pattern()
    
    x_start_5 = zone_1.d_col*4 + zone_2.d_col*4 + zone_3.d_col*4 +  zone_4.d_col*4 + 1.5*8
    
    zone_5 = dense_zone(4.6, 4, widths[4],8,0,0,x_start_5) 
    zone_5.zone_pattern()
    
    # zone_6 = dense_zone(4.6, 2, 100,8,0,0,56) 
    # zone_6.zone_pattern()
    
    # zone_7 = dense_zone(4.6, 2, 70,8,0,0,64)
    # zone_7.zone_pattern()
    
    # zone_8 = dense_zone(4.6, 2, 50,8,0,0,72)
    # zone_8.zone_pattern()
    
    # add all zone's pods to one list
    field = []
    field.append(zone_1.heliostat_field)
    field.append(zone_2.heliostat_field)
    field.append(zone_3.heliostat_field)
    field.append(zone_4.heliostat_field)
    field.append(zone_5.heliostat_field)
    # field.append(zone_6.heliostat_field)
    # field.append(zone_7.heliostat_field)
    # field.append(zone_8.heliostat_field)
    
    plt.figure()
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
        
            
            plt.plot(pod_array[:,0],pod_array[:,1],'ro-')
            
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
    
    # run optical simulation 
    
    num_helios = len(heliostat_field)
    
    time_before = time.time()
    # initialize
    test_simulation = opt.optical_model(-33.8,18.8,'north',2,1.83,1.22,[50],41,45,20,4,num_helios,"../code/build/sunflower_tools/Sunflower","../data/my_field_tests") # initiaze object of type optical_model
    
    # set jsons
    test_simulation.heliostat_inputs() # set heliostat parameters
    test_simulation.receiver_inputs()
    test_simulation.tower_inputs()
    test_simulation.raytracer_inputs()
    # get optical efficiency results
    efficencies, year_sun_angles = test_simulation.annual_hourly_optical_eff()
    time_after =  time.time()
    print('Total simulation time for a single configuration: ',time_after-time_before)
    
    # import dni csv
    dni = np.genfromtxt('SUNREC_hour.csv',delimiter=',')
    receiver_power = dni*efficencies*num_helios*1.83*1.22
    
    annual_eta = sum(receiver_power)/sum(num_helios*1.83*1.22*dni)

    return annual_eta, receiver_power[1907],year_sun_angles

#%% call field layout and optical model tools

eff, noon_power, yearly_sun_angles = field_layout([80,100,100,100,60])
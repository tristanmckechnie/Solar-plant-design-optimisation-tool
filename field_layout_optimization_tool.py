import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

from HeliopodTools import get_x_y_co
from HeliopodTools import heliopod_cornfield
import optical_model_class as opt
import Dispatch_optimization_class_for_import as disp

from scipy.optimize import minimize

import subprocess as sp
import json
from sun_pos import *


# needed for 3d interpolation
from scipy.interpolate import LinearNDInterpolator
from mpl_toolkits import mplot3d
"""

Author: T McKechnie
Stellenbosch University
1 April 2020

Densely packed,staggered field layout with zones.

field_x : contains the x co-ords of pod centres
field_y : contains the y co-rods of pod centres
field_heliostats : contains the x and y co-ords of heliostats

"""
#%% zone layout algorithm
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

#%% field layout simulation tool

def field_layout_sim(width):

    # =============================================================================
    # Field layout generation 
    # =============================================================================
    
    heliostat_field = field_layout(width)
    
    # =============================================================================    
    # run optical simulation 
    # =============================================================================    
    num_helios = len(heliostat_field)
    
    time_before = time.time()
    # initialize
    test_simulation = opt.optical_model(-27.24,22.902,'north',2,1.83,1.22,[50],21,45,20,4,num_helios,"../code/build/sunflower_tools/Sunflower","../data/my_field_tests") # initiaze object of type optical_model
    
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
    dni = np.genfromtxt('Kalagadi_Manganese-hour.csv',delimiter=',')
    receiver_power = dni*efficencies*num_helios*1.83*1.22
    
    # limit receiver power to 2.5 MWth and apply efficiency
    
    for i in range(len(receiver_power)):
        receiver_power[i] = receiver_power[i] * 0.9 
        if receiver_power[i] > 2500000:
            receiver_power[i] = 2500000
    
    annual_eta = sum(dni*efficencies*num_helios*1.83*1.22)/sum(num_helios*1.83*1.22*dni)
    
    # =============================================================================    
    # dispatch optimization section
    # =============================================================================  
    
    eta = efficencies
    receiver_data = receiver_power
    
    # Single plant configuration dispatch optimization and economics
    
    start = 0
    days = 360
    # receiver_data = np.genfromtxt('kalagadi_field_output_new_efficiency.csv',delimiter=',') # note this is the new receiver efficiency applied in the excel sheet. 
    tariff_data = np.genfromtxt('kalagadi_extended_tariff.csv',delimiter=',')#tariff_data = np.load('./data/megaflex_tariff.npy') #
    time_horizon = 48
    process_output = 0.85e6
    TES_hours = 14
    E_start = 0
    no_helios = num_helios
    penality_val = 0
    
    # create object instance and time simulation
    bloob_slsqp = disp.Dispatch(start,days,receiver_data,tariff_data,time_horizon,process_output, TES_hours, E_start, no_helios,penality_val)
    
    # run rolling time-horizon optimization object method
    start_clock = time.process_time()
    # bloob_mmfd.rolling_optimization('dot','zeros',0) # run mmfd with random starting guesses
    bloob_slsqp.rolling_optimization('scipy','zeros',0) # run slsqp with mmfd starting guesses
    end_clock = time.process_time()
    
    # run plotting
    # optimal_cost_temp, heuristic_cost_temp,Cummulative_TES_discharge,Cummulative_Receiver_thermal_energy,Cummulative_dumped_heat = bloob_slsqp.plotting()
    
    # costs
    cum_optical_cost, cum_heuristic_cost, optimal_cost_temp, heuristic_cost_temp = bloob_slsqp.strategy_costs()
    end_clock = time.process_time()
    
    print('########################################################################')
    print('Rolling time-horizon')
    print('Computational expense: ', end_clock - start_clock)
    print('Heuristic cost: ', heuristic_cost_temp)
    print('Optimal cost: ',optimal_cost_temp)
    print('########################################################################')
    
    annual_heat_gen = sum(bloob_slsqp.discharge*bloob_slsqp.thermal_normalization)
    
    # =============================================================================
    #     LCOH calculation
    # =============================================================================
    
    n = 25;
    
    #% CAPEX values
    
    CAPEX_tower = 8288 + 1.73*(40**2.75);
    CAPEX_vert_transport = 140892;
    CAPEX_horz_transport = 248634;
    
    CAPEX_TES = 20443*TES_hours*process_output/1e6; #% where TES is represented in MWh_t's
    
    CAPEX_receiver = 138130;
    
    CAPEX_heliostat = 112.5*1.83*1.22*no_helios #% where Asf is the aperature area of the solar field
    
    CAPEX_HE = 138130*process_output/1e6#% where HE is the kWt of the heat exchanger.
    
    Total_CAPEX = CAPEX_tower + CAPEX_vert_transport + CAPEX_horz_transport + CAPEX_TES + CAPEX_receiver + CAPEX_heliostat + CAPEX_HE;
    
    #% OPEX
    
    OM = 0.039*Total_CAPEX;
    indirect_costs = 0.22*Total_CAPEX;
    
    #% capitcal recovery factor
    kd = 0.07;
    k_ins = 0.01;
    CRF = ((kd*(1+kd)**n)/((1+kd)**n -1)) + k_ins;
    
    #% LCOH 
    
    LCOH_s = ((Total_CAPEX+ indirect_costs)*CRF + OM )/(annual_heat_gen/1e6);
    
    # LCOH electric
    
    annual_elec_gen = days*24*process_output/1e6 - annual_heat_gen/1e6
    
    LCOH_e = (optimal_cost_temp/14.5)/annual_elec_gen # optimal_cost_temp !!!! Remember that costs for optimal cost etc is in Rands so must convert to dollar!!!!
    
    # LCOH combined
    
    LCOH = ((LCOH_e*annual_elec_gen) + (LCOH_s*annual_heat_gen/1e6) )/ (days*24*process_output/1e6)
    
    print('########################################################################')
    print('Solar LCOH: ', LCOH_s)
    print('Electric LCOH, ', LCOH_e)
    print('Combined LCOH, ', LCOH)
    print('Electric heat generated, ', annual_elec_gen)
    print('Solar heat generated, ', annual_heat_gen/1e6)
    print('########################################################################')
    
    return annual_eta, receiver_power[1907],year_sun_angles,LCOH,no_helios

def field_layout(width):
    
    widths = np.zeros((15,1))
    for i in range(15):
        widths[i] = width[i]*160
    # widths[10] = 4.6#width[10]*10
    # widths[11] = 4.6#wwidth[11]*10  
    # widths[12] = 4.6#wwidth[12]*10
    # widths[13] = 4.6#wwidth[13]*10
    # widths[14] = 4.6#wwidth[14]*10
    
    zone_1 = dense_zone(4.6, 2, widths[0],8,0,0,0) # initialize class instance
    zone_1.zone_pattern()
    
    x_start_2 = zone_1.d_col*2 + 1.5
    
    zone_2 = dense_zone(4.6, 2, widths[1],8,0,0,x_start_2) 
    zone_2.zone_pattern()
    
    x_start_3 = zone_1.d_col*2 + zone_2.d_col*2 + 1.5*2
    
    zone_3 = dense_zone(4.6, 2, widths[2],8,0,0,x_start_3) 
    zone_3.zone_pattern()
    
    x_start_4 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 + 1.5*3
    
    zone_4 = dense_zone(4.6, 2, widths[3],8,0,0,x_start_4)
    zone_4.zone_pattern()
    
    x_start_5 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + 1.5*4
    
    zone_5 = dense_zone(4.6, 2, widths[4],8,0,0,x_start_5) 
    zone_5.zone_pattern()
    
    x_start_6 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + 1.5*5
    
    zone_6 = dense_zone(4.6, 2, widths[5],8,0,0,x_start_6) 
    zone_6.zone_pattern()
    
    x_start_7 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + zone_6.d_col*2 + 1.5*6
    
    zone_7 = dense_zone(4.6, 2, widths[6],8,0,0,x_start_7)
    zone_7.zone_pattern()
    
    x_start_8 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + zone_6.d_col*2 + zone_7.d_col*2 + 1.5*7
    
    zone_8 = dense_zone(4.6, 2, widths[7],8,0,0,x_start_8)
    zone_8.zone_pattern()
    
    x_start_9 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + zone_6.d_col*2 + zone_7.d_col*2 + zone_8.d_col*2 + 1.5*8
    
    zone_9 = dense_zone(4.6, 2, widths[8],8,0,0,x_start_9)
    zone_9.zone_pattern()
    
    x_start_10 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + zone_6.d_col*2 + zone_7.d_col*2 + zone_8.d_col*2 + zone_9.d_col*2 + 1.5*9
    
    zone_10 = dense_zone(4.6, 2, widths[9],8,0,0,x_start_10)
    zone_10.zone_pattern()
    
    x_start_11 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + zone_6.d_col*2 + zone_7.d_col*2 + zone_8.d_col*2 + zone_9.d_col*2 + zone_10.d_col*2  + 1.5*10
    
    zone_11 = dense_zone(4.6, 2, widths[10],8,0,0,x_start_11) 
    zone_11.zone_pattern()
    
    x_start_12 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + zone_6.d_col*2 + zone_7.d_col*2 + zone_8.d_col*2 + zone_9.d_col*2 + zone_10.d_col*2 + zone_11.d_col*2  + 1.5*11 
    
    zone_12 = dense_zone(4.6, 2, widths[11],8,0,0,x_start_12)
    zone_12.zone_pattern()
    
    x_start_13 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + zone_6.d_col*2 + zone_7.d_col*2 + zone_8.d_col*2 + zone_9.d_col*2 + zone_10.d_col*2 + zone_11.d_col*2 + zone_12.d_col*2  + 1.5*12
    
    zone_13 = dense_zone(4.6, 2, widths[12],8,0,0,x_start_13)
    zone_13.zone_pattern()
    
    x_start_14 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + zone_6.d_col*2 + zone_7.d_col*2 + zone_8.d_col*2 + zone_9.d_col*2 + zone_10.d_col*2 + zone_11.d_col*2 + zone_12.d_col*2 + zone_13.d_col*2  + 1.5*13
    
    zone_14 = dense_zone(4.6, 2, widths[13],8,0,0,x_start_14)
    zone_14.zone_pattern()
    
    x_start_15 = zone_1.d_col*2 + zone_2.d_col*2 + zone_3.d_col*2 +  zone_4.d_col*2 + zone_5.d_col*2 + zone_6.d_col*2 + zone_7.d_col*2 + zone_8.d_col*2 + zone_9.d_col*2 + zone_10.d_col*2 + zone_11.d_col*2 + zone_12.d_col*2 + zone_13.d_col*2 + zone_14.d_col*2  + 1.5*14                                              
    
    zone_15 = dense_zone(4.6, 2, widths[14],8,0,0,x_start_15)
    zone_15.zone_pattern()
    
    # add all zone's pods to one list
    field = []
    field.append(zone_1.heliostat_field)
    field.append(zone_2.heliostat_field)
    field.append(zone_3.heliostat_field)
    field.append(zone_4.heliostat_field)
    field.append(zone_5.heliostat_field)
    field.append(zone_6.heliostat_field)
    field.append(zone_7.heliostat_field)
    field.append(zone_8.heliostat_field)
    field.append(zone_9.heliostat_field)
    field.append(zone_10.heliostat_field)
    field.append(zone_11.heliostat_field)
    field.append(zone_12.heliostat_field)
    field.append(zone_13.heliostat_field)
    field.append(zone_14.heliostat_field)
    field.append(zone_15.heliostat_field)
    
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
    # plt.axis('equal')
    plt.show()
    
    heliostat_field = np.delete(heliostat_field,0,axis=0) # delete initial empty row
    zeros = np.zeros((len(heliostat_field[:,0]),1))
    heliostat_field = np.hstack((heliostat_field,zeros,zeros))
    heliostat_field[:,1] = heliostat_field[:,1] * -1 # reflect across the x axis
    
    np.savetxt('../data/my_field_tests/positions.csv',heliostat_field,delimiter=",")
    # print('Number of helios: ',len(heliostat_field) )
    return heliostat_field

#%% single moment simulation for contraint 

def square_to_circle(flux_map,num_elements):
    # receiver area
    area = 1 # m^2
    length = (area/np.pi)**0.5
    
    # grid
    n = num_elements # number of points on the grid
    x = np.linspace(-length+length/(n), length-length/(n),n)
    y = np.linspace(-length+length/(n), length-length/(n),n)
    gap = ((x[0]-x[1])**2)**0.5 # gap between points
    
    X, Y = np.meshgrid(x,y)
    
    # co-ordinates of square centers
    X_vec = np.reshape(X,(n**2,1))
    Y_vec = np.reshape(Y,(n**2,1))
        
    # find valid squares
    invalid = np.full((n**2),1)
    count = 0
    for i in range(len(X_vec)):
            if (X_vec[i]**2 + Y_vec[i]**2)**0.5 > length:
                invalid[i] = 0
                count += 1
    area = (2*length/n)**2
    # turn invalid in 2D array matching the flux_map data
    invalid = invalid.reshape(num_elements,num_elements)
    
    flux_circle = invalid*flux_map 
    
    return flux_circle
    

# settings and directory
def single_moment_simulation():
    simulation = 'my_field_tests' #'cornfield_layout'#'helio100_modified_field'# name of file in \data that contains the jsons for input to the ray tracer
    
    args_settings = '--settings=../data/' + simulation
    args_weather = '--weather=../data/' + simulation + '/capetown.epw'
    #total_power = np.zeros((4,1),float)
    
    # loop flux analysis 
    
    # read and write JSON settings files
    with open('../data/' + simulation + '/moments.json') as moment: #open json and change angles and DNI value for the moment simulation
        file = json.load(moment)
        
    file['momentsdata'][0]['azimuth'] = 0# represents design days data
    file['momentsdata'][0]['altitude'] = 63.3
    file['momentsdata'][0]['irradiation'] = 1000
    
    with open('../data/' + simulation + '/moments.json','w') as raytracer: #save as json file (serialization)
        json.dump(file,raytracer,indent = 4)
      
    #Run ray-tracer simulation
    p1 = sp.run(['../code/build/sunflower_tools/Sunflower', args_settings, args_weather])
    
    # retrieve moment simulation results flux map on rectangular plane
    flux_map = np.genfromtxt('../data/'+ simulation +'/fluxmap.csv',delimiter=',') #
    flux_map = np.delete(flux_map,20,1) # remove arb column of nan's
    
    # # call square to solar correction function
    flux_map_circle = square_to_circle(flux_map,20) # !!! Remember to choose correct number of elements !!!
    
    moment_power = sum(sum(flux_map_circle))
    # moment_power = sum(sum(flux_map))
    print('Moment power at solar noon, equinox:', moment_power)


    return moment_power
#%% field layout optimization

obj_func = []

def objective(x):
    eff, noon_power, yearly_sun_angles, LCOH, no_helios = field_layout_sim(x)
    print('*****************************************************************')
    print('Guesses:',x,' Efficiency:', eff, ' LCOH_{comb}: ', LCOH, '# Helios: ', no_helios)
    print('*****************************************************************')
    
    obj_func.append(LCOH)
    
    return LCOH/60 

def constraint1(x):
    field = field_layout(x)
    moment_power = single_moment_simulation()
    return (moment_power/1e6) - (2500000/1e6)

def constraint2(x):
    return x-80/160 

def constraint3(x):
    return 160/160 - x


con1 = {'type': 'eq','fun': constraint1}
con2 = {'type': 'ineq','fun': constraint2}
con3 = {'type': 'ineq','fun': constraint3}

con = [con3]

bound = (0,160/160)
bnds = np.full((15,2),bound)
# bnds = np.append(bnds,[[4.6/10,10/10]],axis=0)
# bnds = np.append(bnds,[[4.6/10,10/10]],axis=0)
# bnds = np.append(bnds,[[4.6/10,10/10]],axis=0)
# bnds = np.append(bnds,[[4.6/10,10/10]],axis=0)
# bnds = np.append(bnds,[[4.6/10,10/10]],axis=0)
x0 = [ 0.66177231,  0.7117486 ,  0.90000414,  0.85565188,  0.85886699,
        0.83229603,  0.82838414,  0.76814612,  0.47815746,  0.44811863,
        0.56748397,  0.29013976, 0.21504752, 0.04065081,  0.10730469] #,0.46,0.46,0.46,0.46,0.46 divided through by 160 ie max bounds ,0.76022055, 0.82678298, 0.83880648, 0.85134496, 0.99735851
time_before = time.time()
result = minimize(objective,x0,method='SLSQP',bounds=bnds,tol=1e-5,options={'maxiter':150,'disp': True,'eps':0.1}) # 'eps':0.5'rhobeg':30/160
time_after = time.time()
print(result)
print('Optimization runtime: ', time_after - time_before)

#%% Field layout simulation for single simulaiton
# width = [0.74001329, 0.74040735 ,0.6760284 , 0.66625524 ,0.65284402, 0.47707358 ,0.81066323 ,0.60202479, 0.57302514 ,0.54827877]
# =============================================================================
# Field layout generation 
# =============================================================================

tower_height = 40

heliostat_field = field_layout([0.6, 0.62, 0.65, 0.65, 0.7,
       0.9 , 0.8, 0.7, 0.65, 0.54])

# =============================================================================    
# run optical simulation 
# =============================================================================    
num_helios = len(heliostat_field)

time_before = time.time()
# initialize
test_simulation = opt.optical_model(-27.24,22.902,'north',2,1.83,1.22,[50],tower_height,45,20,4,num_helios,"../code/build/sunflower_tools/Sunflower","../data/my_field_tests") # initiaze object of type optical_model

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
dni = np.genfromtxt('Kalagadi_Manganese-hour.csv',delimiter=',')
receiver_power = dni*efficencies*num_helios*1.83*1.22

annual_eta = sum(dni*efficencies*num_helios*1.83*1.22)/sum(num_helios*1.83*1.22*dni)

# limit receiver power to 2.5 MWth and apply efficiency
    
for i in range(len(receiver_power)):
    receiver_power[i] = receiver_power[i] * 0.9 
    if receiver_power[i] > 2500000:
        receiver_power[i] = 2500000

# =============================================================================    
# dispatch optimization section
# =============================================================================  

eta = efficencies
receiver_data = receiver_power

# Single plant configuration dispatch optimization and economics

start = 0
days = 360
# receiver_data = np.genfromtxt('kalagadi_field_output_new_efficiency.csv',delimiter=',') # note this is the new receiver efficiency applied in the excel sheet. 
tariff_data = np.genfromtxt('kalagadi_extended_tariff.csv',delimiter=',')#tariff_data = np.load('./data/megaflex_tariff.npy') #
time_horizon = 48
process_output = 0.85e6
TES_hours = 14
E_start = 0
no_helios = num_helios
penality_val = 0

# create object instance and time simulation
bloob_slsqp = disp.Dispatch(start,days,receiver_data,tariff_data,time_horizon,process_output, TES_hours, E_start, no_helios,penality_val)

# run rolling time-horizon optimization object method
start_clock = time.process_time()
# bloob_mmfd.rolling_optimization('dot','zeros',0) # run mmfd with random starting guesses
bloob_slsqp.rolling_optimization('scipy','zeros',0) # run slsqp with mmfd starting guesses
end_clock = time.process_time()

# run plotting
optimal_cost_temp, heuristic_cost_temp,Cummulative_TES_discharge,Cummulative_Receiver_thermal_energy,Cummulative_dumped_heat = bloob_slsqp.plotting()

# costs
# cum_optical_cost, cum_heuristic_cost, optimal_cost_temp, heuristic_cost_temp = bloob_slsqp.strategy_costs()
end_clock = time.process_time()

print('########################################################################')
print('Rolling time-horizon')
print('Computational expense: ', end_clock - start_clock)
print('Heuristic cost: ', heuristic_cost_temp)
print('Optimal cost: ',optimal_cost_temp)
print('########################################################################')

annual_heat_gen = sum(bloob_slsqp.discharge*bloob_slsqp.thermal_normalization)

# =============================================================================
#     LCOH calculation
# =============================================================================

n = 25;

#% CAPEX values

CAPEX_tower = 8288 + 1.73*(tower_height**2.75);
CAPEX_vert_transport = 140892;
CAPEX_horz_transport = 248634;

CAPEX_TES = 20443*TES_hours*process_output/1e6; #% where TES is represented in MWh_t's

CAPEX_receiver = 138130;

CAPEX_heliostat = 112.5*1.83*1.22*no_helios #% where Asf is the aperature area of the solar field

CAPEX_HE = 138130*process_output/1e6#% where HE is the kWt of the heat exchanger.

Total_CAPEX = CAPEX_tower + CAPEX_vert_transport + CAPEX_horz_transport + CAPEX_TES + CAPEX_receiver + CAPEX_heliostat + CAPEX_HE;

#% OPEX

OM = 0.039*Total_CAPEX;
indirect_costs = 0.22*Total_CAPEX;

#% capitcal recovery factor
kd = 0.07;
k_ins = 0.01;
CRF = ((kd*(1+kd)**n)/((1+kd)**n -1)) + k_ins;

#% LCOH 

LCOH_s = ((Total_CAPEX+ indirect_costs)*CRF + OM )/(annual_heat_gen/1e6);

# LCOH electric

annual_elec_gen = days*24*process_output/1e6 - annual_heat_gen/1e6

LCOH_e = (optimal_cost_temp/14.5)/annual_elec_gen # optimal_cost_temp !!!! Remember that costs for optimal cost etc is in Rands so must convert to dollar!!!!

# LCOH combined

LCOH = ((LCOH_e*annual_elec_gen) + (LCOH_s*annual_heat_gen/1e6) )/ (days*24*process_output/1e6)

print('########################################################################')
print('Solar LCOH: ', LCOH_s)
print('Electric LCOH, ', LCOH_e)
print('Combined LCOH, ', LCOH)
print('Electric heat generated, ', annual_elec_gen)
print('Solar heat generated, ', annual_heat_gen/1e6)
print('########################################################################')

#%% Field layout simulation for single simulaiton given solarpilot field

# =============================================================================
# Field layout generation 
# =============================================================================

# paste field layout into postitions.csv in correct data sub-directory

heliostat_field = np.genfromtxt('../data/my_field_tests/positions.csv',delimiter=',')

plt.figure()
plt.plot(heliostat_field[:,0],heliostat_field[:,1],'ro',ms=3)
plt.grid(True)
plt.show()

# =============================================================================    
# run optical simulation 
# =============================================================================    
num_helios = len(heliostat_field)

time_before = time.time()
# initialize
test_simulation = opt.optical_model(-27.22,22.902,'north',2,1.83,1.22,[50],20,45,20,4,num_helios,"../code/build/sunflower_tools/Sunflower","../data/my_field_tests") # initiaze object of type optical_model

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
dni = np.genfromtxt('Kalagadi_Manganese-hour.csv',delimiter=',')
receiver_power = dni*efficencies*num_helios*1.83*1.22

annual_eta = sum(dni*efficencies*num_helios*1.83*1.22)/sum(num_helios*1.83*1.22*dni)

# limit receiver power to 2.5 MWth and apply efficiency
    
for i in range(len(receiver_power)):
    receiver_power[i] = receiver_power[i] * 0.9 
    if receiver_power[i] > 2500000:
        receiver_power[i] = 2500000

# =============================================================================    
# dispatch optimization section
# =============================================================================  

eta = efficencies
receiver_data = receiver_power

# Single plant configuration dispatch optimization and economics

start = 0
days = 360
# receiver_data = np.genfromtxt('kalagadi_field_output_new_efficiency.csv',delimiter=',') # note this is the new receiver efficiency applied in the excel sheet. 
tariff_data = np.genfromtxt('kalagadi_extended_tariff.csv',delimiter=',')#tariff_data = np.load('./data/megaflex_tariff.npy') #
time_horizon = 48
process_output = 0.85e6
TES_hours = 14
E_start = 0
no_helios = num_helios
penality_val = 0

# create object instance and time simulation
bloob_slsqp = disp.Dispatch(start,days,receiver_data,tariff_data,time_horizon,process_output, TES_hours, E_start, no_helios,penality_val)

# run rolling time-horizon optimization object method
start_clock = time.process_time()
# bloob_mmfd.rolling_optimization('dot','zeros',0) # run mmfd with random starting guesses
bloob_slsqp.rolling_optimization('scipy','zeros',0) # run slsqp with mmfd starting guesses
# end_clock = time.process_time()

# run plotting
optimal_cost_temp, heuristic_cost_temp,Cummulative_TES_discharge,Cummulative_Receiver_thermal_energy,Cummulative_dumped_heat = bloob_slsqp.plotting()

# costs
# cum_optical_cost, cum_heuristic_cost, optimal_cost_temp, heuristic_cost_temp = bloob_slsqp.strategy_costs()
end_clock = time.process_time()

print('########################################################################')
print('Rolling time-horizon')
print('Computational expense: ', end_clock - start_clock)
print('Heuristic cost: ', heuristic_cost_temp)
print('Optimal cost: ',optimal_cost_temp)
print('########################################################################')

annual_heat_gen = sum(bloob_slsqp.discharge*bloob_slsqp.thermal_normalization)

# =============================================================================
#     LCOH calculation
# =============================================================================

n = 25;

#% CAPEX values

CAPEX_tower = 8288 + 1.73*(20**2.75);
CAPEX_vert_transport = 140892;
CAPEX_horz_transport = 248634;

CAPEX_TES = 20443*TES_hours*process_output/1e6; #% where TES is represented in MWh_t's

CAPEX_receiver = 138130;

CAPEX_heliostat = 112.5*1.83*1.22*no_helios #% where Asf is the aperature area of the solar field

CAPEX_HE = 138130*process_output/1e6#% where HE is the kWt of the heat exchanger.

Total_CAPEX = CAPEX_tower + CAPEX_vert_transport + CAPEX_horz_transport + CAPEX_TES + CAPEX_receiver + CAPEX_heliostat + CAPEX_HE;

#% OPEX

OM = 0.039*Total_CAPEX;
indirect_costs = 0.22*Total_CAPEX;

#% capitcal recovery factor
kd = 0.07;
k_ins = 0.01;
CRF = ((kd*(1+kd)**n)/((1+kd)**n -1)) + k_ins;

#% LCOH 

LCOH_s = ((Total_CAPEX+ indirect_costs)*CRF + OM )/(annual_heat_gen/1e6);

# LCOH electric

annual_elec_gen = days*24*process_output/1e6 - annual_heat_gen/1e6

LCOH_e = (optimal_cost_temp/14.5)/annual_elec_gen # optimal_cost_temp !!!! Remember that costs for optimal cost etc is in Rands so must convert to dollar!!!!

# LCOH combined

LCOH = ((LCOH_e*annual_elec_gen) + (LCOH_s*annual_heat_gen/1e6) )/ (days*24*process_output/1e6)

print('########################################################################')
print('Solar LCOH: ', LCOH_s)
print('Electric LCOH, ', LCOH_e)
print('Combined LCOH, ', LCOH)
print('Electric heat generated, ', annual_elec_gen)
print('Solar heat generated, ', annual_heat_gen/1e6)
print('########################################################################')

#%% creat resource array

resource_array = np.zeros((8760,4))

for i in range(8760):
    resource_array[i,0] = year_sun_angles[i,0]
    resource_array[i,1] = year_sun_angles[i,1]
    resource_array[i,2] = efficencies[i]
    resource_array[i,3] = dni[i]
    
plt.figure()
plt.plot(range(8760),dni/1000,label='dni')
plt.plot(range(8760),resource_array[:,1]/90,label='elavation angle')
plt.grid(True)
plt.legend()

plt.show()



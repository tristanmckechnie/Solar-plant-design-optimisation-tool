#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:09:04 2020

@author: Tristan Mckechnie

This scrip performs a plant configuration optimization. Field size (via radius), hours of TES and output to preheater are design variables.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import optical_model_class as opt
import Dispatch_optimization_class_for_import as disp
from scipy.optimize import minimize
from matti_field import matti_dense
#%% field layout simulation tool

def plant_configuration_sim(x):

    Radius = x[0]
    tes_hours = x[1] 
    Output = x[2]
    
    # =============================================================================
    # Field layout generation 
    # =============================================================================
    
    heliostat_field, pod_field = matti_dense(Radius*80,1)
    time.sleep(5)
    # =============================================================================    
    # run optical simulation 
    # =============================================================================    
    num_helios = len(heliostat_field)
    
    time_before = time.time()
    # initialize
    test_simulation = opt.optical_model(-27.24,22.902,'north',2,1.83,1.22,[50],21,45,20,4,num_helios,"../code/build/sunflower_tools/Sunflower","../data/my_plant_config") # initiaze object of type optical_model
    
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
    process_output = (Output*2)*1e6
    TES_hours = tes_hours*20
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




#%% field layout optimization

obj_func = []

def objective(x):
    eff, noon_power, yearly_sun_angles, LCOH_obj, no_helios = plant_configuration_sim(x)
    print('*****************************************************************')
    print('Guesses:',x,' Efficiency:', eff, ' LCOH_{comb}: ', LCOH_obj, '# Helios: ', no_helios)
    print('*****************************************************************')
    obj_func.append(LCOH_obj)
    return LCOH_obj/60 

# def constraint1(x):
#     field = field_layout(x)
#     moment_power = single_moment_simulation()
#     return (moment_power/1e6) - (2500000/1e6)

# def constraint2(x):
#     return x-80/160 

# def constraint3(x):
#     return 160/160 - x


# con1 = {'type': 'eq','fun': constraint1}
# con2 = {'type': 'ineq','fun': constraint2}
# con3 = {'type': 'ineq','fun': constraint3}

# con = [con3]


bnds = np.array([[0,1],[0,1],[0,1]])
x0 = [52/80,14/20,0.85/2] #,0.46,0.46,0.46,0.46,0.46 divided through by 160 ie max bounds ,0.76022055, 0.82678298, 0.83880648, 0.85134496, 0.99735851
time_before = time.time()
result = minimize(objective,x0,method='SLSQP',tol=1e-5,bounds=bnds,options={'maxiter':80,'disp': True,}) # 'rhobeg':30/160
time_after = time.time()
print(result)
print('Optimization runtime: ', time_after - time_before)

#%% loop matti dense function

for i in np.arange(20,100,10):
    matti_dense(i,1)
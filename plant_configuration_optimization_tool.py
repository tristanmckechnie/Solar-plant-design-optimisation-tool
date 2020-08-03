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

def plant_configuration_sim(x,sunflower_heliostat_field,sunflower_tower_height,sunflower_tilt_angle):
    
    tower_height = sunflower_tower_height

    Radius = x[0]
    tes_hours = x[1] 
    Output = x[2]
    
    # =============================================================================
    # Field layout generation 
    # =============================================================================
    # heliostat_field = sunflower_heliostat_field
    # np.savetxt('../data/my_plant_config/positions.csv',heliostat_field,delimiter=",")
    heliostat_field, pod_field = matti_dense(Radius*80,1)
    # time.sleep(5)
    # =============================================================================    
    # run optical simulation 
    # =============================================================================    
    num_helios = len(heliostat_field)
    
    time_before = time.time()
    # initialize
    test_simulation = opt.optical_model(-27.24,22.902,'north',2,1.83,1.22,[50],tower_height,sunflower_tilt_angle,20,4,num_helios,"../code/build/sunflower_tools/Sunflower","../data/my_plant_config") # initiaze object of type optical_model
    
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
        if receiver_power[i] > 2500000: #/0.9
            receiver_power[i] = 2500000 #/0.9
    
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
    
    return annual_eta, receiver_power[1907],year_sun_angles,LCOH,no_helios,test_simulation.resource




#%% field layout optimization

# obj_func = []

# def objective(x):
#     eff, noon_power, yearly_sun_angles, LCOH_obj, no_helios = plant_configuration_sim(x)
#     print('*****************************************************************')
#     print('Guesses:',x,' Efficiency:', eff, ' LCOH_{comb}: ', LCOH_obj, '# Helios: ', no_helios)
#     print('*****************************************************************')
#     obj_func.append(LCOH_obj)
#     return LCOH_obj/60 

# # def constraint1(x):
# #     field = field_layout(x)
# #     moment_power = single_moment_simulation()
# #     return (moment_power/1e6) - (2500000/1e6)

# # def constraint2(x):
# #     return x-80/160 

# # def constraint3(x):
# #     return 160/160 - x


# # con1 = {'type': 'eq','fun': constraint1}
# # con2 = {'type': 'ineq','fun': constraint2}
# # con3 = {'type': 'ineq','fun': constraint3}

# # con = [con3]


# bnds = np.array([[0,1],[0,1],[0,1]])
# x0 = [52/80,14/20,0.85/2] #,0.46,0.46,0.46,0.46,0.46 divided through by 160 ie max bounds ,0.76022055, 0.82678298, 0.83880648, 0.85134496, 0.99735851
# time_before = time.time()
# result = minimize(objective,x0,method='SLSQP',tol=1e-5,bounds=bnds,options={'maxiter':80,'disp': True,}) # 'rhobeg':30/160
# time_after = time.time()
# print(result)
# print('Optimization runtime: ', time_after - time_before)

#%% sunflower parametric studies recreation

# tower height studies 
tower_height_fields = np.genfromtxt('/home/tristan/Documents/Sunflower_parametric_studies/receiver_height_parametric/tower_height_param_fields.csv',delimiter=',')

# determine number of heliostats in each field
field_length = []
for k in [1,3,5,7,9,11]:
    for i in range(1,1464):
        if np.isnan(tower_height_fields[i,k]) == True:
            field_length.append(i-1)
            break
        elif i == 1463:
            field_length.append(1463)

# iterate over fields and determine annual optical efficiency
tower_heights = [15,30,45,60,75,90]
annual_eff_tower_heights = []
for i in range(6):
    tower_field = np.zeros((field_length[i]-1,4),dtype=float)
    tower_field[:,0:2] = tower_height_fields[1:field_length[i],i*2:(i*2 +2)]
    
    plt.figure()
    plt.plot(tower_field[:,0],tower_field[:,1],'o')
    
    eff, noon_power, yearly_sun_angles, LCOH_obj, no_helios,resource = plant_configuration_sim([0.1,0.5,0.5],tower_field,tower_heights[i],45)
    
    annual_eff_tower_heights.append(eff)
    
# receiver tilt  studies 
receiver_tilt_fields = np.genfromtxt('/home/tristan/Documents/Sunflower_parametric_studies/receiver_tilt_parametric/tilt_param_fields.csv',delimiter=',')

# determine number of heliostats in each field
field_length = []
for k in [1,3,5,7,9,11]:
    for i in range(1,1301):
        if np.isnan(receiver_tilt_fields[i,k]) == True:
            field_length.append(i-1)
            break
        elif i == 1300:
            field_length.append(1300)

# iterate over fields and determine annual optical efficiency
annual_eff_tilts = []
tilt=[90,75,60,45,30,15]
for i in range(6):
    tilt_field = np.zeros((field_length[i]-1,4),dtype=float)
    tilt_field[:,0:2] = receiver_tilt_fields[1:field_length[i],i*2:(i*2 +2)]
    
    plt.figure()
    plt.plot(tilt_field[:,0],tilt_field[:,1],'o')
    
    eff, noon_power, yearly_sun_angles, LCOH_obj, no_helios,resource = plant_configuration_sim([0.1,0.5,0.5],tilt_field,40,tilt[i])
    
    annual_eff_tilts.append(eff)

# receiver dsitance studies 
receiver_dist_fields = np.genfromtxt('/home/tristan/Documents/Sunflower_parametric_studies/receiver_in_field_parametric/tower_dist_param_fields.csv',delimiter=',')

# determine number of heliostats in each field
field_length = []
for k in [1,3,5,7,9,11,13]:
    for i in range(1,1300):
        if np.isnan(receiver_dist_fields[i,k]) == True:
            field_length.append(i-1)
            break
        elif i == 1299:
            field_length.append(1299)

# iterate over fields and determine annual optical efficiency
annual_eff_dist = []
for i in range(7):
    dist_field = np.zeros((field_length[i]-1,4),dtype=float)
    dist_field[:,0:2] = receiver_dist_fields[1:field_length[i],i*2:(i*2 +2)]
    
    plt.figure()
    plt.plot(dist_field[:,0],dist_field[:,1],'o')
    
    eff, noon_power, yearly_sun_angles, LCOH_obj, no_helios,resource = plant_configuration_sim([0.1,0.5,0.5],dist_field,40,45)
    
    annual_eff_dist.append(eff)  
    
#%% plot efficiency figures

# code to allo spines
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(True)


# mutliple x axes plot 

fig,ax = plt.subplots(figsize=(10,7))

par1 = ax.twiny()
par2 = ax.twiny()

#offset axis
par2.xaxis.set_ticks_position("bottom")
par2.xaxis.set_label_position("bottom")
par2.spines["bottom"].set_position(("axes",-0.125))

make_patch_spines_invisible(par2) # make spine visible


# plot graphs
p1, = ax.plot(tower_heights,annual_eff_tower_heights,'-o',color='navy',label='Tower height',linewidth=3,ms=10)
p2, = par1.plot(tilt,annual_eff_tilts,'-s',color='darkorange',label='Tilt angle',linewidth=3,ms=10)
p3, = par2.plot([0,5,10,15,20,25],annual_eff_dist,'-x',color='lime',label='Tower distance',linewidth=3,ms=10)

# set axes limits
ax.set_ylim([0.6,0.8])
ax.set_xlim([15,90])
par1.set_xlim([15,90])
par2.set_xlim([-5,25])

ax.set_xticks([15,30,45,60,75,90])
par1.set_xticks([15,30,45,60,75,90])
par2.set_xticks([0,5,10,15,20,25])
# par2.set_xticklabels([0,5,10,15,20,25])

# axes labels
ax.set_ylabel('Optical efficiency',fontsize = 15)
ax.set_xlabel('Tower height [m]',fontsize = 15)
par1.set_xlabel('Receiver tilt angle [$^\mathrm{o}}$]',fontsize = 15)
par2.set_xlabel('Tower distance from field [m]',fontsize = 15)

# ticks size
ax.tick_params(labelsize=15)
par1.tick_params(labelsize=15)
par2.tick_params(labelsize=15)
#legend
lines = [p1,p2,p3]
ax.legend(lines, [l.get_label() for l in lines],fontsize = 15)
plt.show()

    

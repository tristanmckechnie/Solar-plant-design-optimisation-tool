# -*- coding: utf-8 -*-
"""
Author: T McKecnie
Stellenbosch University
20 November 2019

Dispatch Optimization class
"""

import numpy as np
import matplotlib.pyplot as plt
from random import random 
from scipy.optimize import minimize 
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint 
import time
import dot as dot
import ctypes as ct
import pandas as pd


# function for plotting multiple spines
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

#%% defining class
class Dispatch:
    
    def __init__(self,start,days,receiver_data,tariff_data,time_horizon,process_output, TES_hours, E_start, no_helios,penality_paramter):
        
        # normalizing values for optimization problem
        self.thermal_normalization = no_helios*1.83*1.22*1000 # helios x area x design_dni
        self.tariff_normalization = max(tariff_data)            # max tariff value
        
        # Global variables used to define problem
        
        self.days = days                                                    # number of day to optimize in total
        self.start = start*24                                               # start position in hours
        self.end = self.start + days*24                                     # end position in hours
        self.rec_therm = receiver_data                                      # receiver thermal power data in hourly resolution
        self.tariff = tariff_data                                           # hourly tariff data
        self.design_n = time_horizon
        self.process_output = process_output                                # thermal output to process in MW
        self.output = process_output/self.thermal_normalization             # normalised thermal output to process 
        self.TES_hours = TES_hours                                          # hours of full load storage
        self.E_max = self.output*self.TES_hours                             # maximum capacity for storage
        self.E_start = E_start                                              # storage level at start
        self.penality_paramter = penality_paramter

        
        # Local variables used in a single optimization run, ie ones that returned by methods used in other methods
        
        # arrays to be filled during optimization routine
        self.thermal_storage = np.zeros(self.days*24)     # amount of energy in storage per hour
        self.electric_heat = np.zeros(self.days*24)       # amount of electric heat required per hour
        self.discharge = np.zeros(self.days*24)           # discharge energy per hour
        self.Dumped_heat = np.zeros(self.days*24)         # amount of heat dumped every hour
        self.E_starting = np.zeros(self.days)             # array to keep track of starting storage levels 
        self.starting_value = np.zeros(self.days*24)      # array to keep track of starting values ie x0's
        
        
# =============================================================================
#  SciPy optimization -  Optimization problem formulation and solver     
# =============================================================================
        
    def optimizer(self,Qrec,cost,E_max,design_n,output,x0,E_start):
    
        # objective function
    
        def objective(x):
            k = self.penality_paramter # penality value
            c = cost
            global Pelec
            Pelec = np.zeros(design_n)
            penality = np.zeros(design_n)
            for i in range(len(x)):
                Pelec[i] = output - x[i] # 0.8 MW / 1000*1608*1.83*1.22
            
            # for i in range(1,len(x)):
            #     penality[i] = ((x[i-1] - x[i])**2)#**0.5 #abs(x[i-1] - x[i])#
            
            
            return sum(Pelec*c) #+# k*sum(penality)
        
        # constraints
            
        def constraint2(x): # max discharging inequality constraint
            global E_TES
            global dump
            
            QRec = Qrec
            dump = np.zeros(design_n)
            E_TES = np.zeros(design_n) # this is house we keep the E_TES from one time horizon to the next !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            E_TES[0] =  E_start
            
            for i in range(1,design_n):
                E_TES[i] = QRec[i] + E_TES[i-1] - x[i-1]# TES charges with all receiver
                
                if E_TES[i] > E_max: # TES has max limit - should this be before are after discharge as that will affect max allowed ??????
                    dump[i] = E_TES[i] - E_max
                    E_TES[i] = E_max          
                    
       
            return  E_TES - x
    
        def constraint3(x): # inequality constraint, cant discharge more than Qout_process
            return  output - x
        
        # def constraint4(x): for cobyla
        #     return x
        
        # slsqp bounds
        bound = (0,output) # bounds, basically constraints 3, discharge must be postive and less than or equal to max storage capacity
        bnds = np.full((design_n,2),bound) # all 24 variables have same bounds
        
        # slsqp and cobyla contraints
        con2 = {'type': 'ineq', 'fun': constraint2}
        con3 = {'type': 'ineq', 'fun': constraint3}
        # con4 = {'type': 'ineq','fun':constraint4} # for cobyla
    
        cons = [con2,con3]

        # call optimizer
        sol = minimize(objective,x0,method='SLSQP',constraints=cons,bounds=bnds,tol=1e-4, options={'ftol':1e-4,'maxiter':300} ) #  #SLSQP, 



        return sol, E_TES, Pelec, dump   

# =============================================================================
#  Dot optimizer    
# =============================================================================
        
    def dot_optimizer(self,Qrec,cost,E_start):
        def myEvaluate(x, obj, g, param): # this is the objective function and constraints evaluation 
    
        #def optimizer(self,Qrec,cost,E_max,nDvar,output,x0,E_0):
            
        
        #       objective value calculation
        
            nDvar = self.design_n
           
            tariff_data = self.tariff
            QRec = self.rec_therm
            
            tariff_data = cost#tariff_data[0:nDvar]
            QRec = Qrec#QRec[0:nDvar]/(1608*1000*1.83*1.22)
            
            output = self.output
            output_array = np.full(nDvar,output)
            TES_hours = self.TES_hours
            E_max =output*TES_hours
            k = self.penality_paramter # penality value
            c = tariff_data/max(tariff_data) 
            
            Pelec = np.zeros(nDvar)
            penality = np.zeros(nDvar)
            for i in range(nDvar):
                Pelec[i] = output - x[i] # 0.8 MW / 1000*1608*1.83*1.22
            
            # for i in range(1,nDvar):
            #     penality[i] = (x[i-1] - x[i])**2
        
        
        #         constraints
        
            
            dump = np.zeros(nDvar)
            E_TES = np.zeros(nDvar) # this is house we keep the E_TES from one time horizon to the next !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            E_TES[0] =  E_start
            
            for i in range(1,nDvar):
                E_TES[i] = QRec[i] + E_TES[i-1] - x[i-1]# TES charges with all receiver
                
                if E_TES[i] > E_max: # TES has max limit - should this be before are after discharge as that will affect max allowed ??????
                    dump[i] = E_TES[i] - E_max
                    E_TES[i] = E_max          
                
        
            # Evaluate the objective function value and use the ".value" notation to
            # update this new value in the calling function, which is DOT in this
            # case
        
            obj.value = sum(Pelec*c) #+ k*sum(Pelec) 
            # Evaluate the constraints and update the constraint vector.  Since this 
            # is a numpy object, the values will be updated in the calling function
         
            g[0:nDvar] = x - E_TES  
                
        
            g[nDvar:nDvar*2] = x - output_array  
        
            return Pelec, dump, E_TES


        #------------------------------------------------------------------------------
        # The main code that setup the optimization problem and calls DOT to solve the
        # problem
        #------------------------------------------------------------------------------
        nDvar = self.design_n  # Number of design variables
        nCons = nDvar*2  # Number of constraints
        
        # Create numpy arrays for the initial values and lower and upper bounds of the
        # design variables
        x  = np.zeros(nDvar, float)
        xl = np.zeros(nDvar, float)
        xu = np.empty(nDvar, float)
        
        # Set the initial values and upper and lower bounds for the design variables
        for i in range(nDvar):
            x[i]  = 0#random()*1e6/(1608*1.83*1.22*1000)
            xl[i] = 0
            xu[i] = self.output
        
        # Initialize the DOT wrapper - this will load the shared library
        aDot = dot.dot(nDvar)
        
        # Set some of the DOT parameters
        aDot.nPrint   = 3
        aDot.nMethod  = 1
        
        # Set the function to call for evaluating the objective function and constraints
        aDot.evaluate = myEvaluate
        
        # initialize required arrays
        electric_heat = np.zeros(nDvar)
        Dumped_heat = np.zeros(nDvar)
        thermal_storage = np.zeros(nDvar)
        
        # Call DOT to perform the optimization
        start_clock = time.process_time()
        dotRet, electric_heat, Dumped_heat, thermal_storage = aDot.dotcall(x, xl, xu, nCons)
        nDvar_clock = time.process_time()
        # Print the DOT return values, this will be the final Objective function value,
        # the worst constaint value at the optimum and the optimum design variable 
        # values.  This is returned as a numpy array.
        # print('######################################################################')
        # print('Execution time: ', nDvar_clock - start_clock)
        # print('######################################################################')
        # print( '\nFinal, optimum results from DOT:' )
        # print( dotRet )
        
        return dotRet, electric_heat, Dumped_heat, thermal_storage

# =============================================================================
#     Rolling time-horizon dispatch optimization
# =============================================================================
        
    def rolling_optimization(self,optimizer_choice,initial_guess_type,initial_guess): # ,initial_guess this must be set to 'dot' or 'scipy' 
        E_start = self.E_start # initial storage value from which to start optimizer
        count = 0
        
        # report optimizer used
        
        if optimizer_choice == 'dot': 
            print('Optimizing dispatch profile using DOT')
        elif optimizer_choice == 'scipy':
            print('Optmizing dispatch profile using SciPy')
        else: 
            print('A valid optimizer was not selected!')
            
        # report initial guesses used
            
        if initial_guess_type == 'provided_guesses':
            print('Using provided initial guesses')
                
        elif initial_guess_type == 'random':
            print('Using provided random guesses')
            
        elif initial_guess_type == 'heuristic':
            print('Using provided heuristic guesses')
            
        elif initial_guess_type == 'zeros':
            print('Using provided zeros guesses')
        
        elif initial_guess_type == 'sinusoidal':
            print('Using provided sinusoidal guesses')
        
        else:
            print('Invalid starting guess choice!\nRolling optimization ended!')

        
        # Rolling time-horizon implemenation. Each iteration represents a single time-horizon optimization
        
        for k in range(self.start,self.end,24): # loop through days, each time optimizing one day by optimizing a single time horizon, then stepping to next days
            # print('start',k,'end',k+self.design_n)            
            self.E_starting[count] = E_start # local E_start
            
            # optmization known value inputs
            
            Qrec = self.rec_therm[k:(k+self.design_n)]/self.thermal_normalization
            cost = self.tariff[k:(k+self.design_n)]/self.tariff_normalization
        
            # initial guesses

            

            if initial_guess_type == 'provided_guesses':
                x0 = np.empty(self.design_n)
                if k < (self.days*24-self.design_n - 1):                                     #!!!!!!!!!!!!!!!!!!!!!!! change 48 to 100 for time-horizon paramtric study
                    x0 = initial_guess[k:k+self.design_n] # initial guesses for SLSQP from MMFD
                    # self.starting_value[24*count:24*(count+1)] = x0[0:24]
                else:
                    x0 = np.zeros(self.design_n)
                    # self.starting_value[24*count:24*(count+1)] = x0[0:24]
                    
            elif initial_guess_type == 'random':
                x0 = np.empty(self.design_n)
                for i in range(1,self.design_n,1):
                    x0[i] = random()*self.output
                
            elif initial_guess_type == 'heuristic':
                x0 = np.empty(self.design_n)
                Qout2, E_TES2 = self.heuristic_dispatch(self.rec_therm[self.start:self.end+self.design_n]/1e6,self.output*self.thermal_normalization/1e6,self.E_max*self.thermal_normalization/1e6) 
                x0 = Qout2[k:k+self.design_n]*1e6/self.thermal_normalization    
                
            elif initial_guess_type == 'zeros':
                x0 = np.zeros(self.design_n)
            
            elif initial_guess_type == 'sinusoidal':
                sine_x = np.linspace(0,np.pi,24)         
                x0 = np.zeros(self.design_n)
                for i in range(1,self.design_n,1):
                    x0[0:24] = self.output*np.sin(sine_x)
                    x0[24:48] = self.output*np.sin(sine_x)
                    x0[48:72] = self.output*np.sin(sine_x)
            
            else:
                print('Invalid starting guess choice!\nRolling optimization ended!')
                k = 8760 # set counter to end value to stop loop
                
            self.starting_value[24*count:24*(count+1)] = x0[0:24]   
              
                
            # check initial feasbility
        
            # ret1, ret2, constraint1,constraint2 = self.initial_feasibility(Qrec,x0,E_start,self.E_max,self.output)
#            print('\n******************************','\n Iteration:',count,'\n******************************')
#            print(E_start)
#            print(ret1)
#            print(ret2)
            
            # Optimize single time-horizon
            
            if optimizer_choice == 'scipy':
                #dispatch optmization for one time horizon - scipy
                sol, E_TES, Pelec, dump = self.optimizer(Qrec,cost,self.E_max,self.design_n,self.output,x0,E_start)
            elif optimizer_choice == 'dot':
                # dispatch optimization for one time horizon - DOT
                sol, Pelec, dump, E_TES = self.dot_optimizer(Qrec,cost,E_start)
            else:
                print('A valid optimizer choice was not selected!')
                
            # fill annaul arrays with time-horizon data
                
            self.thermal_storage[24*count:24*(count+1)] = E_TES[0:24]
            self.electric_heat[24*count:24*(count+1)] = Pelec[0:24]
            
            if optimizer_choice == 'scipy':
                self.discharge[24*count:24*(count+1)] = sol.x[0:24]
            elif optimizer_choice == 'dot':
                self.discharge[24*count:24*(count+1)] =  sol[2:26]
            
            self.Dumped_heat[24*count:24*(count+1)] = dump[0:24]
        
        
            # self.starting_value[24*count:24*(count+1)] = x0[0:24]
            
            # boundary condition between time_horizons
        
            E_start = E_TES[24] # this is how we send on the same storage from one time horizon to the next. !!!!!!! this really should be 23 !!!!!!!!!!!!!!!!
            
            # removes any numerical rounding error
            if E_start < 0:
                E_start = 0
                

            count = count + 1    
        
# =============================================================================
#    Checking feasibility of initial values / starting guesses of optimizer
# =============================================================================
        
    def initial_feasibility(self,Qrec,x0,E_start,E_max,output):
        E_TES = np.empty(len(Qrec))
        E_TES[0] =  E_start

        for i in range(1,len(Qrec)):
            E_TES[i] = Qrec[i] + E_TES[i-1] - x0[i-1]# TES charges with all receiver
            
            if E_TES[i] > E_max: # TES has max limit - should this be before are after discharge as that will affect max allowed ??????
                E_TES[i] = E_max  
    
        constraint1 = E_TES - x0
        constraint2 = output - x0
        ret1 = 'constraint 1 is fine'
        ret2 = 'constraint 2 is fine'
        
        for k in range(len(constraint1)):
            if constraint1[k]  < 0:
                ret1 = 'constraint 1 is violated'
        for k in range(len(constraint2)):   
            if constraint2[k] < 0:
                ret2 = 'constraint 2 is violated'
            
        return ret1, ret2, constraint1,constraint2    

# =============================================================================
#  Heuristic dispatch strategy
# =============================================================================
        
    def heuristic_dispatch(self,Qrec,demand,E_max):
        
        E_TES = np.zeros(len(Qrec)) #initialize TES array
        Qout = np.zeros(len(Qrec)) # initialize output array
    
        for i in range(1,len(Qrec)): # loop through every timestep, starting at second time step (for E_TES[i-1])
            
            if Qrec[i] >= demand: # if receiver alone has sufficient energy for process
                Qout[i] = demand
                E_TES[i] = Qrec[i] - demand + E_TES[i-1] # charge storage    
                if E_TES[i] > E_max: # cap max storage
                    E_TES[i] = E_max
            elif Qrec[i] + E_TES[i-1] >= demand: # together or TES alone have enough
                Qout[i] = demand
                E_TES[i] = E_TES[i-1] + Qrec[i] - demand
            else: # receiver and TES dont have enough so just charge storage
                E_TES[i] = E_TES[i-1] + Qrec[i]
                Qout[i] = 0
                if E_TES[i] > E_max:
                    E_TES[i] = E_max
                    
        return Qout, E_TES   


# =============================================================================
#  Calculating costs
# =============================================================================
    def strategy_costs(self):
        
        Qout2, E_TES2 = self.heuristic_dispatch(self.rec_therm[self.start:self.end]/1e6,self.output*self.thermal_normalization/1e6,self.E_max*self.thermal_normalization/1e6) # !!!!!! note output set to 1 MW
        
        # cost of electrical 
        count = 0
        elec_cost = np.zeros(self.days*24)
        for i in range(self.start,self.end,1):
            elec_cost[count] = self.electric_heat[count]*self.tariff[i]
            count = count + 1
            
        #calc Qelec required for heuristic strategy
        Qelec = np.zeros(len(Qout2)) 
        for i in range(len(Qout2)):
            if Qout2[i] < self.output*self.thermal_normalization/1e6:
                Qelec[i] = self.output*self.thermal_normalization/1e6
        
        opt_cost = np.zeros(self.days*24)
        heur_cost = np.zeros(self.days*24)
        xxx = self.electric_heat*self.thermal_normalization/1e3
        temp1 = (self.electric_heat*self.thermal_normalization/1e3)*(self.tariff[self.start:self.end]/100) # first bracket to kW, second bracket to rands from cents
        temp2 = Qelec*1e3*(self.tariff[self.start:self.end]/100) #  first bracket to kw, second bracket to rands from cents
        for i in range(self.days*24):
            if i > 0:
                opt_cost[i] = sum(temp1[0:i])
                heur_cost[i] = sum(temp2[0:i])  
        #       
        print("Heuristic strategy total electrical heat cost: ", heur_cost[self.days*24 -1],"\nOptimal strategy total electrical heat cost: ", opt_cost[self.days*24 -1])
        
        return opt_cost, heur_cost, opt_cost[self.days*24 -1], heur_cost[self.days*24 -1]
# =============================================================================
#  Plotting
# =============================================================================
        
    def plotting(self):    

        # get costs for different strategies
        
        x,y,opt_cost, heur_cost = self.strategy_costs()
        
        # cummulative energy graphs
        cum_sum1 = np.zeros(self.days*24)
        cum_sum2 = np.zeros(self.days*24) 
        cum_sum3 = np.zeros(self.days*24)
        cum_sum_cost = np.zeros(self.days*24)
        for i in range(self.days*24):
            cum_sum1[0] = self.rec_therm[self.start]/1e6
            cum_sum2[0] = self.discharge[0]*self.thermal_normalization/1e6
            cum_sum3[0] = self.Dumped_heat[0]*self.thermal_normalization/1e6
            if i > 0:
                cum_sum1[i] = sum(self.rec_therm[self.start:i+self.start]/1e6)
                cum_sum2[i] = sum(self.discharge[0:i]*self.thermal_normalization/1e6)
                cum_sum3[i] = sum(self.Dumped_heat[0:i]*self.thermal_normalization/1e6)
        
        # Plots
        
        fig, axes = plt.subplots(3,2, figsize=(8,8),sharex=True)
        
        axes[0,0].set_ylabel('Normalized TES Discharge',color='k')
        axes[0,0].step(np.arange(self.start,self.end,1),(self.discharge/self.output),'g-',label='TES discharge',markersize='5')
        axes[0,0].set_xlabel('Hours of the year')
        ax2 = axes[0,0].twinx()
        ax2.set_ylabel('Normalized Electrical Tariff ',color='k')
        ax2.plot(np.arange(self.start,self.end,1),self.tariff[self.start:self.end]/max(self.tariff),'ko--',markersize='5',label='Tariff')
        ax2.grid(True) 
        h1, l1 = axes[0,0].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(h1+h2, l1+l2,loc=0)
        axes[0,0].set_ylim([0,1.5])
        ax2.set_ylim([0,1.5])
        
        
        axes[0,1].set_ylabel('Receiver [MW]',color='b')
        axes[0,1].plot(np.arange(self.start,self.end,1),self.rec_therm[self.start:self.end]/1e6,'b-.',markersize='5')
        axes[0,1].plot(np.arange(self.start,self.end,1),(self.discharge+self.electric_heat)*self.thermal_normalization/1e6,'r',markersize='5',label='Heat + Electric output')
        axes[0,1].grid(True)
        axes[0,1].grid(True) 
        
        Cummulative_TES_discharge = cum_sum2[-1]
        Cummulative_Receiver_thermal_energy = cum_sum1[-1]
        Cummulative_dumped_heat = cum_sum3[-1] 
        
        bar_width = 0.5
        
        
        # p1 = axes[1,0].bar(0,Cummulative_TES_discharge,width=bar_width,color='g')
        # p2 = axes[1,0].bar(0,Cummulative_dumped_heat,bottom=Cummulative_TES_discharge,width=bar_width,color='b')
        # p3 = axes[1,0].bar(.66,Cummulative_Receiver_thermal_energy,width=bar_width,color='r')
        
        # axes[1,0].legend((p1[0],p2[0],p3[0]),('TES discharge','Dumped heat','Receiver thermal energy'),loc=1)
        # axes[1,0].set_ylabel('Annual cummulative energy [MWh]')

        # axes[1,0].set_ylim([0,1.35*cum_sum1[-1]])
        
        axes[1,0].plot(np.arange(self.start,self.end,1),cum_sum2,'bo-',label='Cummulative TES discharge',linewidth=2)
        axes[1,0].plot(np.arange(self.start,self.end,1),cum_sum1,'ko-',label='Cummulative Receiver thermal energy',linewidth=2)
        axes[1,0].plot(np.arange(self.start,self.end,1),cum_sum3,'go-',label='Cummulative dumped heat',linewidth=2) # heat that couldn't fit into storage
        axes[1,0].plot(np.arange(self.start,self.end,1),cum_sum3 + cum_sum2,'go--',label='Cummulative dumped + TES discharge')
        axes[1,0].set_ylabel('Cummulative heat [MWh]')
        axes[1,0].set_xlabel('Hours')
        # ax3 = axes[1,0].twinx()
        # ax3.plot(np.arange(self.start,self.end,1),opt_cost,'r',label='Optimised cost')
        # ax3.plot(np.arange(self.start,self.end,1),heur_cost,'m',label='Heuristic cost')
        # ax3.legend()
        # ax3.set_ylabel('Cummulative cost')
        axes[1,0].legend()
        axes[1,0].grid(True)
        axes[1,0].set_xlabel('Hours of the year')
        
        
        axes[1,1].set_xlabel('time [hr]')
        axes[1,1].set_ylabel('Receiver thermal energy  [MW]',color='b')
        axes[1,1].plot(np.arange(self.start,self.end,1),self.rec_therm[self.start:self.end]/1e6,'b-',label='Qrec')
        
        ax5 = axes[1,1].twinx()
        ax5.set_ylabel('Discharge  [MW]',color='g')
        ax5.plot(np.arange(self.start,self.end,1),self.discharge*self.thermal_normalization/1e6,'g-',label='Qrec')
        ax5.grid(True)
        
        ax6 = axes[1,1].twinx()
        ax6.spines["right"].set_position(("axes",1.085))
        make_patch_spines_invisible(ax5)
        # Second, show the right spine.
        ax6.spines["right"].set_visible(True)
        ax6.set_ylabel('Thermal energy storage [hrs]',color='r')
        ax6.step(np.arange(self.start,self.end,1),self.thermal_storage*self.thermal_normalization/self.process_output,'r-.',label='cost')
        
        ax7 = axes[1,1].twinx()
        ax7.spines["right"].set_position(("axes",1.17))
        # Second, show the right spine.
        ax7.spines["right"].set_visible(True)
        ax7.set_ylabel('Dump heat [MW]',color='k')
        ax7.plot(np.arange(self.start,self.end,1),self.Dumped_heat*self.thermal_normalization/1e6,'k-o',markersize='3')
            
        axes[1,1].set_ylim([0,max(self.rec_therm[self.start:self.end])*1.5/1e6])
        ax5.set_ylim([0,2])
        ax6.set_ylim([0,self.TES_hours*1.2])
        ax7.set_ylim([0,max(self.Dumped_heat*self.thermal_normalization/1e6)*1.5])
        
        string = "Start day: " + str(self.start) + " , End day: " + str(self.end) + ", Time horizon: " + str(self.design_n) +" hours"+ "\nTES: " + str(self.TES_hours) + " hours" + " ,Qout: " + str(self.output*self.thermal_normalization/1e6) + "MW"
        plt.suptitle(string)
        plt.show()
        
        count3=0
        for i in range(0,self.days*24,24):
            axes[2,0].step(np.arange(count3,count3+24,1),self.thermal_storage[count3:count3+24],'-x')
            count3 = count3+24
        axes[2,0].plot(np.arange(0,self.days*24,24),self.E_starting,'ko',markersize=5)
        axes[2,0].grid(True)
        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        
        fig, ax = plt.subplots(2,1,figsize=(10,7),sharex=True)
        
        string = "Start day: " + str(self.start) + " , End day: " + str(self.end) + ", Time horizon: " + str(self.design_n) +" hours"+ "\nTES: " + str(self.TES_hours) + " hours" + " ,Qout: " + str(self.output*self.thermal_normalization/1e6) + "MW" + "Penalty value: " + str(self.penality_paramter)
        fig.suptitle(string)
        
        start_time = 0
        end_time = 360*24

        ax[0].set_ylabel(' Thermal Energy  [MW]',color='k')
        ax[0].plot(np.arange(start_time,end_time,1),self.rec_therm[start_time:end_time]/1e6,'b-',label='$Q_{rec}$')#self.start:self.end
        ax[0].plot(np.arange(start_time,end_time,1),self.discharge[start_time:end_time]*self.thermal_normalization/1e6,'g-',label='$\dot{x}$')
        ax[0].plot(np.arange(start_time,end_time,1),self.Dumped_heat[start_time:end_time]*self.thermal_normalization/1e6,'k-o',markersize='3',label = 'Dumped')
        ax[0].set_ylim([0,2.7])
        
        ax1 = ax[0].twinx()
        ax1.set_ylim([0,self.TES_hours*1.2])
        ax1.set_ylabel('TES hours [hr]',color='k')
        ax1.step(np.arange(start_time,end_time,1),self.thermal_storage[start_time:end_time]*self.thermal_normalization/self.process_output,'r-.',label='$E_{tes}$')
        ax[0].plot(np.nan, 'r-.',label = '$E_{tes}$' )
        
        ax[1].plot(np.arange(start_time,end_time,1),self.discharge[start_time:end_time]*self.thermal_normalization/1e6,'g-',label='$\dot{x}^*$')
        ax[1].plot(np.arange(start_time,end_time,1),self.starting_value[start_time:end_time]*self.thermal_normalization/1e6,'x',label='$\dot{x}_0$')
        ax[1].set_ylabel('Thermal energy [MW]' )
        ax2 = ax[1].twinx()
        ax2.plot(np.arange(start_time,end_time,1),self.tariff[start_time:end_time]/max(self.tariff),'ko--',markersize='4',label='Tariff')
        ax2.set_ylabel('Normalized Tariff')
        ax[1].set_xlabel('Hour of the year [hr]')
        ax[1].plot(np.nan,'ko--',label='Tariff')
        
        ax[1].set_ylim([0,1.1])
        ax2.set_ylim([0,1.3])
        ax[0].grid(True)        
        ax[1].grid(True)

        
        ax[0].legend(ncol=4,loc=1)
        ax[1].legend(ncol=3,loc=1)

        #####################################################################################################
        #####################################################################################################
        #####################################################################################################
        return opt_cost, heur_cost, Cummulative_TES_discharge,Cummulative_Receiver_thermal_energy,Cummulative_dumped_heat

#%% Receiver data
dni = np.genfromtxt('SUNREC_hour.csv')
eta =    np.load('optimized_field_eta.npy') #np.load('matti_eta.npy') #
receiver_data = dni*eta*1527*1.83*1.22

annual_eta = sum(receiver_data)/sum(1527*dni*1.22*1.83)
#%% Single plant configuration dispatch optimization and economics

start = 0
days = 360
# receiver_data = np.genfromtxt('kalagadi_field_output_new_efficiency.csv',delimiter=',') # note this is the new receiver efficiency applied in the excel sheet. 
tariff_data = np.genfromtxt('kalagadi_extended_tariff.csv',delimiter=',')#tariff_data = np.load('./data/megaflex_tariff.npy') #
time_horizon = 48
process_output = 0.85e6
TES_hours = 14
E_start = 0
no_helios = 1527
penality_val = 0

# create object instance and time simulation
bloob_mmfd = Dispatch(start,days,receiver_data,tariff_data,time_horizon,process_output, TES_hours, E_start, no_helios,penality_val)  # initialize a 'Dispatch' object 
bloob_slsqp = Dispatch(start,days,receiver_data,tariff_data,time_horizon,process_output, TES_hours, E_start, no_helios,penality_val)

# run rolling time-horizon optimization object method
start_clock = time.process_time()
bloob_mmfd.rolling_optimization('dot','random',0) # run mmfd with random starting guesses
bloob_slsqp.rolling_optimization('scipy','zeros',0) # run slsqp with mmfd starting guesses , bloob_mmfd.discharge


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
# print('########################################################################')

annual_heat_gen = sum(bloob_slsqp.discharge*bloob_slsqp.thermal_normalization)
# =============================================================================
#     LCOH calculation
# =============================================================================

n = 25;

#% CAPEX values

CAPEX_tower = 8288 + 1.73*40**2.75;
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

LCOH_e = (heuristic_cost_temp/14.5)/annual_elec_gen # optimal_cost_temp !!!! Remember that costs for optimal cost etc is in Rands so must convert to dollar!!!!

# LCOH combined

LCOH = ((LCOH_e*annual_elec_gen) + (LCOH_s*annual_heat_gen/1e6) )/ (365*24*process_output/1e6)
print('Solar LCOH:', LCOH_s)
print('Electric LCOH', LCOH_e)
print('Combined LCOH:', LCOH)

# check sum of gradients
# gradients = np.zeros(8759) 
# for i in np.arange(0,8758):
#     gradients[i] = abs(bloob_slsqp.discharge[i] - bloob_slsqp.discharge[i+1])
# sum_of_gradients = sum(gradients)
# print('Sum of gradients:',sum_of_gradients)
print('########################################################################')


#%%

LCOH_e_total = (sum((tariff_data[0:8759]/100)*0.85*1e3)/(0.85*365*24))/14.5
A = sum(bloob.electric_heat*bloob.thermal_normalization)/1e6
AA = sum(bloob.discharge*bloob.thermal_normalization)/1e6

#%% Time-horizon parametric study
        
start = 0
days = 365
receiver_data = np.genfromtxt('kalagadi_field_output_new_efficiency.csv',delimiter=',') # note this is the new receiver efficiency applied in the excel sheet. 
tariff_data = np.genfromtxt('kalagadi_extended_tariff.csv',delimiter=',')#tariff_data = np.load('./data/megaflex_tariff.npy') #
process_output = 0.85e6
TES_hours = 14
E_start = 0
no_helios = 1608
penality_val = 0

time_horizon = np.arange(25,242,24)
optimal_cost = np.zeros(10)
heuristic_cost = np.zeros(10)
sum_of_gradients_param = np.zeros(10)
results_array = np.zeros((10,4))

for i in range(len(time_horizon)):
        
    # create object instance and time simulation
    bloob_mmfd = Dispatch(start,days,receiver_data,tariff_data,time_horizon[i],process_output, TES_hours, E_start, no_helios,penality_val) # initialize a 'Dispatch' object 
    bloob_slsqp = Dispatch(start,days,receiver_data,tariff_data,time_horizon[i],process_output, TES_hours, E_start, no_helios,penality_val)
    
    # run rolling time-horizon optimization object method
    start_clock = time.process_time()
    bloob_mmfd.rolling_optimization('dot','zeros',0) # run mmfd with random starting guesses
    bloob_slsqp.rolling_optimization('scipy','provided_guesses',bloob_mmfd.discharge) # run slsqp with mmfd starting guesses
    end_clock = time.process_time()
    
    # run cost method
    optimal_cost_temp_array, heuristic_cost_temp_array ,optimal_cost_temp, heuristic_cost_temp = bloob_slsqp.strategy_costs()
    
    optimal_cost[i] = optimal_cost_temp
    heuristic_cost[i] = heuristic_cost_temp
    
    gradients = np.zeros(8759) 
    for k in np.arange(0,8758):
        gradients[k] = abs(bloob_slsqp.discharge[k] - bloob_slsqp.discharge[k+1])
    sum_of_gradients = sum(gradients)
    
    sum_of_gradients_param[i] = sum_of_gradients
    # add results into an array
    
    results_array[i,0] = time_horizon[i]# penalty parameter
    results_array[i,1] = end_clock-start_clock# computational time
    results_array[i,2] = optimal_cost_temp# optimal cost 
    results_array[i,3] = sum_of_gradients# sum of gradients
    #%%
plt.figure()
plt.plot(time_horizon, optimal_cost, 'ms-',label='Optimal cost')
#plt.plot(penality_val, heuristic_cost, 'rs-',label='Heuristic cost')
plt.xlabel('Time_horizon value')
plt.ylabel('Cost of electrical heat [Rands]')
plt.ylim([0,max(optimal_cost)*1.15])
plt.grid(True)

sub_axes = plt.axes([.4, .4, .25, .25]) 

# plot the zoomed portion
sub_axes.plot(time_horizon, optimal_cost, 'ms-',label='Optimal cost')
sub_axes.grid(True)
#sub_axes.set_ylim([0])
plt.show()

# saving results to dataframe

col_headings = ['penalty value','computational time','optimal cost','sum of gradients']
index_array = np.arange(1,11,1)
results_df = pd.DataFrame(data=results_array,index=index_array,columns=col_headings)

results_df.to_csv('../Results_rolling_time_horizon_parametric/SLSQP_and_mmfd.csv')
#%% Plotting comparison of rolling vs single time-horizon

x = np.load('3000_single.npy')*bloob.thermal_normalization/1e6
y = np.load('3000_time_horizon.npy')*bloob.thermal_normalization/1e6


plt.figure()
plt.plot(range(len(x)),x,'b-',label = 'single')
plt.plot(range(len(y)),y,'r-',label = 'time horizon')
plt.legend()
plt.grid(True)
plt.show()    

#%% parametric study to determine optical solar multiple and amount of thermal energy storage
parametric_costs = np.empty((7,10))
parametric_LCOH_S = np.empty((7,10))
parametric_LCOH_E = np.empty((7,10))
SM_output = np.linspace(0.5,1.5,10)
TES_trys = np.arange(8,22,2)
for k in range(10): # vary SM
    for i in range(7): # vary TES hours
    
        start = 0
        days = 365
        receiver_data = np.genfromtxt('kalagadi_field_output_new_efficiency.csv',delimiter=',') # note this is the new receiver efficiency applied in the excel sheet. 
        tariff_data = np.genfromtxt('kalagadi_extended_tariff.csv',delimiter=',')
        time_horizon = 48
        process_output = SM_output[k]*1e6
        TES_hours = TES_trys[i]
        E_start = 0
        no_helios = 1608
        penality_val = 0.025
        
        # create object instance and time simulation
        bloob = Dispatch(start,days,receiver_data,tariff_data,time_horizon,process_output, TES_hours, E_start, no_helios,penality_val)  # initialize a 'Dispatch' object 
        
        # run rolling time-horizon optimization object method
        start_clock = time.process_time()
        solution = bloob.rolling_optimization()
        end_clock = time.process_time()
        
        # run plotting
        optimal_cost_temp_array, heuristic_cost_temp_array ,optimal_cost_temp, heuristic_cost_temp = bloob.plotting()
        
        print('########################################################################')
        print('Rolling time-horizon')
        #print(solution)
        print('Computational expense: ', end_clock - start_clock)
        print('Heuristic cost: ', heuristic_cost_temp)
        print('Optimal cost: ',optimal_cost_temp)
        print('########################################################################')   
              
        annual_heat_gen = sum(bloob.discharge*bloob.thermal_normalization)
        # =============================================================================
        #     LCOH calculation
        # =============================================================================
        
        n = 25;
        
        #% CAPEX values
        
        CAPEX_tower = 8288 + 1.73*40**2.75;
        CAPEX_vert_transport = 140892;
        CAPEX_horz_transport = 248634;
        
        CAPEX_TES = 20443*TES_hours*process_output/1e6; #% where TES is represented in MWh_t's
        
        CAPEX_receiver = 138130;
        
        CAPEX_heliostat = 112.5*3563 #% where Asf is the aperature area of the solar field
        
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
        
        annual_elec_gen = 365*24*process_output/1e6 - annual_heat_gen/1e6
        
        LCOH_e = (optimal_cost_temp/14.5)/annual_elec_gen # !!!! Remember that costs for optimal cost etc is in Rands so must convert to dollar!!!!
        
        # LCOH combined
        
        LCOH = ((LCOH_e*annual_elec_gen) + (LCOH_s*annual_heat_gen/1e6) )/ (365*24*process_output/1e6)
              
        parametric_costs[i,k] = LCOH
        parametric_LCOH_S[i,k] = LCOH_s
        parametric_LCOH_E[i,k] = LCOH_e

#%%         
parametric_costs_sm = np.empty((7,10))
for i in range(10):
    parametric_costs_sm[:,i] = parametric_costs[:,9 - i]

plt.figure()       
count = 0;     
for c in np.arange(8,22,2):            
    string = 'TES ' + str(c) + 'hours'
    plt.plot(2.5/np.linspace(1.5,0.5,10),parametric_costs_sm[count,:],label=string,marker='o')
    plt.grid(True)
    plt.xlabel('SM',fontsize=14)
    plt.ylabel(r'$\mathrm{LCOH_{comb}~[\$/MWt]}$',fontsize=14)    
    count = count + 1
plt.legend(fontsize=12)    

# plt.plot(3.33333333333333333, 36.0, '^', color='red', ms=5)
# plt.text(3.33333333333333333, 35.5,"(3.33,36.31)")

plt.show()
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
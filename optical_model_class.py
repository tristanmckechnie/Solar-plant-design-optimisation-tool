#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:38:25 2020

@author: tristan

optical model class:
- takes a field position.csv file
- json's defining tower, receiver and heliostats
- location of plant

this contains the class for annual optical simulation of a plant. Similar to optical_model script but a .py format so it can be used with import functionality.
"""
# Required packages

import subprocess as sp # used to run Sunflower compiled executable
import numpy as np
import json # used to read and write Sunflower JSON files
import matplotlib.pyplot as plt
from sun_pos import *
import time as time
from scipy.interpolate import LinearNDInterpolator # used to generated interpolation function
from mpl_toolkits import mplot3d # used to plots interpolation map in 3D

#%%
class optical_model:
    ###########################################################################
    # class attributes: these are initialised by users and only contain what is parametrised for configuration runs. ie number of ray etc arent changed here.
    ###########################################################################
    
    def __init__(self,latitude,longitude,hempishere,time_zone,facet_width,facet_height,focal_lengths, tower_optical_height, receiver_tilt_angle, receiver_descretization,raytracer_grid,num_helios, executable_directory, json_input_directory):
        
        # location data
        self.latitude = latitude
        self.longitude = longitude
        self.hempishere = hempishere # 'north' or 'south'
        self.time_zone = time_zone
        
        # heliostat data
        self.facet_width = facet_width
        self.facet_height = facet_height
        self.focal_lengths = focal_lengths # can be a single value or a list [ , , ]
        
        # tower data
        self.tower_optical_height = tower_optical_height # height to centre of receiver
        
        # receiver data
        self.receiver_tilt_angle = receiver_tilt_angle
        self.receiver_descretization = receiver_descretization
        
        # file locations
        self.executable_directory = executable_directory # where the sunflower ray tracer executable lives 
        self.json_input_directory = json_input_directory # where the json input files for the ray tracer live
        
        # raytracer data
        self.raytracer_grid = raytracer_grid
        
        # resource data used for annual simulation
        self.resource = []
        
        # number of heliostats in field
        self.num_helios = num_helios

    ###########################################################################
    # read and write data to heliostat input json
    ###########################################################################
        
    def raytracer_inputs(self):
        
        with open(self.json_input_directory + '/raytracer.json') as raytracer: #open json as 'file'
            file = json.load(raytracer)
            
        file['raytracer']['num_rays_per_facet_width'] = self.raytracer_grid
        file['raytracer']['num_rays_per_facet_height'] = self.raytracer_grid
        
        with open(self.json_input_directory + '/raytracer.json','w') as new_raytracer: #save as json file (serialization)
            json.dump(file,new_raytracer,indent = 4)
    
    ###########################################################################
    # read and write data to heliostat input json
    ###########################################################################
        
    def heliostat_inputs(self):
        
        with open(self.json_input_directory + '/heliostat.json') as heliostat: #open json as 'file'
            file = json.load(heliostat)
            
        file['heliostat']['focal_length'] = self.focal_lengths
        file['heliostat']['facets'][0]['facet_dimensions'] = [self.facet_width,self.facet_height]
        file['heliostat']['facets'][0]['facet_position'][0] = file['heliostat']['facets'][0]['facet_dimensions'][0] / -2 # needed for different facet size
        file['heliostat']['facets'][0]['facet_position'][1] = file['heliostat']['facets'][0]['facet_dimensions'][1] / -2 
        
        with open(self.json_input_directory + '/heliostat.json','w') as raytracer: #save as json file (serialization)
            json.dump(file,raytracer,indent = 4)
            
    ###########################################################################
    # read and write data to receiver input json
    ###########################################################################
            
    def receiver_inputs(self):
        
        with open(self.json_input_directory + '/receiver.json') as receiver: #open json as 'file'
            file = json.load(receiver)
            
        file['receiver']['flat_receiver']['tilt_angle'] = self.receiver_tilt_angle
        file['receiver']['dist_to_towertop'] = 0.25#0.4358#1 # ie 1 m higher than receiver edge
        file['receiver']['flat_receiver']['num_horizontal_pieces'] = self.receiver_descretization
        file['receiver']['num_vertical_pieces'] = self.receiver_descretization

        
        with open(self.json_input_directory + '/receiver.json','w') as new_receiver: #save as json file (serialization)
            json.dump(file,new_receiver,indent = 4)    
            
    ###########################################################################
    # read and write data to receiver input json
    ###########################################################################
            
    def tower_inputs(self):
        
        with open(self.json_input_directory + '/tower.json') as tower: #open json as 'file'
            file = json.load(tower)
            
        file['tower']['height'] = self.tower_optical_height #+ 1.1284/2 + 1 # ie 1 m above receiver  
        
        with open(self.json_input_directory + '/tower.json','w') as new_tower: #save as json file (serialization)
            json.dump(file,new_tower,indent = 4)   
    
    ###########################################################################
    # run ray tracer moment simulation
    ###########################################################################
            
    def run_ray_tracer(self):
        
        # Sunflower executable call, with settings
        args_settings = "--settings=" + self.json_input_directory
        args_weather = "--weather=" + self.json_input_directory + "/capetown.epw"
        p1 = sp.run([self.executable_directory, args_settings, args_weather])
        
        # retrieve moment simulation results flux map on rectangular plane
        flux_map = np.genfromtxt(self.json_input_directory+'/fluxmap.csv',delimiter=',') #
        flux_map = np.delete(flux_map,self.receiver_descretization,1) # remove arb column of nan's
        
        # call square to solar correction function
        flux_map_circle = self.square_to_circle(flux_map) # !!! Remember to choose correct number of elements !!!
        
        moment_power = sum(sum(flux_map_circle))
        
        # moment_power = sum(sum(flux_map))
        
        # print('Moment power: ',moment_power,' W')
        
        return moment_power
    
    ###########################################################################    
    # square to circle correction method
    ###########################################################################
        
    def square_to_circle(self,flux_map): # note user must give number of discretized squares on receiver
        # receiver area
        area = 1 # m^2
        length = (area/np.pi)**0.5
        
        # grid
        n = self.receiver_descretization # number of points on the grid
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
        invalid = invalid.reshape(self.receiver_descretization,self.receiver_descretization)
        
        flux_circle = invalid*flux_map 
        
        return flux_circle

    ###########################################################################    
    # design day angles for interpolation map
    ###########################################################################
        
    def design_day_sun_angles(self):
        
        c = 24 # number of evaluations per day

        azimuth = np.zeros(c*3) # vectors for angles
        altitude = np.zeros(c*3)
        azimuth_1 = np.zeros(6) # added space for solar noon
        altitude_1 = np.zeros(6)
        DNI_design_days = np.zeros(c*3 +6) # for 78 interp sim
        
        # days and local times to simulate
        days = [79, 172, 356] # days to be evaluated for generating optical efficiency map
        hours = np.linspace(1,24,c)
        
        # solar angles for solar noon of design days
        count3 = 0
        for i in days:
            b = sun_pos(i,self.time_zone,self.latitude,self.longitude) # create sun_pos object called a
            b.sun_angles(12, 0)
            
            azimuth_1[count3] = b.azimuth
            altitude_1[count3] = b.altitude # repeat altitude for 360 and 0
            altitude_1[count3+3] = b.altitude
            
            count3 += 1
        azimuth_1[3] = 360
        azimuth_1[4] = 360
        azimuth_1[5] = 360
        
        count = 0
        
        # solar angles for local clock time of design days
        for k in days: # loop through the three days
            a = sun_pos(k,self.time_zone,self.latitude,self.longitude) # create sun_pos object called a
            for i in hours:
                a.local_to_solar(i) # call object from module for sun angles calcs
                a.sun_angles(0,1)
                
                azimuth[count] = a.azimuth
                altitude[count] = a.altitude
                if altitude[count] < 0:
                    altitude[count] = 0
                    
                count = count + 1
            
        # add solar noon angles to local clock time arrays
        azimuth = np.append(azimuth,azimuth_1)
        altitude = np.append(altitude,altitude_1)
        
        # resource for square to cirle correlation
        resource = np.column_stack((DNI_design_days, altitude,azimuth)) 
        
        # set up resource data for optical efficiency interpolation
        for i in range(len(resource[:,1])):
            if resource[i,1] > 0:
                resource[i,0] = 1000 # used to generate optical efficiency data
            else:
                resource[i,0] = 0
                
        self.resource = resource
        return resource
    
    ###########################################################################    
    # generate interpolation map
    ###########################################################################
    
    def interpolation_map(self): # remember to changer number of helios here
        
        # get moments to simulate using ray-tracer
        moment_data = self.design_day_sun_angles()
        
        # loop single moment simulations for all design day moments

        results = np.zeros(len(moment_data[:,1]))
        time_before = time.time()
        
        for k in range(len(moment_data[:,1])): # loop for each time-step of the selected design days
            if moment_data[k,1] > 0: # if altitude angle is positive, ie dont simulate night
                # read and write JSON settings files
                with open(self.json_input_directory + '/moments.json') as moment: #open json and change angles and DNI value for the moment simulation
                    file = json.load(moment)
                    
                file['momentsdata'][0]['azimuth'] = moment_data[k,2] # represents design days data
                file['momentsdata'][0]['altitude'] = moment_data[k,1]
                file['momentsdata'][0]['irradiation'] = moment_data[k,0]
                
                with open(self.json_input_directory+ '/moments.json','w') as new_moment: #save as json file (serialization)
                    json.dump(file,new_moment,indent = 4)
              
                #Run ray-tracer simulation for each unique moment
                results[k] = self.run_ray_tracer()

            else:
                results[k] = 0
                
        time_after = time.time()
        # print('Computation time: ', time_after - time_before)
        
        # generate interpolation map
        # return results, moment_data
        design_power = 1000*self.num_helios*self.facet_width*self.facet_height # design_dni * #helios * facet_area - 115 for solarpilot modified helio field
        opt_eff_design_days = results/design_power
        
        # design days sun position and optical efficiency
        x = moment_data[:,2]
        y = moment_data[:,1]
        z = opt_eff_design_days
        
        cart_coords = list(zip(x,y))
        interp_function = LinearNDInterpolator(cart_coords, z,fill_value = 0)
        
        X = np.linspace(0,360)
        Y = np.linspace(0,90)
        
        XX, YY = np.meshgrid(X, Y)
        xx,yy = np.meshgrid(x,y)
        
        Z = interp_function(XX,YY) # this function is used to interp onto the surface given azimtuh and altitude angles.
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        dp = ax.scatter(x, y, z, zdir='z',s=25,c='r',marker='*')
        ax.plot_surface(XX,YY,Z,cmap='viridis')
        ax.set(xlabel='Azimuth', ylabel = 'Altitude', zlabel = 'Optical Efficiency')
        fig.show()
        
        return interp_function # return the interp function
 
    ###########################################################################    
    # generate annual optical efficiency data
    ###########################################################################
    
    def annual_hourly_optical_eff(self):
        
        # generate interpolation map
        interp_function = self.interpolation_map()
        
        # generate hourly solar angles
        yearly_sun_angles = np.zeros((365*24,2))
        count = 0
        for k in np.arange(1,366,1): # loop through the three days
            a = sun_pos(k,self.time_zone,self.latitude,self.longitude) # create sun_pos object called a
            for i in np.arange(1,25,1):
                a.local_to_solar(i) # call object from module for sun angles calcs
                a.sun_angles(0,1)
                
                yearly_sun_angles[count,0] = a.azimuth
                yearly_sun_angles[count,1] = a.altitude
                if yearly_sun_angles[count,1] < 0:
                    yearly_sun_angles[count,1] = 0
                    
                count = count + 1

        # generate optical efficieny vector for every day of the year
        
        year_interp_eff = np.empty((8760),dtype=float)
        count = 0
        for k in np.arange(0,8760,1): # loop over every hour of the year
        
            interp_opt_eff = interp_function((yearly_sun_angles[k,0],yearly_sun_angles[k,1]))
            
        
            year_interp_eff[k] = interp_opt_eff
            
        # plt.figure()
        # plt.plot(range(8760),year_interp_eff,'*-',label='optical efficiency')
        # # plt.plot(range(8760),yearly_sun_angles[:,1]/90,'*-',label='Elevation angel')
        # # plt.plot(range(8760),DNI/max(DNI),'*-',label='DNI')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
        return year_interp_eff, yearly_sun_angles
    
#%% testing class

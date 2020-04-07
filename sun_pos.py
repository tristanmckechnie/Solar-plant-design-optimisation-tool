# -*- coding: utf-8 -*-
"""
Author: T McKecnie
Stellenbosch University
30 September 2019

Sun position calculator
"""

import math
import numpy as np
import csv

class sun_pos:
    ''' longitude: east is positive, west is negative
        Time xone / UTC: east is positive, west is negative
        latitude: north is positve, south is negative
        
            North: 0 or 360 degs
             *
             *
             *
        ************* East: 90 degs
             *
             *
             *
        South: 180 degs
        
        in order to fill the class with solar, local times and angles you must run;
        
        a = sun_pos(design_days[k],2,-25.88,29.23) # creates new object instance
        a.local_to_solar(i) # takes in desired local time and converts to solar time, ie fills both times
        a.sun_angles() # fills in the suns position based on geographic location and time
        
        '''
    def __init__(self,day,time_zone,lat,long):
        self.day = day
        self.time_zone = time_zone
        self.longitude = long
        self.latitude = lat
        
        self.solar_time = 0
        self.local_time = 0
        
        self.azimuth = 0
        self.altitude = 0
    
    def local_to_solar(self,given_time):
        
        self.local_time = given_time
        
        LSM = self.time_zone*15 # longitude of standard meridian
        LC = (LSM-self.longitude)/15
        x = 360*(self.day-1)/365.242 # this is in degrees
        x_rad = x * math.pi/180
        EOT = 0.258*math.cos(x_rad) - 7.416*math.sin(x_rad) - 3.648*math.cos(2*x_rad) - 9.228*math.sin(2*x_rad) # equations of time
        
        self.solar_time = self.local_time + EOT/60 - LC 
        # print('Solar time: ', self.solar_time)
    def sun_angles(self,provided_solar_time,boolean): # this function can use the above calculated solar time (boolean == 1) or the time given
        
        if boolean == 1: # calc solar time
            time = self.solar_time
        else: 
            time = provided_solar_time
         
        delta = math.asin(0.39795*math.cos(0.98563*(self.day-173)*(math.pi/180))) # declination angle [radians]
        hour_angle = 15*(time - 12)*(math.pi/180) # hour angle [radians], remember to use solar time here
        # print('Declination angle: ', math.degrees(delta))
        # print('Hour angle: ', math.degrees(hour_angle))
        
        self.altitude = math.asin(math.sin(delta)*math.sin(self.latitude*(math.pi/180)) + math.cos(delta)*math.cos(hour_angle)*math.cos(self.latitude*(math.pi/180)))*(180/math.pi)
        # print('Altitude angle: ', self.altitude)
        
        temp = (math.sin(delta)*math.cos(math.radians(self.latitude)) - math.cos(delta)*math.cos(hour_angle)*math.sin(math.radians(self.latitude)))/math.cos(math.radians(self.altitude))
        if temp > 1 or temp < -1:
            print('Domain error for arccosine')
            if self.altitude > 0:
                if self.latitude > 0: 
                    self.azimuth = 180
                else:
                    self.azimuth = 0
            else:
                if self.latitude > 0: 
                    self.azimuth = 0
                else:
                    self.azimuth = 180
                
        else:    
            A_dash = math.acos((math.sin(delta)*math.cos(math.radians(self.latitude)) - math.cos(delta)*math.cos(hour_angle)*math.sin(math.radians(self.latitude)))/math.cos(math.radians(self.altitude))) # equation 3.19 power from the sun [radians]
            
            if math.sin(hour_angle) > 0: 
                self.azimuth = 360 - A_dash*180/math.pi
            else:
                self.azimuth = A_dash*180/math.pi
            # print('Azimuth angle: ', self.azimuth)

        
    
        

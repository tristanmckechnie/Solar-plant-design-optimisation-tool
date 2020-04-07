# -*- coding: utf-8 -*-
"""
Author: T McKecnie
Stellenbosch University
30 September 2019

3D interpolator

solution from: https://stackoverflow.com/questions/36210769/plotting-interpolated-values-using-linearndinterpolator-python
"""


from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%% Data

DNI = np.genfromtxt('./validation/DNI_data.csv',delimiter=',')
rec_therm = np.load('./validation/results_year.npy') # simulated receiver results  
resource_data = np.load('./validation/resource_data.npy')

#%% Plot DNI data vs Qtherm data

ARRAY = np.array([DNI[:,1],rec_therm])

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax2.plot(range(8760),rec_therm/1e6,'g',label='Qrec')
ax.plot(range(8760),DNI[:,1], 'b',label='DNI')
ax.set_ylabel("DNI [W/m^2]",fontsize=14,color='blue')
ax2.set_ylabel("Q_rec [MW]",fontsize=14,color='green')
#ax.set_ylim(ymax=100)
#ax.set_xlim(xmax=100)
ax.grid(True)
plt.title("DNI vs Qrec", fontsize=20,color='black')
ax.set_xlabel('Time [hours]', fontsize=14, color='b')
plt.show()

#%%
plt.figure()
plt.plot(range(8760),resource_data[:,1],'b*-')
plt.grid(True)
plt.show()
#%% Solstice 1 & 2  and equinox sun angles and Qrec simulation results for those hours

results_3d = np.load('./validation/results_3d.npy')
resource_3d = np.load('./validation/resource_3d.npy')

n = [79,172,356] # days for interpolation

azimuth = resource_3d[:,2]
altitude =resource_3d[:,1]
Qrec = results_3d
DNI_2 = resource_3d[:,0]

Opt_eff = np.zeros(len(results_3d))
for i in range(len(results_3d)):
    if DNI_2[i] == 0:
        Opt_eff[i] = 0    
    else:
        Opt_eff[i] = Qrec[i]/(DNI_2[i]*1608*1.83*1.22)

# plot altitude to check days are correct
plt.figure()
plt.plot(range(len(results_3d)),altitude,'b*-')
plt.grid(True)
plt.show()


#%% 3D interpolation plot and function
data = np.genfromtxt('data_full.csv', delimiter = ',') # azimuth,altitude,efficiency
x = azimuth#np.array(data[:,1])
y = altitude#np.array(data[:,0])
z = Opt_eff#np.array(data[:,2])

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

#%% Compare interp data vs actual data

# select random day to test
all_interp_eff = np.empty((1),dtype=float)
all_sim_eff = np.empty((1),dtype=float)

for k in range(365):
    test_day = k
    
    test_azimuth = resource_data[test_day*24:test_day*24 + 24,2]
    test_altitude = resource_data[test_day*24:test_day*24 + 24,1]
    test_dni  = resource_data[test_day*24:test_day*24 + 24,0]
    test_output = rec_therm[test_day*24:test_day*24 + 24]
    #
    interp_opt_eff = np.zeros(24)
    test_opt_eff = np.zeros(24)
    for i in range(24):
        interp_opt_eff[i] = interp_function((test_azimuth[i],test_altitude[i]))
        if test_dni[i-1] > 0 and interp_opt_eff[i] == 0 and test_dni[i+1] > 0:
            interp_opt_eff[i] = interp_opt_eff[i-1]
        if test_dni[i] == 0:
            test_opt_eff[i] = 0
        else:
            test_opt_eff[i] = test_output[i]/(test_dni[i]*1608*1.83*1.22)
    all_interp_eff = np.append(all_interp_eff,interp_opt_eff)
    all_sim_eff = np.append(all_sim_eff,test_opt_eff)
    
plt.figure()
plt.plot(range(6500,6788,1),all_interp_eff[6500:6788],'b*-',label='interpolated')
plt.plot(range(6500,6788,1),all_sim_eff[6500:6788],'g*-',label='simulated')
plt.grid(True)
plt.legend()
plt.ylim([0 ,1])
plt.ylabel('Optical effciency')
plt.xlabel('Time [hr]')
plt.show()

#%% total interp and simulated power - for comparison
total_interp_power = sum(all_interp_eff[1:8761]*resource_data[:,0]*1608*1.83*1.22)
total_simulated_power = sum(all_sim_eff[1:8761]*resource_data[:,0]*1608*1.83*1.22)
total_simulated_power2 = sum(rec_therm[:])
#%% annual average optical efficiency by interp and simulated - for comparison

annual_interp_efficency = total_interp_power/sum(resource_data[:,0]*1608*1.83*1.22)
annual_simulated_efficency = total_simulated_power/sum(resource_data[:,0]*1608*1.83*1.22)
#%%





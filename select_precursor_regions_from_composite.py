# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:14:41 2022

@author: dicapua

"""

import matplotlib
#matplotlib.rcParams['backend'] = "Qt4Agg" #!!!!!!!!!!!!!!!!!!!!!!!!!!! not working
matplotlib.use('TkAgg')

from cartopy import config
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#matplotlib.rcParams['backend'] = "Qt4Agg"
from pylab import *
from netCDF4 import Dataset
#import math
import numpy as np
#import seaborn as sns
import scipy
from scipy import signal
from statsmodels.sandbox.stats import multicomp
import sys, os
import seaborn as sns

import string
   
import scipy
from scipy.stats import linregress
           
# *****************************************************************************
# CREATE MASK for REGIONS
# *****************************************************************************
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#def region_number(mask_sign):
mask_sign = np.copy(mask_reg)
print(mask_sign.shape[0])
    
print('region_number is running')

# init
x = np.arange(mask_sign.shape[1])  #[1, 2, 3, 4]
y = np.arange(mask_sign.shape[0])  #[1, 2, 3, 4]
m = mask_sign # [[15, 14, 13, 12], [14, 12, 10, 8], [13, 10, 7, 4], [12, 8, 4, 0]]

cs = plt.contour(x, y, m)
count = 1  

# *****************************************************************************
# create region label
# *****************************************************************************
import scipy as sp

#arr = np.array([[1, 0, 0, 1, 1, 0],
#                [1, 1, 0, 0, 0, 1],
#                [0, 0, 1, 1, 0, 1],
#                [1, 1, 0, 1, 1, 0],
#                [1, 0, 0, 1, 1, 0]])

arr = mask_reg
labels, num_regions = sp.ndimage.label(arr)

print(labels)


lons = lons_all[(lons_all>=lon_west)&(lons_all<=lon_east)]
lats = lats_all[(lats_all>=lat_south)&(lats_all <=lat_north)]
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = ccrs.PlateCarree())

cs = plt.contour(lons, lats, m,  transform=ccrs.PlateCarree(),)


# *****************************************************************************
# plot        
# *****************************************************************************

fig = plt.figure()
fig.set_size_inches(20,10)
                 

# plot 1


field_max =   0.4#np.max(field) # 
field_min =  -0.4#np.min(field)# 

bins = 0.05 #(field_max -field_min)/10.
clevs = np.arange(field_min, field_max + bins, bins)

ax = fig.add_subplot(3, 3,1, projection = ccrs.PlateCarree())

x = np.arange(mask_reg.shape[1])  #[1, 2, 3, 4]
y = np.arange(mask_reg.shape[0])  #[1, 2, 3, 4]
m = mask_reg # [[15, 14, 13, 12], [14, 12, 10, 8], [13, 10, 7, 4], [12, 8, 4, 0]]
    


# add region number
for i in np.arange(1,num_regions+1,1):
    mask1 = labels==i #1
    plt.contour(lons, lats, mask1,  transform=ccrs.PlateCarree())
    lo1 = np.where(labels == i)[1][0]
    la1 = np.where(labels == i)[0][0]
    print(str(lons[lo1])+' '+str(lats[la1]))
    plt.text(lons[lo1],lats[la1], str(labels[la1, lo1]), fontsize = 'large')

ax.coastlines() 

ax.add_feature(ccrs.cartopy.feature.COASTLINE, edgecolor='lightgrey')
plt.contourf(lons, lats, corr_map, clevs, cmap =  plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extend = 'both')

plt.colorbar(orientation="horizontal", fraction=0.06,  pad=0.015)

segments = cs.allsegs


colors = ['r', 'b', 'k','m','g','y', 'o', 'royalblue']

cmap = matplotlib.cm.get_cmap('Spectral')

rgba = cmap(0.5)
print(rgba)
count = 0.1
#for contour_nb in range(0,3):
region_mask = np.zeros((mask_sign.shape[0], mask_sign.shape[1]))

for polygon in segments[0]:
    #print(contour_nb)
    xs, ys = zip(*polygon)
    rgba = cmap(0.5*count)
    print(rgba)
    ax.fill(xs,ys,color=rgba,transform=ccrs.PlateCarree())
    count = count+0.1
    
    
# plot 2

ax = fig.add_subplot(3, 3,2, projection = ccrs.PlateCarree())

plt.contourf(lons, lats, corr_map, clevs, cmap =  plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extend = 'both')
ax.coastlines() 

ax.add_feature(ccrs.cartopy.feature.COASTLINE, edgecolor='lightgrey')
plt.colorbar(orientation="horizontal", fraction=0.06,  pad=0.015)

cmap = matplotlib.cm.get_cmap('Spectral')

rgba = cmap(0.5)
print(rgba)
count = 0.05

polygon = segments[0][0]
#print(contour_nb)
xs, ys = zip(*polygon)
rgba = cmap(0.5*count)
print(rgba)
ax.fill(xs,ys,color=rgba,transform=ccrs.PlateCarree())
count = count+0.1    
    
# plot 3



#print(mask1)
cmap = matplotlib.cm.get_cmap('Spectral')
region_timesrs = np.zeros((n_years*time_cycle, labels.max().astype(int)))

for i in np.arange(1,num_regions+1,1):
    mask1 = labels==i #1
    if i <=7:
        field = np.ma.array(corr_map, mask = np.invert(mask1))
        ax = fig.add_subplot(3, 3,2+i, projection = ccrs.PlateCarree())
        plt.contour(lons, lats, mask1,  transform=ccrs.PlateCarree())
        plt.contourf(lons, lats, field, clevs, cmap =  plt.cm.Spectral, transform=ccrs.PlateCarree(), extend = 'both')
  
        ax.coastlines() 
        
        ax.add_feature(ccrs.cartopy.feature.COASTLINE, edgecolor='lightgrey')
        plt.colorbar(orientation="horizontal", fraction=0.06,  pad=0.015)
        plt.title(str(i))
     
        lo1 = np.where(labels == i)[1][0]
        la1 = np.where(labels == i)[0][0]
        print(str(lons[lo1])+' '+str(lats[la1]))
        plt.text(lons[lo1],lats[la1], str(labels[la1, lo1]), fontsize = 'large')


    # *****************************************************************************
    # CALCULATE time series for each region
    # *****************************************************************************
    for j in range(n_years*time_cycle):
        Field_v_j = Field_v[j,:,:]
        region_timesrs[j, i-1] = np.mean(Field_v_j[np.where(labels == i)])
    
    print(region_timesrs.shape)
    #plt.plot(region_timesrs[:, i])    


plt.savefig('Figure_set2_FIG3_ERA5_corr_maps_Etesians_JAS_Z200_identifyRegions_'+noIAV+'.pdf')

plt.show()


# save
labels.dump('ERA5_1981-2023_3-day_Z200_prec_Etesians_JAS_region_mask_lag1_'+str(pval)+'_'+str(c_val)+noIAV)
region_timesrs.dump('ERA5_1981-2023_3-day_Z200_prec_Etesians_JAS_region_timesrs_lag1_'+str(pval)+'_'+str(c_val)+noIAV)



"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

map_proj = ccrs.Orthographic(central_latitude=45.0, central_longitude=150.0)

map_proj._threshold /= 100.  # the default values is bad, users need to set them manually

ax = plt.axes(projection=map_proj)

ax.set_global() # added following an answer to my question
ax.gridlines()

ax.coastlines(linewidth=0.5, color='k', resolution='50m')

lat_corners = y1#np.array([-20.,  0., 50., 30.])
lon_corners = x1 #np.array([ 20., 90., 90., 30.]) + 15.0 # offset from gridline for clarity

poly_corners = np.zeros((len(lat_corners), 2), np.float64)
poly_corners[:,0] = lon_corners
poly_corners[:,1] = lat_corners

poly = mpatches.Polygon(poly_corners, closed=True, ec='r', fill=True, lw=1, fc="yellow", transform=ccrs.Geodetic())
ax.add_patch(poly)

plt.show()

"""   

#print('matplotlib: {}'.format(matplotlib.__version__))



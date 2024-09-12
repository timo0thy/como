# -*- coding: utf-8 -*-

# Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#from netCDF4 import Dataset 
#%matplotlib inline     
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import sklearn
import sys, os
import numpy
import scipy
from scipy import signal

import warnings
import math
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
#from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb # tig4.1
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.models import LinearMediation, Prediction
from tigramite.models import Models
from tigramite.toymodels import structural_causal_processes as toys

import iris
import iris.coord_categorisation as coord_cat
import pandas as pd
import xarray as xr

from dask.diagnostics import ProgressBar

# Load datasets
z500_GL_mean = iris.load_cube ('z500_GL_mean_ECEarth_2001-14.nc', 'zg500')
t2m_EMed_mean = iris.load_cube ('t2m_EMed_mean_ECEarth_2001-14.nc', 'tas')
t2m_UK_mean = iris.load_cube ('t2m_UK_mean_ECEarth_2001-14.nc', 'tas')

# Standardise and detrend
z500_trend = z500_GL_mean - signal.detrend(z500_GL_mean.data)
z500_det = z500_GL_mean  - z500_trend
Z500 = (z500_det - np.mean(z500_det.data))/np.std(z500_det.data)
t2m_trend = t2m_EMed_mean - signal.detrend(t2m_EMed_mean.data)
t2m_det = t2m_EMed_mean  - t2m_trend
T2M_EMED = (t2m_det  - np.mean(t2m_det.data))/np.std(t2m_det.data)
t2m_trend = t2m_UK_mean - signal.detrend(t2m_UK_mean.data)
t2m_det = t2m_UK_mean  - t2m_trend
T2M_UK = (t2m_det  - np.mean(t2m_det.data))/np.std(t2m_det.data)

# PCMCI
data = np.array ([Z500.data, T2M_EMED.data, T2M_UK.data]).T
var_names = ['Greenland Z500', 'EMed temp', 'UK temp']
dataframe = pp.DataFrame(data, 
                         datatime = np.arange(len(T2M_EMED.data)), 
                         var_names=var_names)
			 #mask=data_mask

parcorr = ParCorr(significance='analytic', verbosity=4)
pcmci = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=parcorr,
    verbosity=4)
#correlations = pcmci.get_lagged_dependencies(tau_max=8, val_only=True)['val_matrix']
#pcmci.verbosity = 4
results = pcmci.run_pcmciplus(tau_max=8, pc_alpha=None)
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=8, fdr_method='fdr_bh', exclude_contemporaneous=False)
pcmci.print_significant_links(
        p_matrix = q_matrix, 
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)
graph = pcmci.get_graph_from_pmatrix(p_matrix=results['p_matrix'], alpha_level=0.01, tau_min=0, tau_max=8)
results['graph'] = graph
all_parents = pcmci.return_parents_dict(graph, val_matrix=results['val_matrix'], include_lagzero_parents=True) # include_lagzero_parents=False
#pcmci.graph_to_dict(graph)  
print(all_parents)
link_matrix = graph

# Just let the model find the links
# Beta coefficient
from tigramite.models import Models
med = Models(dataframe=dataframe,            
             model = sklearn.linear_model.LinearRegression(),
            data_transform = None)    
med.fit_full_model(all_parents = all_parents, tau_max=8)           
Links = med.get_val_matrix()
# save Links
Links.dump('GBtoEMed_withUK_Pplus')       
tp.plot_graph(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='coef',
    node_colorbar_label='auto-coef',
    figsize = [8,8],
    vmin_edges = -0.5,
    vmax_edges = 0.5,
    vmin_nodes = -1.0,
    vmax_nodes = 1.,
    cmap_nodes = 'RdBu_r',
    alpha = 1,
    node_label_size = 15,
    link_label_fontsize  = 10,
    show_autodependency_lags=False, save_name='GBtoEMed_WithUK_Pplus_ECEarth_allseasons.pdf')

# Constraining the links based on the observation
# Control with all parents
all_parents = pcmci.return_parents_dict(graph, val_matrix=results['val_matrix'], include_lagzero_parents=True)
# Constrain the parents...
#['Greenland Z500', 'EMed temp', 'UK temp']
all_parents = {0: [(0, -1), (1, 0)], 1: [(1, -1),(0, 0)], 2: [(2, -1), (0, -2), (0, -1)]}
#link_matrix = graph

# initialise graph
graph = np.zeros((3,3,3),  dtype='<U3')
for i in range(3):
    for j in range(3):
        for tau in range(3):
            graph[i, j, tau] = ''
            
graph[0, 1, 0] = 'o-o'# '-->'
graph[1, 0, 0] = 'o-o'

graph[0, 0, 1] = '-->'
graph[1, 1, 1] = '-->'
graph[2, 2, 1] = '-->'
graph[0, 2, 2] = '-->'
graph[0, 2, 1] = '-->'

# Calculate beta coeff.
med = Models(dataframe=dataframe,            
             model = sklearn.linear_model.LinearRegression(),
             data_transform = None)    

med.fit_full_model(all_parents = all_parents, tau_max=8)     
      
Links = med.get_val_matrix()

Links[0,1,0] = Links[1,0,0]
Links[0,2,0] = Links[2,0,0]
Links[1,2,0] = Links[2,1,0]

# save Links
Links.dump('GBtoEMed_WithUK_Pplus_constrainedlinks')

tp.plot_graph(
    val_matrix=Links,
    graph=graph,
    var_names=var_names,
    link_colorbar_label='coef',
    node_colorbar_label='auto-coef',
    figsize = [8,8],
    vmin_edges = -0.5,
    vmax_edges = 0.5,
    vmin_nodes = -1.0,
    vmax_nodes = 1.,
    cmap_nodes = 'RdBu_r',
    alpha = 1,
    node_label_size = 15,
    link_label_fontsize  = 10,
    show_autodependency_lags=False, save_name='GBtoEMed_WithUK_Pplus_constrainedlinks_ECEarth_allseasons.pdf')
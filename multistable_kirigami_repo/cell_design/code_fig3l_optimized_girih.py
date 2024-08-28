# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:25:53 2023

@author: Masca
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.collections as coll
import itertools as it
from mpl_toolkits.mplot3d import Axes3D
from tile_design_functions import *
from tile_design_dictionaries import *
from scipy.optimize import minimize
import pickle
from copy import deepcopy

def unpickle(filename):
    inputfile = open(filename,'rb')
    pickled = pickle.load(inputfile)
    inputfile.close()
    return pickled

def nupickle(data,filename):
    outputfile = open(filename,'wb')
    pickle.dump(data,outputfile,protocol=pickle.HIGHEST_PROTOCOL)
    outputfile.close()

###############################################################################
#user input

#cost function weights
Dweight = 10000. #ensure points lie inside cell
Aweight = 200. #ensure hinge point 2 lies close to cell edge for tree-bar linkage behaviour
Bweight = 100. #ensure hinge point 1 lies at good fabrication location
Cweight = 100. #ensure distance between paired cell hinge points is small for three-bar linkage behaviour

s = 1.4 #expansion factor
Girih_length = 24 #base length (mm)
h = 0.7 #cut width (mm)
d1_reg = 2. #match length for general struts between cells
vocal = False #print optimization results
newfolder = r'./Girih_corner_g-{:.0f}mm_s-{:.2f}_h-{:.1f}mm_weights-{:}'.format(Girih_length, s, h, [Dweight, Aweight, Bweight, Cweight])
tiledict =  generate_tiledict_Girih_corner(Girih_length)
matchdict = get_matchdict_Girih_corner()

###############################################################################
#optimize cells
fig, ax = plt.subplots(1,1, figsize=[9,3])
ax.axis('equal')
ax.set_xlabel(r'mm')
ax.set_ylabel(r'mm')
ax.axis('equal')
newtiledict = deepcopy(tiledict)
if not os.path.exists(newfolder):
    os.mkdir(newfolder)
for ki, k in enumerate(tiledict.keys()):
    print('optimizing', k)
    td = tiledict[k]
    a = td['length_a']
    b = td['length_b']
    gamma = td['base_angle']
    c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(gamma))
    d1 = d1_reg
    if 'd_2' in td.keys():
        d2 = td['d_2']
    else: 
        d2 = 1.

    paired_td = tiledict[matchdict[k]]
    a2 = paired_td['length_a']
    b2 = paired_td['length_b']
    gamma2 = paired_td['base_angle']
    c2 = np.sqrt(a2**2 + b2**2 - 2*a2*b2*np.cos(gamma2))
    s2 = s
    
    return_best_d2 = False
    args = [a, b, gamma, s, d1, d2, h, False, b2, gamma2, s2, Dweight, Aweight, Bweight, Cweight, return_best_d2]
    x0 = [1,2,1, 2] #ch, deltat first guess
    bounds = ((d1+h*1.1, c-h*1.1),(1e-2, np.pi/2-1e-2), (d1+h*1.1, c2-h*1.1),(1e-2, np.pi/2-1e-2))
    res = minimize(cost_function_fabrication_pairwise_v5, x0=x0, args=args,
                   bounds=bounds)
    opt_ch, opt_deltat, opt_ch_2, opt_deltat_2 = res.x
    
    newargs = args = [a, b, gamma, s, d1, d2, h, vocal, b2, gamma2, s2, Dweight, Aweight, Bweight, Cweight, True]
    finalcost, d2_opt = cost_function_fabrication_pairwise_v5(res.x, newargs)
    
    
    print('\t optimized length, angle, and d2:', opt_ch, opt_deltat, opt_ch_2, opt_deltat_2, d2_opt)
    success, triangle_points = generate_fabrication_tile(a=a, b=b, gamma=gamma, sprime=s,
                  ch=opt_ch, deltaprime=opt_deltat, 
                  d1=d1, d2=d2_opt,
                  h=h, vocal=vocal)
    fig = draw_lasercut_fabrication_points(*triangle_points, ax=ax, draw_offset=np.array([Girih_length*ki*3,0.]))
    newtiledict[k] = triangle_points  
fig.savefig(newfolder + r'/Girih_tiles_optimized.pdf')
fig.savefig(newfolder + r'/Girih_tiles_optimized.png')

###############################################################################
#visualize cells
fig, ax = plt.subplots(1,1, figsize=[9,6])
ax.set_xlabel(r'mm')
ax.set_ylabel(r'mm')
ax.axis('equal')
drawtiledict = tesselate_Girih_cells_corner(newtiledict, tiledict)
for ki, k in enumerate(drawtiledict.keys()):
    cellname = k
    ax.text(Girih_length*ki*3, -3*Girih_length, s=cellname, fontsize=8, rotation=30)
    for drawpoints in drawtiledict[k]:
        draw_lasercut_fabrication_points(*drawpoints, ax=ax, draw_offset=np.array([Girih_length*ki*3,0.]))
fig.savefig(newfolder + r'/Girih_cells_optimized.pdf')
fig.savefig(newfolder + r'/Girih_cells_optimized.png')

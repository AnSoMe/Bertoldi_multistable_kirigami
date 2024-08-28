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
import pandas as pd

def rc_layout():    
    plt.rcParams['figure.autolayout']=True
    plt.rcParams['font.size']=12
    plt.rcParams['legend.edgecolor']='1'
    plt.rcParams['pdf.fonttype']=42

rc_layout() 
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

# a = 10 #mm
h = 0.7 #mm
# gamma_vals = np.linspace(np.pi/20, np.pi/2-np.pi/20, 50)
# e_vals = np.linspace(h, 8, 100)

gamma_vals_todraw = [np.pi/10, np.pi/4, np.pi/3, 4*np.pi/10.]
beta_vals_todraw = [np.pi/10, np.pi/4, np.pi/3, 4*np.pi/10.]
# e_vals_todraw = np.linspace(h, 8, 6)
# e_vals_todraw = np.linspace(0, 8, 6)
# a_vals_todraw = [10, 20, 30, 40]
# c_vals_todraw = [15]
b_vals_todraw = [24]
smax = 4.

###############################################################################
#designs

if not os.path.exists(r'./parallel_designsweep'):
    os.mkdir(r'./parallel_designsweep')

fig, ax = plt.subplots(1,1, figsize=[24,12])
ax.set_xlabel(r'mm')
ax.set_ylabel(r'mm')
ax.axis('equal')

offset_x = 0
offset_y = 0

my_y_offset = 60.

# for ai, aval in enumerate(a_vals_todraw):
# for ci, c in enumerate(c_vals_todraw):
for bi, b in enumerate(b_vals_todraw):
    
    df = pd.DataFrame()
    mydict = {}
    ctr = 0
    for gi, gamma in enumerate(gamma_vals_todraw):
        
        # aval = c/np.tan(gamma) #right angles
        # b = aval/np.cos(gamma) #simplest choice
        # aval = b*np.cos(gamma) #simplest choice
        beta = beta_vals_todraw[gi]
        aval = 2*b*np.cos(gamma)
        # aval = beta/(2*np.cos(gamma))
        c = b
        
        # c = np.sqrt(aval**2 + b**2 - 2*aval*b*np.cos(gamma))
        offset_x += 3*c
        
        # e_vals_todraw = np.linspace(h, c/2., 6)
        
        # beta = np.arccos((b**2 - aval**2 - c**2)/(-2*aval*c))
        alpha = np.pi-gamma-beta
        max_e = (c - h/np.sin(alpha))/(1/np.sin(alpha) + 1/np.sin(beta))*1.2
        # print(h/np.sin(alpha))
        e_vals_todraw = np.linspace(1, max_e, 5)
        e_vals_todraw = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4., 4.5, 5, 5.5, 6., 7.5, 7, 8.5, 8.])

        
        for ei, e in enumerate(e_vals_todraw):
            # if ei == len(e_vals_todraw)-1: continue
            success, triangle_points = generate_parallel_tile(a=aval, b=b, gamma=gamma, h=h, e=e, vocal=False)
            [p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2] = triangle_points
            s = expansion_factor_lowest_frompoints(a=aval, j_1=j_1, j_2=j_2)
            if True: 
                draw_lasercut_fabrication_points(*triangle_points, ax=ax, draw_offset=np.array([offset_x,offset_y]),
                                                          cut_color=cm.magma((s-1)/smax))   
                drawstring = r'design {:d}: $\gamma={:.2f}$, $d={:.1f}$'.format(ctr, gamma, e)
                ax.text(offset_x-c, offset_y-aval, s=drawstring, fontsize=8, rotation=-30)
            
            
            #calculate unit cell area
            vec_gamma_alpha = p_alpha - p_gamma
            vec_gamma_beta = p_beta - p_gamma
            cell_area = la.norm(np.cross(vec_gamma_beta, vec_gamma_alpha))
            #calculate rotating unit area
            vec_r_1_reduced = e_1 - j_2
            vec_alpha_beta = p_beta - p_alpha
            pt_tri_1 = find_crossing_point(e_1, vec_r_1_reduced, p_beta, vec_alpha_beta)
            vec_r_2_reduced = j_1 - h_2
            pt_tri_2 = find_crossing_point(j_1, vec_r_2_reduced, p_beta, vec_alpha_beta)
            pt_tri_3 = find_crossing_point(j_1, vec_r_2_reduced, e_1, vec_r_1_reduced)
            rotating_area = la.norm(np.cross(pt_tri_1-pt_tri_3, pt_tri_2-pt_tri_3))
            ru_pts = np.vstack([pt_tri_1, pt_tri_2, pt_tri_3, pt_tri_1])
            # ax.scatter(offset_x +pt_tri_1[0], offset_y + pt_tri_1[1], c='k')
            # ax.scatter(offset_x +pt_tri_2[0], offset_y + pt_tri_2[1], c='b')
            # ax.scatter(offset_x +pt_tri_3[0], offset_y + pt_tri_3[1], c='r')
            ax.fill(ru_pts[:, 0]+offset_x, ru_pts[:, 1]+offset_y, color='yellow', alpha=0.3)
            #store a, b, c, gamma, e values
            # gamma_ru = 
            l_ru = la.norm(j_1 - j_2)
            gamma_ru = np.abs(np.arctan((j_1-j_2)[0]/(j_1-j_2)[1]))
            mydict.update({ctr: {'a [mm]':aval, 
                                'b [mm]': b,
                                'c [mm]': c,
                                'gamma [rad]': gamma,
                                'd  [mm]': e,
                                's* []': s,
                                'S_cell [mm^2]': cell_area,
                                'S_ru [mm^2]': rotating_area,
                                'S_ru/S_cell []': rotating_area/cell_area,
                                'gamma_ru': gamma_ru,
                                'l_ru': l_ru
                                }}) #add gamma_ru and l_ru
            ctr += 1
            offset_y += my_y_offset
            # sys.exit()
        df = df.from_dict(mydict, orient='index')
        offset_y = 0
        # sys.exit()
    fig.savefig(r'./parallel_designsweep/parallel_designsweep_selectdesigns_v2.pdf'.format(aval))
    fig.savefig(r'./parallel_designsweep/parallel_designsweep_selectdesigns_v2.png'.format(aval))
    df.to_csv(r'./parallel_designsweep/parallel_designsweep_selectdesigns_details_v2.csv')
sys.exit()
###############################################################################
#phase diagram
fig2, ax2 = plt.subplots(1,1, figsize=[3,2.5])
ax2.set_xlabel(r'$\gamma$  [rad]')
ax2.set_ylabel(r'$d$  [mm]')

offset_x = 0
offset_y = 0
all_g = []
all_e = []
all_s = []
for gi, gamma in enumerate(gamma_vals):
    b = a/np.cos(gamma) #simplest choice
    c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(gamma))
    # beta = np.arccos((b**2 - a**2 - c**2)/(-2*a*c))
    # alpha = np.pi-gamma-beta
    # max_e = (c - h/np.sin(alpha))/(1/np.sin(alpha) + 1/np.sin(beta))*1.2
    # e_vals = np.linspace(h*1.5, max_e-h, 52)
    for ei, e in enumerate(e_vals):
        success, triangle_points = generate_parallel_tile(a=a, b=b, gamma=gamma, h=h, e=e, vocal=False)
        [p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2] = triangle_points
        s = expansion_factor_lowest_frompoints(a=a, j_1=j_1, j_2=j_2)
        if success: 
            # draw_lasercut_fabrication_points(*triangle_points, ax=ax, draw_offset=np.array([offset_x,offset_y]),
            #                                           cut_color=cm.magma((s-1)/smax))
            # # if s > 2.4: print(s)
            all_s.append(s)
            all_g.append(gamma)
            all_e.append(e)
            # ax2.scatter(gamma, e, color=cm.magma((s-1)/smax), alpha=0.5)        
        offset_y += 3*a
    offset_x += 3*c
    offset_y = 0
rax = ax2.scatter(all_g, all_e, c=all_s, cmap=cm.magma, alpha=0.5, vmin=1, vmax=smax)  
plt.colorbar(rax, label='$a_{ext}/a$')
fig2.savefig(r'./parallel_designsweep/parallel_designsweep_expansionratios.pdf')
fig2.savefig(r'./parallel_designsweep/parallel_designsweep_expansionratios.png')

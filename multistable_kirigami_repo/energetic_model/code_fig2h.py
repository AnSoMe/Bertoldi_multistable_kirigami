import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.collections as coll
import itertools as it
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from scipy.optimize import minimize
import pickle
from copy import deepcopy
import scipy.linalg as la
from tile_design_functions import *
import pandas as pd

import matplotlib.tri as tri
import scipy.interpolate as sci

def rc_layout():    
    plt.rcParams['figure.autolayout']=True
    plt.rcParams['font.size']=12
    plt.rcParams['legend.edgecolor']='1'
    plt.rcParams['pdf.fonttype']=42

def unpickle(filename):
    inputfile = open(filename,'rb')
    pickled = pickle.load(inputfile)
    inputfile.close()
    return pickled


# def triangle_interior_check(v, p_alpha, p_beta, p_gamma):
#     D1 = (np.cross(v, p_alpha) - np.cross(p_gamma, p_alpha))/(np.cross(p_beta, p_alpha))
#     D2 = -(np.cross(v, p_beta) - np.cross(p_gamma, p_beta))/(np.cross(p_beta, p_alpha))
#     Dtot = D1 + D2

#     return D1, D2, Dtot

def nupickle(data,filename):
    outputfile = open(filename,'wb')
    pickle.dump(data,outputfile,protocol=pickle.HIGHEST_PROTOCOL)
    outputfile.close()
    
def rotate_point(point, rotation_point, rotation_angle, rotation_axis):
    rotmat = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                       [np.sin(rotation_angle), np.cos(rotation_angle)]])
    new_point = np.dot(rotmat, point - rotation_point) + rotation_point 
    return new_point

def rotate_points(points, rotation_point, rotation_angle):
    rotmat = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                       [np.sin(rotation_angle), np.cos(rotation_angle)]])
    new_points = np.array([np.dot(rotmat, point - rotation_point) + rotation_point for point in points])
    return new_points

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def get_torsion_constant(a, b):
    # https://en.wikipedia.org/wiki/Torsion_constant
    short = a
    long = b
    if a > b:
        long = a
        short = b
    J = (long*short**3)/16 * (16/3 - 3.36 *short/long * (1 - short**4/(12*long**4)))
    return J

def Mises_lagrangian(opt_args, args):
    
    p_sy = opt_args
    
    b_RU, c_RU, alpha, H, k_RU, k_theta, k_tau = args
    
    
    p_s = np.sqrt(c_RU**2 + p_sy**2 + H**2)
    dl_RU = p_s - b_RU
    dtheta_t = np.arctan(-H/c_RU) - alpha
    dtau_t = np.arcsin(p_sy/p_s)    
    
    
    energy = 1/2*k_RU*dl_RU**2 + k_theta*dtheta_t**2 + k_tau*dtau_t**2
    
    return energy

def get_max_d(b, gamma, h):
    aval = b*np.cos(gamma) #simplest choice, right-angle triangle
    c = np.sqrt(aval**2 + b**2 - 2*aval*b*np.cos(gamma))
    
    # e_vals_todraw = np.linspace(h, c/2., 6)
    beta = np.arccos((b**2 - aval**2 - c**2)/(-2*aval*c))
    alpha = np.pi-gamma-beta
    max_e = (c - h/np.sin(alpha))/(1/np.sin(alpha) + 1/np.sin(beta))*1.2
    return max_e

def get_cell_geometry(length_a, length_b, gamma, hinge_size, length_d):
        success, triangle_points = generate_parallel_tile(a=length_a, b=length_b, gamma=gamma, h=hinge_size, e=length_d, vocal=False)
        # if not success: return success, None, None, None
        [p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2] = triangle_points
        s = expansion_factor_lowest_frompoints(a=length_a, j_1=j_1, j_2=j_2)
        
        #calculate unit cell area
        vec_gamma_alpha = p_alpha - p_gamma
        vec_gamma_beta = p_beta - p_gamma
        cell_area = la.norm(np.cross(vec_gamma_beta, vec_gamma_alpha))
        length_c = la.norm(p_alpha-p_beta)
        
        #calculate rotating unit area
        vec_r_1_reduced = e_1 - j_2
        vec_alpha_beta = p_beta - p_alpha
        pt_tri_1 = find_crossing_point(e_1, vec_r_1_reduced, p_beta, vec_alpha_beta)
        print('hey', e_1, pt_tri_1)
        vec_r_2_reduced = j_1 - h_2
        pt_tri_2 = find_crossing_point(j_1, vec_r_2_reduced, p_beta, vec_alpha_beta)
        pt_tri_3 = find_crossing_point(j_1, vec_r_2_reduced, e_1, vec_r_1_reduced)
        rotating_area = la.norm(np.cross(pt_tri_1-pt_tri_3, pt_tri_2-pt_tri_3))
        ru_pts = np.vstack([pt_tri_1, pt_tri_2, pt_tri_3, pt_tri_1])
        
        #reduced cell area
        cell_area_reduced = la.norm(np.cross(p_beta-p_gamma, pt_tri_2-p_gamma))

        #store a, b, c, gamma, e values
        mydict = {'a [mm]':length_a, 
                'b [mm]': length_b,
                'c [mm]': length_c,
                'gamma [rad]': gamma,
                'd  [mm]': length_d,
                's* []': s,
                'S_cell [mm^2]': cell_area,
                'S_ru [mm^2]': rotating_area,
                'S_ru/S_cell []': rotating_area/cell_area,
                'S_cell_reduced [mm^2]': cell_area_reduced}
        return success, mydict, triangle_points, ru_pts

###############################################################################
#User variables. All units in N, mm.
ktauprefactors = [2]
kthetaprefactors = [1]

# %     Laser cutting patterns for periodic kirigami metamaterials 
# were designed as shown in Fig.~\ref{fig_simple_tessellations}a--d. 
# Quadrilateral designs were realized with cell side length $c=10$mm, 
# hinge width $h=0.7$mm, and strut thicknesses 
# $d\in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.4, 2.6)$mm. 
# Triangular designs were constructed using $c=12$mm, $h=0.7$mm, 
# and $d\in(1.0, 1.5, 2.0, 2.5, 3.0)$mm. Designs shown in fig.2 a-d have d=1.6 and d=2.0 (quad, triangle).
    
tess_hingewidth = 0.7 #mm
tess_avals = np.array([12.]*5 + [10*np.sqrt(2)]*10)
tess_bvals =np.array([12.]*5 + [10]*10) #mm
tess_gammavals = np.array([np.pi/3]*5 + [np.pi/4]*10) #rad
# tess_dvals = np.array([3., 2.5, 2., 1.5, 1.] + [2.6, 2.4,2.2,2,1.8,1.6,1.4,1.2,1,.8]) #mm
tess_markers = np.array(['H']*5 + ['D']*10)
tess_datadict = unpickle('./dvals_vs_energyratio.pkl')
tess_dvals = np.concatenate([tess_datadict['dvals_tri'], tess_datadict['dvals_quad']])
tess_energyratios = np.concatenate([tess_datadict['eratios_tri'], tess_datadict['eratios_quad']])

vmax=1.1

rc_layout()

overview_fig, overview_axs = plt.subplots(len(kthetaprefactors)+1,len(ktauprefactors)+1, figsize=[(len(ktauprefactors)+1)*3, (len(kthetaprefactors)+1)*3])
for ki, ktauprefactor in enumerate(ktauprefactors):
    for khi, kthetaprefactor in enumerate(kthetaprefactors):
        
        figname = r'./energy_sweep_ktautest_ktautest_{:.2f}_kthetatest_{:.2f}_test16_BC2_tesselations'.format(ktauprefactor, kthetaprefactor)
        datafolder = r'./energy_sweep_test16_ktauprefactor_{:.1f}_kthetaprefactor_{:.1f}/'.format(ktauprefactor, kthetaprefactor)
        df = pd.read_csv(datafolder + r'model_v15_sweep_result_dict.csv')
        
        #gather model data
        d_values = []
        py_values = []
        arearatio_values = []
        gamma_values = []
        gamma_RU_values = []
        lengths_RU_b = []
        lengths_RU_a = []
        new_arearatio_values = []
        nums_energy_minima = []
        energy_vals = []
        energy_free_proxies = []
        energy_conf_proxies = []
        for ri, row in df.iterrows():
            d_values.append(row['d  [mm]'])
            py_values.append(row['p_y_max [mm]'])
            arearatio_values.append(row['S_ru/S_cell []'])
            gamma_values.append(row['gamma [rad]'])
            gamma_RU_values.append(row['gamma_RU [rad]'])
            lengths_RU_b.append(row['length_RU_b [mm]'])
            lengths_RU_a.append(row['length_RU_b [mm]']*np.cos(row['gamma_RU [rad]']))
            new_arearatio_values.append(row['S_ru [mm^2]']/row['S_cell_reduced [mm^2]'])
            nums_energy_minima.append(row['num_energy_minima'])
            energy_free_proxies.append(row['energy_at_max_deflection [mNm]'])
            energy_conf_proxies.append(row['energy_at_confined_max_deflection [mNm]'])
                               
        #gather experimental data      
        tess_lrus = np.zeros_like(tess_avals)
        tess_gammarus =  np.zeros_like(tess_avals)
        for di, length_d in enumerate(tess_dvals):
            length_a = tess_avals[di]
            length_b = tess_bvals[di]
            gamma = tess_gammavals[di]
            hinge_size = tess_hingewidth
            success, mydict, triangle_points, ru_pts = get_cell_geometry(length_a, length_b, gamma, hinge_size, length_d)
            # p_gamma, p_beta, p_alpha, e_1, e_2, h_1, h_2, j_1, j_2
            [p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2] = triangle_points
            # draw_lasercut_fabrication_points(*triangle_points)
            pt_truss = j_1
            pt_strut = j_2
            length_RU_b = la.norm(pt_truss-pt_strut) #estimate, may be refined if needed (see design drawing)
            RU_unitvec = (pt_truss-pt_strut)/length_RU_b
            gamma_RU = np.arccos(RU_unitvec[1])
            tess_lrus[di] = length_RU_b
            tess_gammarus[di] = gamma_RU
            print(di, gamma, length_d, gamma_RU, length_RU_b)
            
        #smooth numerical data
        xi = np.linspace(np.min(gamma_RU_values), np.max(gamma_RU_values), 300)
        yi = np.linspace(np.min(lengths_RU_b), np.max(lengths_RU_b), 300)
        smoothing = None
        tck = sci.bisplrep(np.array(gamma_RU_values), np.array(lengths_RU_b), np.array(py_values), s=smoothing, kx=5, ky=5) 
        zi_interp = sci.bisplev(xi, yi, tck).T
        #mask values outside of convex hull of g_ru, l_ru range
        hull = np.vstack([np.array(gamma_RU_values), np.array(lengths_RU_b)]).T
        mask = np.zeros_like(zi_interp)
        for i in range(len(yi)):
            mypts = np.array([xi, np.ones_like(xi)*yi[i]]).T
            mask[i] = in_hull(mypts, hull)
        zi_interp[np.where( (mask==0) | (zi_interp<1e-3) )] = np.nan
        # zi_check = [sci.bisplev(np.array(gamma_RU_values)[i], np.array(lengths_RU_b)[i], tck) for i in range(len(gamma_RU_values))]
        # zi_check = np.array(zi_check)
        # zi_diff = zi_check - np.array(py_values)
        # print('contour deviation', zi_diff.min(), zi_diff.max())
        
        #plot numerical data for deflection
        plot_deflection = False
        plot_eratio_proxy = True
        
        if plot_deflection:
            hao = overview_axs[ki][khi].scatter(gamma_RU_values, np.array(lengths_RU_b), c=np.array(py_values), 
                                                cmap='YlGn',vmin=-0.1, vmax=13, alpha=.3, zorder=-1000, lw=0)   
            h2 = overview_axs[ki][khi].contour(xi, yi, zi_interp, 
                                                cmap='YlGn',vmin=-0.1, vmax=13, levels=20, zorder=1000, lw=2)        
            alpha_vals = np.array(nums_energy_minima) < 1.5
            alpha_vals = 0.1*alpha_vals.astype('float')
            ha2 = overview_axs[ki][khi].scatter(gamma_RU_values, np.array(lengths_RU_b), c='k',
                                                                      alpha = alpha_vals, zorder=1000, lw=0)
        
        ##############
        #plot numerical data for energy ratios (test)
        if plot_eratio_proxy:
            hao = overview_axs[ki][khi].scatter(gamma_RU_values, np.array(lengths_RU_b), c=np.array(energy_conf_proxies)/np.array(energy_free_proxies), 
                                                cmap='pink_r',vmin=0.9, vmax=2.5, alpha=1., zorder=-1000, lw=0)   
            
        #plot stability regimes
        alpha_vals = np.array(nums_energy_minima) < 1.5
        alpha_vals = 0.1*alpha_vals.astype('float')
        ha2 = overview_axs[ki][khi].scatter(gamma_RU_values, np.array(lengths_RU_b), c='k',
                                                                      alpha = alpha_vals, zorder=1000, lw=0)
           
            
        #plot experimental data for energy ratios
        h3 = overview_axs[ki][khi].scatter(tess_gammarus[tess_markers=='D'], tess_lrus[tess_markers=='D'], c=tess_energyratios[tess_markers=='D'], 
                                            cmap='pink_r', marker='D',
                                            zorder=1000, edgecolor='k',
                                            vmin=0.9, vmax=2.5)
        h3 = overview_axs[ki][khi].scatter(tess_gammarus[tess_markers=='H'], tess_lrus[tess_markers=='H'], c=tess_energyratios[tess_markers=='H'], 
                                            cmap='pink_r', marker='H',
                                            zorder=1000, edgecolor='k',
                                            vmin=0.9, vmax=2.5)
        


        overview_axs[ki][khi].set_title(r'$\tilde{K}_{\theta}$'+ r'={:.1f}'.format(kthetaprefactor) + r', $\tilde{K}_{\tau}$'+ r'={:.1f} '.format( ktauprefactor))
        overview_axs[ki][khi].set_xlabel(r'$\gamma_{\mathrm{RU}}$  [rad]')
        overview_axs[ki][khi].set_xlim(0, np.pi/2.)
        overview_axs[ki][khi].set_xlim(0.7,1.4)
        overview_axs[ki][khi].set_xticks([0.75,1,1.25])
        overview_axs[ki][khi].set_ylabel(r'$l_{RU}$  [mm]')
        overview_axs[ki][khi].set_ylim(0, 25)
        overview_axs[ki][khi].set_ylim(0, 12)
        overview_axs[ki][khi].set_yticks([0,5,10])
        
        
overview_axs[-1][-1].axis('off')
if plot_deflection:
    cbar = plt.colorbar(hao, ax=overview_axs[-1][-1],  label=r'$max({p}^{eq}_{y})$  [mm]')
if plot_eratio_proxy: 
    cbar = plt.colorbar(hao, ax=overview_axs[-1][-1],  label=r'$\mathcal{E}_{conf}/\mathcal{E}_{free}$  [ ]')
overview_fig.savefig(figname+'.pdf')
overview_fig.savefig(figname+'.png')

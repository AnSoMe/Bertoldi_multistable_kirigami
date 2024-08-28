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
        if not success: return success, None, None, None
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

# expdict = {1: {'p_y_max [mm]': 43*(22.9/98)*24/21}, #mm, measured from photos
#            0 : {'p_y_max [mm]': 40*(22.9/87)*24/21},
#            8 : {'p_y_max [mm]': 30*(22.9/99)*24/21},
#            4 : {'p_y_max [mm]': 40*(22.9/100)*24/21},
#            5 : {'p_y_max [mm]': 26*(22.9/101)*24/21},
#            }
# expdict = {1: {'p_y_max [mm]': 8*24/21}, #mm, measured from photos
#            0 : {'p_y_max [mm]': 12.3*24/21},
#            8 : {'p_y_max [mm]': 6.2*24/21},
#            4 : {'p_y_max [mm]': 10.8*24/21},
#            5 : {'p_y_max [mm]': 6.6*24/21},
#            }
# exp_err = 5*(22/9/95) #mm

###############################################################################
#User variables. All units in N, mm.

df_fabrication =  pd.read_csv(r'./parallel_designsweep/parallel_designsweep_selectdesigns_details.csv')
dict_fabrication =  unpickle(r'exp_overview_data_kinovea_v15.pkl')

ktauprefactors = [1]
kthetaprefactors = [1]

# ktauprefactors = [1, 2, 5, 10, 20, 50]
# kthetaprefactors = [0.1, 0.2, 0.5, 1, 2, 5]
ktauprefactors = [1, 5, 20, 50]
kthetaprefactors = [0.1, 1, 5]

vmax=1.1


overview_fig, overview_axs = plt.subplots(len(kthetaprefactors)+1,len(ktauprefactors)+1, figsize=[(len(ktauprefactors)+1)*3, (len(kthetaprefactors)+1)*3])
# overview_axs = overview_axs.flatten()
    

for ki, ktauprefactor in enumerate(ktauprefactors):
    for khi, kthetaprefactor in enumerate(kthetaprefactors):
        print(ki, khi)
        figname = r'./energy_sweep_ktautest_ktautest_{:.2f}_kthetatest_{:.2f}_overview'.format(ktauprefactor, kthetaprefactor)
        
        length_b = 24 #mm
        

        # kthetaprefactors=0.15
        datafolder = r'./energy_sweep_test15_ktauprefactor_{:.1f}_kthetaprefactor_{:.1f}/'.format(ktauprefactor, kthetaprefactor)
        # datafolder = r'./energy_sweep_test10_ktauprefactor_{:.1f}/'.format(ktauprefactor)
        datafolder = r'./energy_sweep_test13_ktauprefactor_{:.1f}_kthetaprefactor_{:.1f}/'.format(ktauprefactor, kthetaprefactor)
        # df = pd.read_csv(datafolder + r'model_v15_sweep_result_dict.csv')
        df = pd.read_csv(datafolder + r'model_v13_sweep_result_dict.csv')
        
        rc_layout()
        
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
            # energy_vals.append(row['max_rest_energy'])
        # sys.exit()
        # d_fab = []
        # ar_fab = []
        # g_fab = []
        py_exp = dict_fabrication['exp_max_deflections']
        lru_fab = dict_fabrication['exp_l_ru']
        gru_fab = dict_fabrication['exp_gamma_ru']
        markers_fab = dict_fabrication['marker']
        # lra_fab = []
        # py_exp = []
        # nar_fab = []
        # for ri, row in df_fabrication.iterrows():
        #     d_fab.append(row['d  [mm]'])
        #     ar_fab.append(row['S_ru/S_cell []'])
        #     g_fab.append(row['gamma [rad]'])
        #     gru_fab.append(row['gamma_RU [rad]'])
        #     lru_fab.append(row['length_RU_b [mm]'])
        #     lra_fab.append(row['length_RU_b [mm]']*np.cos(row['gamma_RU [rad]']))
        #     expval = 0.
        #     if ri in expdict.keys():
        #         expval = expdict[row['design']]['p_y_max [mm]']
        #     py_exp.append(expval)
        #     nar_fab.append(row['S_ru [mm^2]']/row['S_cell_reduced [mm^2]'])
        # sys.exit()
            
        
        # fig, axs = plt.subplots(1,10, figsize=[10*3, 1*3])
        
        # axs[0].set_xlabel(r'$\gamma$  [rad]')
        # axs[0].set_xlim(0, np.pi/2.)
        # axs[0].set_ylabel(r'$d  [mm]$')
        # ha1 = axs[0].scatter(gamma_values, d_values, c=py_values, cmap='YlGn', vmin=-2, vmax=15)
        # axs[1].axis('off')
        # plt.colorbar(ha1, ax=axs[1],  label='$p_y^{\mathrm{max}}$  [mm]')
        # axs[0].scatter(g_fab, d_fab, c=py_exp, zorder=1000, s=100, marker='*', alpha=1, cmap='YlGn', vmin=-2, vmax=15,
        #                edgecolor='k')
        
        # axs[2].set_xlabel(r'$\gamma$  [rad]')
        # axs[2].set_xlim(0, np.pi/2.)
        # axs[2].set_ylabel(r'$S_{RU}/S_{UC}$  []')
        # axs[2].set_ylim(0, 1)
        # ha2 = axs[2].scatter(gamma_values, arearatio_values, c=py_values, cmap='YlGn', vmin=-2, vmax=15)
        # axs[3].axis('off')
        # plt.colorbar(ha2, ax=axs[3],  label='$p_y^{\mathrm{max}}$  [mm]')
        # axs[2].scatter(g_fab, ar_fab, c=py_exp, zorder=1000, s=100, marker='*', alpha=1, cmap='YlGn', vmin=-2, vmax=15,
        #                edgecolor='k')
        
        # axs[4].set_xlabel(r'$\gamma$  [rad]')
        # axs[4].set_xlim(0, np.pi/2.)
        # axs[4].set_ylabel(r'$l_{RU}$  [mm]')
        # # axs[4].set_ylim(0, 1)
        # ha2 = axs[4].scatter(gamma_values, lengths_RU_b, c=py_values, cmap='YlGn', vmin=-2, vmax=15)
        # axs[5].axis('off')
        # plt.colorbar(ha2, ax=axs[5],  label='$p_y^{\mathrm{max}}$  [mm]')
        # axs[4].scatter(g_fab, lru_fab, c=py_exp, zorder=1000, s=100, marker='*', alpha=1, cmap='YlGn', vmin=-2, vmax=15,
        #                edgecolor='k')
        
        # axs[6].set_xlabel(r'$\gamma$  [rad]')
        # axs[6].set_xlim(0, np.pi/2.)
        # axs[6].set_ylabel(r'$l_{RU}$  [mm]')
        # # axs[4].set_ylim(0, 1)
        # ha2 = axs[6].scatter(gamma_values, lengths_RU_b, c=np.array(py_values)/(np.array(lengths_RU_b)*np.cos(np.array(gamma_values))), cmap='YlGn',vmin=-0.1, vmax=1.5)
        # axs[7].axis('off')
        # plt.colorbar(ha2, ax=axs[7],  label=r'${p}_y^{\mathrm{max}}/a_{RU}$  [mm]')
        # axs[6].scatter(g_fab, lru_fab, c=np.array(py_exp)/(np.array(lru_fab)*np.cos(np.array(g_fab))), zorder=1000, s=100, marker='*', alpha=1, cmap='YlGn',vmin=-0.1, vmax=1.5,
        #                edgecolor='k')
        
        # axs[8].set_xlabel(r'$\gamma_{\mathrm{RU}}$  [rad]')
        # axs[8].set_xlim(0, np.pi/2.)
        # axs[8].set_ylabel(r'$S_{RU}/S_{UC}$  []')
        # axs[8].set_ylim(0, 1)
        # ha2 = axs[8].scatter(gamma_RU_values, arearatio_values, c=py_values, cmap='YlGn', vmin=-2, vmax=15)
        # axs[9].axis('off')
        # plt.colorbar(ha2, ax=axs[9],  label='$p_y^{\mathrm{max}}$  [mm]')
        # axs[8].scatter(gru_fab, ar_fab, c=py_exp, zorder=1000, s=100, marker='*', alpha=1, cmap='YlGn', vmin=-2, vmax=15,
        #                edgecolor='k')
        
        # fig.savefig(datafolder + r'model_v5_sweep_results.pdf')
        # fig.savefig(datafolder + r'model_v5_sweep_results.png')
        # plt.close(fig)
        
        # overview_axs[ki][khi].set_title(r'$\tilde{K}_{\theta}$'+ r'={:.0f}'.format(kthetaprefactor))
        # overview_axs[ki][khi].set_xlabel(r'$\gamma$  [rad]')
        # overview_axs[ki][khi].set_xlim(0, np.pi/2.)
        # overview_axs[ki][khi].set_ylabel(r'$S_{RU}/S_{UC}$  []')
        # overview_axs[ki][khi].set_ylim(0, 1)
        # hao = overview_axs[ki][khi].scatter(gamma_values, arearatio_values, c=py_values, cmap='YlGn',vmin=-2, vmax=15)
        # overview_axs[ki][khi].scatter(g_fab, ar_fab, c=py_exp, zorder=1000, s=100, marker='*', alpha=1, cmap='YlGn', vmin=-2, vmax=15,
        #                edgecolor='k')
        
        # overview_axs[ki+len(ktauprefactors)+1].set_title(r'$\tilde{K}_{\theta}$'+ r'={:.0f}'.format(kthetaprefactor))
        # overview_axs[ki+len(ktauprefactors)+1].set_xlabel(r'$\gamma_{\mathrm{RU}}$  [rad]')
        # overview_axs[ki+len(ktauprefactors)+1].set_xlim(0, np.pi/2.)
        # overview_axs[ki+len(ktauprefactors)+1].set_ylabel(r'$l_{RU}/l_{RU}^{max}$  []')
        # overview_axs[ki+len(ktauprefactors)+1].set_ylim(0, 1)
        
        hao = overview_axs[ki][khi].scatter(gamma_RU_values, np.array(lengths_RU_b), c=np.array(py_values), 
                                            cmap='YlGn',vmin=-0.1, vmax=13, alpha=0.9)



        xi = np.linspace(np.min(gamma_RU_values), np.max(gamma_RU_values), 300)
        yi = np.linspace(np.min(lengths_RU_b), np.max(lengths_RU_b), 300)
        # zi_interp = griddata((np.array(gamma_RU_values), np.array(lengths_RU_b)), np.array(py_values), (xi[None, :], yi[:, None]), method='cubic', fill_value=0)
        # #smooth
        # # import scipy.ndimage
        # # zi_smooth = scipy.ndimage.zoom(zi, 3)
        import scipy.interpolate as sci
        smoothing = None
        tck = sci.bisplrep(np.array(gamma_RU_values), np.array(lengths_RU_b), np.array(py_values), s=smoothing, kx=5, ky=5) 
        zi_interp = sci.bisplev(xi, yi, tck).T
        
        # sortidxru = np.argsort(np.array(gamma_RU_values).flatten())
        # zi_check = sci.bisplev(np.array(gamma_RU_values)[sortidxru].flatten(), np.array(lengths_RU_b)[sortidxru], tck)
        zi_check = [sci.bisplev(np.array(gamma_RU_values)[i], np.array(lengths_RU_b)[i], tck) for i in range(len(gamma_RU_values))]
        zi_check = np.array(zi_check)
        zi_diff = zi_check - np.array(py_values)
        print('contour deviation', zi_diff.min(), zi_diff.max())
        
        #mask values outside of convex hull of g_ru, l_ru range
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
        hull = np.vstack([np.array(gamma_RU_values), np.array(lengths_RU_b)]).T
        mask = np.zeros_like(zi_interp)
        for i in range(len(yi)):
            mypts = np.array([xi, np.ones_like(xi)*yi[i]]).T
            mask[i] = in_hull(mypts, hull)
            # sys.exit()
            
        plt.imshow(mask)
        # sys.exit()
        zi_interp[np.where( (mask==0) | (zi_interp<1e-3) )] = np.nan
        
        # hao = overview_axs[ki][khi].tricontour(gamma_RU_values, np.array(lengths_RU_b), np.array(py_values), 
        #                                     cmap='YlGn',vmin=-0.1, vmax=13, levels=20)        
     

        # haoc = overview_axs[ki][khi].contour(xi, yi, zi_interp, 
        #                                     cmap='YlGn',vmin=-0.1, vmax=13, levels=20, 
        #                                     alpha=1, zorder=1000, lw=2)      

        

 
        if False:
            hao = overview_axs[ki][khi].scatter(gru_fab[markers_fab=='d'], lru_fab[markers_fab=='d'], c=py_exp[markers_fab=='d'], 
                                                cmap='YlGn', marker='d', vmin=-0.1, vmax=13,
                                                zorder=1000, edgecolor='k')
            hao = overview_axs[ki][khi].scatter(gru_fab[markers_fab=='v'], lru_fab[markers_fab=='v'], c=py_exp[markers_fab=='v'], 
                                                cmap='YlGn', marker='v', vmin=-0.1, vmax=13,
                                                zorder=1000, edgecolor='k')
            print(np.max(py_values))
        # overview_axs[ki+len(ktauprefactors)+1].scatter(gru_fab, np.array(lru_fab)/length_b, c=np.array(py_exp)/np.array(lra_fab), zorder=1000, s=100, marker='*', alpha=1, cmap='YlGn', vmin=-0.1, vmax=1.1,
        #                edgecolor='k')
        
        overview_axs[ki][khi].set_title(r'$\tilde{K}_{\theta}$'+ r'={:.1f}'.format(kthetaprefactor) + r', $\tilde{K}_{\tau}$'+ r'={:.0f} '.format( ktauprefactor))
        overview_axs[ki][khi].set_xlabel(r'$\gamma_{\mathrm{RU}}$  [rad]')
        overview_axs[ki][khi].set_xlim(0, np.pi/2.)
        overview_axs[ki][khi].set_ylabel(r'$l_{RU}$  [mm]')
        overview_axs[ki][khi].set_ylim(0, 25)
        # ha2 = overview_axs[ki][khi].scatter(gamma_RU_values, new_arearatio_values, c=np.array(nums_energy_minima), vmin=0, vmax=2)
        # ha2 = overview_axs[ki][khi].scatter(gamma_RU_values, np.array(lengths_RU_b)/length_b, c=np.array(nums_energy_minima), vmin=1, vmax=2)
        alpha_vals = np.array(nums_energy_minima) < 1.5
        alpha_vals = 0.1*alpha_vals.astype('float')
        ha2 = overview_axs[ki][khi].scatter(gamma_RU_values, np.array(lengths_RU_b), c='k',
                                                                  alpha = alpha_vals, zorder=1000)
       
        
        
        # ha2 = overview_axs[ki][khi].scatter(gamma_RU_values, np.array(lengths_RU_b)/length_b, c='k',
        #                                                           alpha = alpha_vals)
        # ha2 = overview_axs[ki][khi].scatter(gamma_RU_values, np.array(lengths_RU_b)/length_b, c=np.array(energy_vals), vmax=1e-3)
        # ha2 = overview_axs[ki][khi].tricontourf(gamma_RU_values, np.array(lengths_RU_b)/length_b, np.array(energy_vals), levels = [-0.5, 1e-5])
        # ha2 = overview_axs[ki][khi].tricontourf(gamma_RU_values, np.array(lengths_RU_b)/length_b, nums_energy_minima, levels = [-0.5, 1.5])
        # overview_axs[ki][khi].scatter(gru_fab, np.array(lru_fab)/length_b, c=np.array(py_exp)/np.array(lra_fab), zorder=1000, s=100, marker='*', alpha=1, cmap='YlGn', vmin=-0.1, vmax=1.1,
        #                edgecolor='k')
        # sys.exit()
        
        
overview_axs[-1][-1].axis('off')
cbar = plt.colorbar(hao, ax=overview_axs[-1][-1],  label=r'$max({p}^{eq}_{y})$  []')
    # cbar = plt.colorbar(ha2, ax=overview_axs[-2],  label=r'N')
overview_fig.savefig(figname+'.pdf')
overview_fig.savefig(figname+'.png')

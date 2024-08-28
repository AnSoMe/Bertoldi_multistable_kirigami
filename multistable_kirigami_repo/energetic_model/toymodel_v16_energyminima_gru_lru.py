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
from scipy.signal import argrelmin

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
        vec_r_2_reduced = j_1 - h_2
        pt_tri_2 = find_crossing_point(j_1, vec_r_2_reduced, p_beta, vec_alpha_beta)
        pt_tri_3 = find_crossing_point(j_1, vec_r_2_reduced, e_1, vec_r_1_reduced)
        rotating_area = la.norm(np.cross(pt_tri_1-pt_tri_3, pt_tri_2-pt_tri_3))
        ru_pts = np.vstack([pt_tri_1, pt_tri_2, pt_tri_3, pt_tri_1])
        
        # print('hey', e_1, pt_tri_1)
        # sys.exit()
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
#User variables. All units in N, mm, rad.

rc_layout()

#material parameters
youngsmod = 10 #Young's modulus, MPa or N/(mm)^2
poisson = 0.5 
sheet_t = 2.38125 #mm, 3/32 in

#cell parameters
length_b = 24 #truss spring length, mm
l_laser = 0.7 # laser cut width, mm
hinge_size = 0.7 #hinge width, mm

#parameters to sweep
# min_gamma = np.pi/20.
# max_gamma = 9*np.pi/20.

min_gamma= 1e-3
max_gamma = np.pi/2.-1e-3

# num_gamma_vals = 20

# min_d = hinge_size #max_d will be calculated automatically.
min_d = 1e-3
# num_d_vals = 205

#fine-ness of energy minimization grid
num_pz_vals = 101
num_py_vals = 103

#indicate special points (match to fabricated samples)
# gamma_vals_todraw = [np.pi/10, np.pi/4, np.pi/3, 4*np.pi/10.]
# b_vals_todraw = [24]
max_plot_energy = 100 #mNm
deflection_transition_limit = 0.1 #mm, anything below this is not considered a transition

ktauprefactors = [1, 2, 5, 10, 20, 50]
kthetaprefactors = [0.1, 0.2, 0.5, 1, 2, 5]

ktauprefactors = [1.5]
kthetaprefactors = [1]

num_gamma_vals = 50
num_d_vals = 50

for ktauprefactor in ktauprefactors:
    for kthetaprefactor in kthetaprefactors:
        datafolder = r'./energy_sweep_test16_ktauprefactor_{:.1f}_kthetaprefactor_{:.1f}/'.format(ktauprefactor, kthetaprefactor)
        print(datafolder)
        
        ###############################################################################
        #Loop over gamma, d. For each pair, calculate the minimal-energy config E_min, p_y_min
        #for fixed p_z. 
         # b
        if not os.path.exists(datafolder):
            os.mkdir(datafolder)
            
        model_v5_sweep_result_dict = {}
        ctr = 0
        
        gamma_vals = np.linspace(min_gamma, max_gamma, num_gamma_vals)
        for gi, gamma in enumerate(gamma_vals):
            
            d_vals = np.linspace(min_d, get_max_d(b=length_b, gamma=gamma, h=hinge_size)+hinge_size, num_d_vals)
            for di, length_d in enumerate(d_vals):
                
                #######################################################################
                #calculate cell geometry.
                #assume a right triangle for simplicity
                length_a = length_b*np.cos(gamma)
                length_c = length_b*np.sin(gamma)
                alpha = np.pi/2.-gamma
                success, mydict, design_points, RU_points = get_cell_geometry(length_a, length_b, gamma, hinge_size, length_d)
                if not success: continue
            
                #[p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2] = design_points
                model_offset = np.array([0, design_points[-2][-1]]) #move designs along z by j_1 to ensure model is correct (pt_truss in z=0 plane)
                design_points = [dp - model_offset for dp in design_points]
                RU_points = np.array([dp - model_offset for dp in RU_points])
                
                #optional: draw cell geometry
                # draw_lasercut_fabrication_points(*triangle_points, ax=ax, draw_offset=draw_offset) 
                # ax.fill(RU_points[:, 0]+draw_offset[0], RU_points[:, 1]+draw_offset[1], color='yellow', alpha=0.3)
                
                #######################################################################
                #calculate corresponding spring model geometry.
                [p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2] = design_points
                pt_truss = j_1
                pt_strut = j_2
                length_RU_b = la.norm(pt_truss-pt_strut) #estimate, may be refined if needed (see design drawing)
                RU_unitvec = (pt_truss-pt_strut)/length_RU_b
                gamma_RU = np.arccos(RU_unitvec[1])
                alpha_RU_adj = np.pi/2-gamma_RU
                length_RU_c = length_RU_b*np.sin(gamma_RU)
                length_RU_a = length_RU_b*np.cos(gamma_RU)
                RU_h_max = length_RU_b *np.cos(gamma_RU)*np.sin(gamma_RU) #max height of right triangle 
                
                
                #optional: draw model geometry
                # ax.plot([pt_truss[0], pt_strut[0]], [pt_truss[1], pt_strut[1]], c='k')
                # ax.scatter(*pt_truss, c='k')
                # ax.scatter(*pt_strut, c='k')
                
                #######################################################################'
                #calculate spring stiffnesses given model geometry.
                #stiffness values
                k_RU = youngsmod* 0.5*(hinge_size + RU_h_max)*sheet_t / length_RU_b
                k_theta = youngsmod/12. * (hinge_size**3 * sheet_t)/l_laser*kthetaprefactor
                k_tau = youngsmod/(l_laser)* get_torsion_constant(hinge_size, sheet_t)*ktauprefactor
                
                #######################################################################
                #Calculate energy of possible configurations.
                pz_vals = np.linspace(pt_strut[-1]*1.2, -pt_strut[-1]*1.2, num_pz_vals)
                py_vals = np.linspace(pt_strut[-1]*1.2, -pt_strut[-1]*1.2, num_py_vals)  
                py_grid, pz_grid = np.meshgrid(py_vals, pz_vals)
                energy_grid = np.zeros_like(py_grid)
                for (i,j), e in np.ndenumerate(energy_grid):
                    py = py_grid[i][j]
                    pz = pz_grid[i][j]
                    args = [length_RU_b, length_RU_c, alpha_RU_adj, pz, k_RU, k_theta, k_tau]
                    energy_grid[i][j] = Mises_lagrangian(py, args)
                
                #######################################################################
                #Calculate min-energy configurations.
                min_energy_py_vals = np.zeros_like(pz_vals)
                min_energy_vals = np.zeros_like(pz_vals)
                energies_at_zero_deflection = np.zeros_like(pz_vals)
                for ei, energy_vals in enumerate(energy_grid):
                    nodeflection_idx = np.argmin(np.abs(py_vals))
                    minidx = np.argmin(energy_vals)
                    min_energy_py_vals[ei] = py_grid[ei][minidx]
                    min_energy_vals[ei] = energy_vals[minidx]
                    energies_at_zero_deflection[ei] = energy_vals[nodeflection_idx]
                maximal_deflection_py = np.max(np.abs(min_energy_py_vals))
                
                ##################################################################
                #calculate energy at max deflection, and what corresponding elastic energy at zero deflection would be
                maximal_deflection_idx = np.argmax(np.abs(min_energy_py_vals))
                energy_at_max_deflection = min_energy_vals[maximal_deflection_idx]
                energy_at_confined_max_deflection = energies_at_zero_deflection[maximal_deflection_idx]
                
                ###############################################################
                #calculate transition z-value into deflection 
                transition_pz = None
                if maximal_deflection_py > deflection_transition_limit:
                    transition_index = np.where(np.abs(min_energy_py_vals)> deflection_transition_limit)[0][0]
                    transition_pz = pz_vals[transition_index]
                    
                #######################################################################
                #Calculate number of minima.
                energy_minima_idx = argrelmin(min_energy_vals)                   
                num_energy_minima = len(argrelmin(min_energy_vals)[0])   
                max_rest_energy = 0.
                if num_energy_minima > 1:
                    max_rest_energy = np.max(min_energy_vals[energy_minima_idx])
     
                #######################################################################
                #Store results needed for later analysis.
                mydict.update({'p_z_crit [mm]': transition_pz, 
                                'p_y_max [mm]':  maximal_deflection_py,
                               'length_RU_b [mm]': length_RU_b,
                               'energy_at_max_deflection [mNm]': energy_at_max_deflection,
                               'energy_at_confined_max_deflection [mNm]': energy_at_confined_max_deflection,
                               # 'length_RU_a [mm]': length_RU_a,
                               'k_RU [N/mm]': k_RU,
                               'k_theta [mNm]': k_theta,
                               'k_tau [mNm]': k_tau,
                               'gamma_RU [rad]': gamma_RU,
                               'num_energy_minima': num_energy_minima,
                               'max_rest_energy': max_rest_energy
                               })
                model_v5_sweep_result_dict.update({ctr: mydict})
                ctr += 1
                
                #######################################################################
                #visualize result.
                
                fig, axs = plt.subplots(1,5, figsize=[5*3, 1*3])
                
                #draw cell and "truss"
                draw_offset = np.array([0.,0.])
                draw_lasercut_fabrication_points(*design_points, ax=axs[0], draw_offset=draw_offset) #, c='orange') 
                axs[0].fill(RU_points[:, 0]+draw_offset[0], RU_points[:, 1]+draw_offset[1], color=[.7,.5,.2], alpha=0.3)
                axs[0].set_xlim(-length_b*1.25, length_b*0.25)
                axs[0].set_ylim(-length_b*1.25, length_b*0.25)
                axs[0].plot([pt_truss[0], pt_strut[0]], [pt_truss[1], pt_strut[1]], c='k', zorder=1000)
                axs[0].scatter(*pt_truss, c='k', zorder=1000)
                axs[0].scatter(*pt_strut, c='k', zorder=1000)
                axs[0].set_xlabel(r'x  [mm]')
                axs[0].set_ylabel(r'z  [mm]')
                
                #draw energy landscape
                ha = axs[1].imshow(energy_grid, origin='lower', extent=[py_vals[0], py_vals[-1], pz_vals[0], pz_vals[-1]],
                                cmap='Greys_r')
                axs[1].contour(py_grid, pz_grid, energy_grid, cmap='OrRd_r', levels=5) #, levels=[1e-5, 1e-3, 1e-1, 1e1])
                axs[2].axis('off')
                plt.colorbar(ha, ax=axs[2],  label='$\mathcal{E}$  [mNm]')
                axs[1].set_xlabel(r'$p_y$  [mm]')
                axs[1].set_ylabel(r'$p_z$  [mm]')
                axs[1].set_xlim(-length_b, length_b)
                axs[1].set_ylim(-length_b, length_b)
                axs[1].scatter(min_energy_py_vals, pz_vals, c='r', s=1)
                axs[1].scatter(-min_energy_py_vals, pz_vals, c='r', s=1)
                
                #draw min-energy configurations and energy
                axs[3].plot(pz_vals, np.abs(min_energy_py_vals), color='k', label='$\gamma_{\mathrm{RU}}$' + r'={:.1f}$\pi, d={:.2f}mm$'.format(gamma, length_d))
                axs[3].set_xlabel(r'$p_z$  [mm]')
                axs[3].set_ylabel(r'$p_y$  [mm]')
                axs[3].set_ylim(-length_b*0.25, 1.75*length_b)
                axs[3].set_xlim(-length_b, length_b)
                if transition_pz is not None:
                    axs[3].plot([transition_pz, transition_pz], [-length_b*0.25, 1.75*length_b], color=[.7,.7,.7], linestyle=':', dash_capstyle='round')
                axs[3].plot([-length_b, length_b], [length_RU_a,length_RU_a], color=[.7,.7,.7], linestyle=':', dash_capstyle='round')
                
                axs[4].plot(pz_vals, min_energy_vals, color='k', label='$\gamma$={:.1f}$\pi$'.format(gamma))
                axs[4].set_xlabel(r'$p_z$  [mm]')
                axs[4].set_ylabel(r'$\mathcal{E}$  [mNm]')
                axs[4].set_xlim(-length_b, length_b)
                axs[4].set_ylim(0, max_plot_energy)
                axs[4].scatter(pz_vals[energy_minima_idx], min_energy_vals[energy_minima_idx], c='k', marker='*', s=100)
                
                # sys.exit()
                fig.savefig(datafolder + r'design_{:d}.pdf'.format(ctr))
                fig.savefig(datafolder + r'design_{:d}.png'.format(ctr))
        
                plt.close(fig)
        
        
        df = pd.DataFrame()
        df = df.from_dict(model_v5_sweep_result_dict, orient='index')
        df.to_csv(datafolder + r'model_v15_sweep_result_dict.csv')
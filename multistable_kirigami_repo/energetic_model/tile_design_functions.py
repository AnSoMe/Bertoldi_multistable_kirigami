import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import matplotlib.lines as mlines
from itertools import product
import scipy.linalg as la
from scipy.optimize import minimize


###############################################################################
###############################################################################
#Functions for geometric calculations
###############################################################################
###############################################################################

def expansion_factor(lt, a, deltat):
    return 1 + 2 *lt/a*np.sin(deltat)

def expansion_factor_lowest_frompoints(a, j_1, j_2):
    dx, dy = j_2-j_1
    deltat_adjusted = np.abs(np.arctan(dy/dx))
    lt_adjusted = la.norm(j_1-j_2)
    s_adjusted = expansion_factor(lt_adjusted, a, deltat_adjusted)
    return s_adjusted

def expansion_factor_lowest_with_hinges(a, b, gamma, s,
                  ch, deltat,
                  d1, d2,
                  h,
                  adjust_h=False):
    
    p_beta = np.array([0, a])
    p_alpha = b*np.array([-np.sin(gamma), np.cos(gamma)])
    c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(gamma))
    uvec_c = 1/c*(p_alpha - p_beta)
    
    #calculate edge match point positions e_1, e_2.
    e_1 = p_beta + d1*uvec_c
    
    #calculate hinge point positions h_1, h_2.
    lt = a*(s-1)/(2*np.sin(deltat))
    uvec_l = np.array([np.cos(deltat), -np.sin(deltat)])
    h_1 = p_beta + ch*uvec_c
    h_2 = p_beta + ch*uvec_c+ lt*uvec_l
    
    #calculate hinge cut point positions j_1, j_2.
    sin_zeta = np.cross(e_1-h_2, h_1-h_2) / (lt*la.norm(e_1-h_2))
    # print(h, s, sin_zeta)
    if adjust_h:
        alpha = np.arccos((c**2 + b**2 - a**2)/(2*c*b))#cosine law
        j_1 = h_1 + h/(-np.cos(alpha + gamma + deltat))*uvec_l
        j_2 = h_2 + h/sin_zeta * (e_1 - h_2)/la.norm(e_1-h_2)
    else:
        j_1 = h_1 + h*uvec_l
        j_2 = h_2 + h* (e_1 - h_2)/la.norm(e_1-h_2) 
    
    lt_adjusted = la.norm(j_1-j_2)
    dx, dy = j_2-j_1
    deltat_adjusted = np.abs(np.arctan(dy/dx))
    print('target angle, actual angle', deltat, deltat_adjusted)
    s_adjusted = expansion_factor(lt_adjusted, a, deltat_adjusted)
    
    return s_adjusted

def triangle_interior_check(v, p_alpha, p_beta, p_gamma):
    D1 = (np.cross(v, p_alpha) - np.cross(p_gamma, p_alpha))/(np.cross(p_beta, p_alpha))
    D2 = -(np.cross(v, p_beta) - np.cross(p_gamma, p_beta))/(np.cross(p_beta, p_alpha))
    Dtot = D1 + D2

    return D1, D2, Dtot

def find_crossing_point(point_1, tangent_1, point_2, tangent_2):
    '''
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    '''
    x1, y1 = point_1
    x2, y2 = point_1 + tangent_1
    x3, y3 = point_2
    x4, y4 = point_2 + tangent_2
    cross_x = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)) 
    cross_y = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    crossing_point = np.array([cross_x, cross_y])
    return crossing_point

def mindist_point_to_line(point, linepoints):
    '''
    calculate minimal distance from a point to a line between to points.
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    '''
    x3, y3 = point
    x1, y1 = linepoints[0]
    x2, y2 = linepoints[1]
    px = x2-x1
    py = y2-y1
    
    norm = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    
    x = x1 + u * px
    y = y1 + u * py
    
    dx = x - x3
    dy = y - y3
    
    dist = (dx*dx + dy*dy)**.5

    return dist

def mirror_points(points, mirror_point, mirror_angle):
    mirmat = np.array([[np.cos(2*mirror_angle),np.sin(2*mirror_angle)],[np.sin(2*mirror_angle),-np.cos(2*mirror_angle)]])
    new_points = np.array([np.dot(mirmat, point-mirror_point) + mirror_point for point in points])
    return new_points

def transform_points(points, translation_vec, rotation_point, rotation_angle):
    
    rotmat = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                       [np.sin(rotation_angle), np.cos(rotation_angle)]])
    new_points = np.array([np.dot(rotmat, point - rotation_point) + rotation_point + translation_vec for point in points])
    
    return new_points

###############################################################################
###############################################################################
#Drawing and plotting functions
###############################################################################
###############################################################################

def draw_lasercut_fabrication_points(p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2,
                         ax=None, draw_offset=np.array([0,0.]), cut_color=[.7,.5,.2],
                         outline_color=[.7,.7,.7]):
    
    #draw result
    if ax is None:
        fig, ax = plt.subplots(1,1)
        ax.axis('equal')
    
    #draw triangle cell outline
    corner_points = np.array([p_gamma, p_beta, p_alpha, p_gamma]) + draw_offset
    # ax.scatter(corner_points.T[0], corner_points.T[1],
    #            label='cell', zorder=2000)
    ax.plot(corner_points.T[0], corner_points.T[1], 
        linestyle=':', dash_capstyle='round',
        c=outline_color)
    
    #draw cut lines
    cut1_points = np.array([p_alpha, e_1]) + draw_offset
    cut2_points = np.array([e_1, j_2]) + draw_offset
    cut3_points = np.array([j_1, h_2]) + draw_offset
    cut4_points = np.array([h_2, e_2]) + draw_offset
    ax.plot(cut1_points.T[0], cut1_points.T[1], 
        linestyle='-', dash_capstyle='round',
        c=cut_color, zorder=1000, lw = 1)
    ax.plot(cut2_points.T[0], cut2_points.T[1], 
        linestyle='-', dash_capstyle='round',
        c=cut_color, zorder=1000, lw=1)
    ax.plot(cut3_points.T[0], cut3_points.T[1], 
        linestyle='-', dash_capstyle='round',
        c=cut_color, zorder=1000, lw=1)
    ax.plot(cut4_points.T[0], cut4_points.T[1], 
        linestyle='-', dash_capstyle='round',
        c=cut_color, zorder=1000, lw=1)
    
    return ax.get_figure()

def draw_triangle_points(p_gamma, p_beta, p_alpha, e_1, e_2, h_1, h_2, j_1, j_2,
                         ax=None, draw_offset=np.array([0,0.])):
    
    #draw result
    if ax is None:
        fig, ax = plt.subplots(1,1)
        ax.axis('equal')
    
    #draw triangle cell outline
    corner_points = np.array([p_gamma, p_beta, p_alpha, p_gamma]) + draw_offset
    ax.scatter(corner_points.T[0], corner_points.T[1],
               label='cell', zorder=2000)
    ax.plot(corner_points.T[0], corner_points.T[1], 
        linestyle=':', dash_capstyle='round',
        c=[.7,.7,.7])
    
    #draw triangular rotating unit outline
    triangle_points = np.array([h_1, e_1, h_2, h_1]) + draw_offset
    ax.scatter(triangle_points.T[0], triangle_points.T[1],
               label='triangle', zorder=2000) 
    ax.plot(triangle_points.T[0], triangle_points.T[1], 
        linestyle='-', dash_capstyle='round',
        c=[.6,.6,.76])
    
    #draw pusher strut outline
    pusher_points = np.array([e_1, h_2, e_2]) + draw_offset
    ax.plot(pusher_points.T[0], pusher_points.T[1], 
        linestyle='-', dash_capstyle='round',
        c=[.5,.5,.5])
    
    #draw truss strus outline
    strut_points = np.array([p_alpha, h_1, h_2, e_2]) + draw_offset
    ax.plot(strut_points.T[0], strut_points.T[1], 
        linestyle='-', dash_capstyle='round',
        c=[.4,.4,.4])
    
    #draw cut lines
    cut1_points = np.array([p_alpha, e_1, j_2]) + draw_offset
    cut2_points = np.array([j_1, h_2, e_2]) + draw_offset
    ax.plot(cut1_points.T[0], cut1_points.T[1], 
        linestyle='-', dash_capstyle='round',
        c=[.7,.5,.2], zorder=1000, lw = 4, label = 'cut line')
    ax.plot(cut2_points.T[0], cut2_points.T[1], 
        linestyle='-', dash_capstyle='round',
        c=[.7,.5,.2], zorder=1000, lw=4)
    
    #draw joins
    join_points = np.array([j_1, j_2]) + draw_offset
    ax.scatter(join_points.T[0], join_points.T[1],
               label='joins', zorder=2000) 
    
    return ax.get_figure()

###############################################################################
###############################################################################
#tile design with parallel edges
###############################################################################
###############################################################################

def generate_parallel_tile(a, b, gamma, h, e, vocal=False):
    
    success = True
    
    #calculate triangular unit cell points, side vectors, and side unit vectors.
    p_gamma = np.array([0, 0.])
    p_beta = np.array([0, a])
    p_alpha = b*np.array([-np.sin(gamma), np.cos(gamma)])
    c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(gamma))
    uvec_c = 1/c*(p_alpha - p_beta)
    uvec_b = np.array([-np.sin(gamma), np.cos(gamma)])
    uvec_a = np.array([0,1.])
    uvec_c_ortho = np.array([-uvec_c[1], uvec_c[0]])
    uvec_b_ortho = np.array([uvec_b[1], -uvec_b[0]]) #all ortho vecs point inward
    uvec_a_ortho = np.array([-uvec_a[1], uvec_a[0]])
    beta = np.arccos((b**2 - a**2 - c**2)/(-2*a*c))
    alpha = np.pi-gamma-beta
    
    #calculate edge match point positions e_1, e_2.
    d1 = e/np.sin(beta)
    d2 = e/np.sin(gamma)
    e_2 = d2*uvec_a
    e_1 = p_beta + d1*uvec_c
    # print(e_1, e_2)
    
    #calculate hinge point positions j1, j2
    j_1 = p_alpha - (e/np.sin(alpha))*uvec_c - (h/np.sin(alpha))*uvec_b
    j_2 = find_crossing_point(j_1+h*uvec_b_ortho, e_2-j_1, e_1, uvec_a)
    
    # j_2 = p_gamma + (e+h)/np.sin(gamma)*uvec_a + (e/np.sin(gamma))*uvec_b
    # j_2 = p_gamma + (e+h)/np.sin(gamma)*uvec_a + (e+h)*np.tan(gamma)*uvec_b
    
    
    # j_2 = p_beta + (c - (e+h)/np.sin(alpha))*uvec_c - b/c*(c - (e + h)/np.sin(alpha) - e/np.sin(beta))*uvec_b
    # print( (c - (e+h)/np.sin(alpha)), b/c*(c - (e + h)/np.sin(alpha) - e/np.sin(beta)))
    
    #calculate point h2 on line j1-e2
    h_2 = j_2 - h*uvec_b_ortho
    
    #check point locations
    ptnames = ['e_1', 'e_2', 'h_2', 'j_1', 'j_2']
    for pi, pt in enumerate([e_1, e_2, h_2, j_1, j_2]):
        D1, D2, Dtot = triangle_interior_check(pt, p_alpha, p_beta, p_gamma)
        if D1 < 0 or D2 < 0 or Dtot >1:
            success = False
            if vocal:
                print('requested hinge angle not compatible for {:s} Try again.'.format(ptnames[pi]))
    if mindist_point_to_line(point=j_2, linepoints=[p_beta, p_alpha]) < h:
        success = False
        if vocal:
            print(r'j_2 too close to top cell edge.')
    if mindist_point_to_line(point=j_2, linepoints=[p_beta, p_gamma]) < h:
        success = False
        if vocal:
            print(r'j_2 too close to right cell edge.')
    
    #done
    triangle_points = np.array([p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2])
    
    return success, triangle_points

###############################################################################
###############################################################################
#tile design with fabrication expansion factor in mind 
###############################################################################
###############################################################################
                  
def generate_fabrication_tile(a, b, gamma, sprime,
                              ch, deltaprime,
                             d1, d2,
                             h,
                             vocal=False):
    #draw tile design based on provided design parameters.
    #triangular unit cell set by a, b, gamma.
    #cell expansion ratio set by s.
    #cell hinge designed via ch, deltat.
    #cell joins designed via d1, d2.
    #hinge length designed via h.
    
    success= True
    
    #check if ch is larger than d_1, or d_2 larger than a.
    if d1 > ch: 
        success = False
        print('requested d_1 bigger than c_h. Try again.')
    if d2 > a: 
        success = False
        print('requested d_2 bigger than a. Try again.')
    #check if gamma is over np.pi2.
    if gamma >= np.pi/2:
        success = False
        print('tile with requested base angle cannot expand. Try again.')
    
    #calculate triangular unit cell points, side vectors, and side unit vectors.
    p_gamma = np.array([0, 0.])
    p_beta = np.array([0, a])
    p_alpha = b*np.array([-np.sin(gamma), np.cos(gamma)])
    c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(gamma))
    uvec_c = 1/c*(p_alpha - p_beta)
    uvec_b = np.array([-np.sin(gamma), np.cos(gamma)])
    uvec_a = np.array([0,1.])

    #calculate edge match point positions e_1, e_2.
    e_2 = d2*uvec_a
    e_1 = p_beta + d1*uvec_c
    
    #calculate hinge point positions j1, j2
    lprime = a*(sprime-1)/(2*np.sin(deltaprime))
    uvec_lprime = np.array([np.cos(deltaprime), -np.sin(deltaprime)])
    uvec_c_ortho = np.array([-uvec_c[1], uvec_c[0]])
    j_1 = p_beta + ch*uvec_c + h*uvec_c_ortho
    j_2 = j_1 + lprime*uvec_lprime
    
    
    #check whether j1, j2 lie within triangle cell
    D1, D2, Dtot = triangle_interior_check(j_1, p_alpha, p_beta, p_gamma)
    if D1 < 0 or D2 < 0 or Dtot >1:
        success = False
        if vocal:
            print('requested hinge angle not compatible for j_1. Try again.')
    D1, D2, Dtot = triangle_interior_check(j_2, p_alpha, p_beta, p_gamma)
    if D1 <= 0 or D2 <= 0 or Dtot >=1:
        success = False
        if vocal:
            print('requested hinge angle not compatible for j_2. Try again.')

    #calculate cut design location h2 so that j2 lies at distance
    #h from line j1-h2. There is a DOF we eliminate here
    #so that h2 lies at the shortest distance point.
    l = np.sqrt(lprime**2 - h**2)
    delta = np.arcsin(h/lprime)
    uvec_l = np.array([np.cos(delta+deltaprime), -np.sin(delta+deltaprime)])
    h_2 = j_1 + l*uvec_l
    #alternative
    # h_2 = j_1 + (l+h)*uvec_l
    

    #check whether h2 lies within requested triangle cell
    #check h1
    D1, D2, Dtot = triangle_interior_check(h_2, p_alpha, p_beta, p_gamma)
    if D1 < 0 or D2 < 0 or Dtot >1:
        success = False
        if vocal:
            print('requested hinge angle not compatible for h_2. Try again.')

    triangle_points = np.array([p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2])
    
    return success, triangle_points

def cost_function_fabrication_pairwise_v5(opt_args, args):
    
    ch, deltaprime, ch2, deltaprime2 = opt_args
    a, b, gamma, sprime, d1, d2, h, vocal, b2, gamma2, sprime2, Dweight, Aweight, Bweight, Cweight, return_best_d2 = args
    success_1, success_2 = [True, True] #flag for valid designs
    
    ###########################################################################
    #get unit cell 1 points and cost function
    success_1, triangle_points_1 = generate_fabrication_tile(a=a, b=b, gamma=gamma, 
                                                           sprime=sprime, ch=ch,
                                                           deltaprime=deltaprime, 
                                                           d1=d1, d2=d2, h=h)
    p_gamma, p_beta, p_alpha, e_1, e_2, h_2, j_1, j_2 = triangle_points_1
    #check whether j2 lies within requested triangle cell
    D1, D2, Dtot = triangle_interior_check(j_2, p_alpha, p_beta, p_gamma)
    D_cost_value = 0
    if D1 <= 0: D_cost_value += -D1
    if D2 <= 0: D_cost_value += -D2
    if Dtot >= 1: D_cost_value += (Dtot-1)
    #check whether j1 lies within requested triangle cell
    D1, D2, Dtot = triangle_interior_check(j_1, p_alpha, p_beta, p_gamma)
    if D1 <= 0: D_cost_value += -D1
    if D2 <= 0: D_cost_value += -D2
    if Dtot >= 1: D_cost_value += (Dtot-1)
    #check whether h2 lies within requested triangle cell
    D1, D2, Dtot = triangle_interior_check(h_2, p_alpha, p_beta, p_gamma)
    if D1 <= 0: D_cost_value += -D1
    if D2 <= 0: D_cost_value += -D2
    if Dtot >= 1: D_cost_value += (Dtot-1)
    
    #calculate joint positions relative to cell edges
    dist_a_j2 = mindist_point_to_line(point=j_2, linepoints = [p_gamma, p_beta])
    opt_dist_a_j2 = 1*h
    
    dist_b_j1 = mindist_point_to_line(point=j_1, linepoints = [p_gamma, p_alpha])
    opt_dist_b_j1 = 1*h
    c = la.norm(p_beta-p_alpha)
    dist_b_j1 = la.norm(j_1- p_alpha)
    opt_dist_b_j1 =(c-d1)/2.
    # if vocal: print('dist',dist_b_j1, 'opt', opt_dist_b_j1)
    b_j1_cost = (dist_b_j1-opt_dist_b_j1)**2
    min_dist_b_j1 = d1+h
    max_dist_b_j1 = la.norm(p_beta-p_alpha)-h
    b_j1_cost = 1/(ch-min_dist_b_j1)**2+ 1/(ch-max_dist_b_j1)**12 #favour max dist
    # b_j1_cost = 1/(ch-min_dist_b_j1)**2 #favour max dist
    
    cost_1 = Dweight*D_cost_value + Aweight*(dist_a_j2-opt_dist_a_j2)**2 + Bweight*b_j1_cost
    if vocal: print('\t dist 1',dist_b_j1, 'limits',min_dist_b_j1, max_dist_b_j1, 'cost', b_j1_cost)
    
    ###########################################################################
    #get unit cell 2 points and cost function
    success_2, triangle_points_2 = generate_fabrication_tile(a=a, b=b2, gamma=gamma2, 
                                                           sprime=sprime2, ch=ch2,
                                                           deltaprime=deltaprime2, 
                                                           d1=d1, d2=d2, h=h)
    p_gamma_2, p_beta_2, p_alpha_2, e_1_2, e_2_2, h_2_2, j_1_2, j_2_2 = triangle_points_2
    # j_2_2_mirrored = mirror_points(j_2_2, p_gamma_2, np.pi/2.)
    j_2_2_mirrored = np.array([-j_2_2[0], j_2_2[1]])
    #check whether j2 lies within requested triangle cell
    D1, D2, Dtot = triangle_interior_check(j_2_2, p_alpha_2, p_beta_2, p_gamma_2)
    D_cost_value_2 = 0
    if D1 <= 0: D_cost_value_2 += -D1
    if D2 <= 0: D_cost_value_2 += -D2
    if Dtot >= 1: D_cost_value_2 += (Dtot-1)
    #check whether j1 lies within requested triangle cell
    D1, D2, Dtot = triangle_interior_check(j_1_2, p_alpha_2, p_beta_2, p_gamma_2)
    if D1 <= 0: D_cost_value_2 += -D1
    if D2 <= 0: D_cost_value_2 += -D2
    if Dtot >= 1: D_cost_value_2 += (Dtot-1)
    #check whether h2 lies within requested triangle cell
    D1, D2, Dtot = triangle_interior_check(h_2_2, p_alpha_2, p_beta_2, p_gamma_2)
    if D1 <= 0: D_cost_value_2 += -D1
    if D2 <= 0: D_cost_value_2 += -D2
    if Dtot >= 1: D_cost_value_2 += (Dtot-1)
    
    #calculate joint positions relative to cell edges
    dist_a_j2_2 = mindist_point_to_line(point=j_2_2, linepoints = [p_gamma_2, p_beta_2])
    opt_dist_a_j2_2 = 1*h
    
    dist_b_j1_2 = mindist_point_to_line(point=j_1_2, linepoints = [p_gamma_2, p_alpha_2])
    opt_dist_b_j1_2 = 1*h
    c_2 = la.norm(p_beta_2-p_alpha_2)
    dist_b_j1 = la.norm(j_1_2- p_alpha_2)
    opt_dist_b_j1_2 = (c_2-d1)/2.
    
    
    b_j1_2_cost = (dist_b_j1_2-opt_dist_b_j1_2)**2
    min_dist_b_j1_2 = d1+h
    max_dist_b_j1_2 = la.norm(p_beta_2-p_alpha_2)-h
    b_j1_2_cost = 1/(ch2-min_dist_b_j1_2)**2+ 1/(ch2-max_dist_b_j1_2)**12 #favour max dist
    # b_j1_2_cost = 1/(ch-min_dist_b_j1_2)**2 #favour max dist
    
    cost_2 = Dweight*D_cost_value_2 + Aweight*(dist_a_j2_2-opt_dist_a_j2_2)**2 + Bweight*b_j1_2_cost
    if vocal: print('\t dist 2',dist_b_j1_2, 'limits',min_dist_b_j1_2, max_dist_b_j1_2 ,'cost', b_j1_2_cost)
    
    ###########################################################################
    #get cell edge match cost function
    # print('j2', j_2, 'j22', j_2_2, 'j22m', j_2_2_mirrored)
    # sys.exit()
    dist_j2_cells = la.norm(j_2 - j_2_2_mirrored)
    if vocal: print('\t j2 dist', dist_j2_cells)
    opt_dist_j2_cells = opt_dist_a_j2 + opt_dist_a_j2_2
    
    cost_12 = Cweight*(dist_j2_cells-opt_dist_j2_cells)**2
    
    total_cost = cost_1 + cost_2 + cost_12
    if vocal: print('\t cost 1: ', cost_1, 'cost 2', cost_2, 'pair cost', cost_12, 'total', total_cost)
    
    if return_best_d2:
        ###########################################################################
        #get d2 that optimizes cell edge match
        cutline_crossing_1 = find_crossing_point(j_1, h_2-j_1, p_gamma, p_beta-p_gamma)
        cutline_crossing_2 = find_crossing_point(j_1_2, h_2_2-j_1_2, p_gamma_2, p_beta_2-p_gamma_2)
        cutline_crossing_avg = (cutline_crossing_1 + cutline_crossing_2)/2.
        d2_opt = la.norm(cutline_crossing_avg-p_gamma)
        if vocal: print('\t crossings: ', cutline_crossing_1, cutline_crossing_2)
        
        pd2_opt = p_gamma + (p_beta-p_gamma)/la.norm(p_beta - p_gamma)*d2_opt
                
        # len_j2_pd2 = mindist_point_to_line(point=j_2, linepoints=[h_2, pd2_opt])
        # len_j22_pd2 = mindist_point_to_line(point=j_2_2, linepoints=[h_2_2, pd2_opt])
        # if len_j2_pd2 < h:
        #     cutline_crossing_avg = (3*cutline_crossing_1 + cutline_crossing_2)/4.
        #     d2_opt = la.norm(cutline_crossing_avg-p_gamma)
        # elif len_j22_pd2 < h:
        #     cutline_crossing_avg = (cutline_crossing_1 + 3*cutline_crossing_2)/4.
        #     d2_opt = la.norm(cutline_crossing_avg-p_gamma)
        
        return total_cost, d2_opt
        
    return total_cost
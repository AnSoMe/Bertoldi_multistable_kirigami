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
# from general_quadtile_design_costfunctiontests import *
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
    
def mirror_points(points, mirror_point, mirror_angle):
    mirmat = np.array([[np.cos(2*mirror_angle),np.sin(2*mirror_angle)],[np.sin(2*mirror_angle),-np.cos(2*mirror_angle)]])
    new_points = np.array([np.dot(mirmat, point-mirror_point) + mirror_point for point in points])
    return new_points

def transform_points(points, translation_vec, rotation_point, rotation_angle):
    
    rotmat = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                       [np.sin(rotation_angle), np.cos(rotation_angle)]])
    new_points = np.array([np.dot(rotmat, point - rotation_point) + rotation_point + translation_vec for point in points])
    
    return new_points

def generate_tiledict_penrose(penrose_length=20.):
    tiledict = {
                'thinquad_tile1': {'base_angle': np.pi/10.,
                                  'length_a': penrose_length*np.cos(np.pi/10.),
                                  'length_b': penrose_length/2.,
                                  'expansion_ratio': 1.5,
                                  'deltat': 8*np.pi/10./2.,
                                  'd1': 4,
                                  'd2': 4,
                                  'hingelength':0.7,
                                  'hingedist':penrose_length/2-2.,
                                  },
                'thinquad_tile2': {'base_angle': 4*np.pi/10.,
                                  'length_a': penrose_length*np.cos(4*np.pi/10.),
                                  'length_b': penrose_length/2.,
                                  'expansion_ratio': 1.5,
                                  'deltat': 2*np.pi/10./2.,
                                  'd1': 4,
                                  'd2': 4,
                                  'hingelength':0.7,
                                  'hingedist':penrose_length/2-2.,
                                  },
                'fatquad_tile1': {'base_angle': 2*np.pi/10.,
                                  'length_a': penrose_length*np.cos(2*np.pi/10.),
                                  'length_b': penrose_length/2.,
                                  'expansion_ratio': 1.5,
                                  'deltat': 6*np.pi/10./2.,
                                  'd1': 4,
                                  'd2': 4,
                                  'hingelength':0.7,
                                  'hingedist':penrose_length/2-2.,
                                  },
                'fatquad_tile2': {'base_angle': 3*np.pi/10.,
                                  'length_a': penrose_length*np.cos(3*np.pi/10.),
                                  'length_b': penrose_length/2.,
                                  'expansion_ratio': 1.5,
                                  'deltat': 4*np.pi/10./2.,
                                  'd1': 4,
                                  'd2': 4,
                                  'hingelength':0.7,
                                  'hingedist':penrose_length/2-2.,
                                  },
                }
    return tiledict

def get_matchdict_Penrose():
    mydict = {
                'thinquad_tile1':'thinquad_tile1',
                'thinquad_tile2':'thinquad_tile2',
                'fatquad_tile1':'fatquad_tile1',
                'fatquad_tile2':'fatquad_tile2',
                }
    
    return mydict

def tesselate_Penrose_cells(newtiledict, tiledict):
    dict_cells_to_draw = {}
    zeropt = np.array([0.,0.])
    
    ###########################################################################
    #Quad cell 1
    quad1_points = newtiledict['thinquad_tile1']
    quad1_points_paired = mirror_points(quad1_points, mirror_point=zeropt, mirror_angle=np.pi/2)
    q1p_2m = mirror_points(quad1_points, mirror_point=zeropt, mirror_angle=np.pi)
    q1p_2pm = mirror_points(quad1_points_paired, mirror_point=zeropt, mirror_angle=np.pi)
    quad2_points = newtiledict['thinquad_tile2']
    quad2_points_paired = mirror_points(quad2_points, mirror_point=zeropt, mirror_angle=np.pi/2)
    q2p_2 = transform_points(quad2_points, np.array([0,0]), rotation_point=zeropt, rotation_angle=np.pi/2.)
    q2p_2p = transform_points(quad2_points_paired, np.array([0,0]), rotation_point=zeropt, rotation_angle=np.pi/2.)
    q2p_2m = mirror_points(q2p_2, mirror_point=zeropt, mirror_angle=np.pi/2.)
    q2p_2pm = mirror_points(q2p_2p, mirror_point=zeropt, mirror_angle=np.pi/2.)

    dict_cells_to_draw['thinquad'] = [quad1_points, quad1_points_paired, 
                                         q1p_2m, q1p_2pm, q2p_2, q2p_2p,
                                         q2p_2m, q2p_2pm]
    ###########################################################################
    #Quad cell 2
    quad1_points = newtiledict['fatquad_tile1']
    quad1_points_paired = mirror_points(quad1_points, mirror_point=zeropt, mirror_angle=np.pi/2)
    q1p_2m = mirror_points(quad1_points, mirror_point=zeropt, mirror_angle=np.pi)
    q1p_2pm = mirror_points(quad1_points_paired, mirror_point=zeropt, mirror_angle=np.pi)
    quad2_points = newtiledict['fatquad_tile2']
    quad2_points_paired = mirror_points(quad2_points, mirror_point=zeropt, mirror_angle=np.pi/2)
    q2p_2 = transform_points(quad2_points, np.array([0,0]), rotation_point=zeropt, rotation_angle=np.pi/2.)
    q2p_2p = transform_points(quad2_points_paired, np.array([0,0]), rotation_point=zeropt, rotation_angle=np.pi/2.)
    q2p_2m = mirror_points(q2p_2, mirror_point=zeropt, mirror_angle=np.pi/2.)
    q2p_2pm = mirror_points(q2p_2p, mirror_point=zeropt, mirror_angle=np.pi/2.)

    dict_cells_to_draw['fatquad'] = [quad1_points, quad1_points_paired, 
                                         q1p_2m, q1p_2pm, q2p_2, q2p_2p,
                                         q2p_2m, q2p_2pm]
    
    return dict_cells_to_draw

def generate_tiledict_Girih(Girih_length=20.):
    #custom hex parameters
    l = Girih_length
    h = l/2 + l*np.cos(2*np.pi/10.)
    w = l*np.sin(2*np.pi/10)
    f = np.sqrt(h**2 + (l/2)**2 - l*h*np.cos(2*np.pi/10))
    d = np.sqrt(f**2 + (l/2)**2 - l*f*np.cos(3*np.pi/10))
    zeta = np.arctan(1/(2*np.sin(2*np.pi/10)))
    omega = 4*np.pi/10-zeta
    x1 = l/2 * 1/(np.cos(2*np.pi/10.))
    x2 = l/2*np.tan(2*np.pi/10)
    x3 = np.sqrt(x1**2 + (l/2)**2 - x1*l*np.cos(6*np.pi/10))
    mygamma = np.arccos((x1**2 + x3**2 - (l/2)**2)/(2*x1*x3))
    mynu = mygamma + np.pi/10.
    myxi = 4/10*np.pi - mygamma
    x4 = x3*np.sin(mynu)
    
    lengthdict = {'p': Girih_length/2. * (2*np.cos(4/10*np.pi))**2,
                  'r': Girih_length/2. * np.tan(2/10*np.pi),
                  'o': Girih_length * (2*np.cos(4/10*np.pi))**2,
                  'l/2': Girih_length/2.,
                  's':  Girih_length * np.sin(2/10*np.pi),
                  'm':  Girih_length * 2*np.cos(4/10*np.pi),
                  't':  Girih_length/2  *np.tan(2/10*np.pi)**-1,
                  'q':  Girih_length/2 * (2*np.cos(4/10*np.pi))**-1,
                  'u': Girih_length/2 * np.sin(2/10*np.pi)**-1,
                  'l': Girih_length,
                  'n': Girih_length * (2*np.cos(4/10*np.pi))**-1,
                  'f': f,
                  'd': d,
                  'h': h,
                  'w': w,
                  'k': Girih_length/2. * np.tan(2/10*np.pi)**2,
                  'j': Girih_length/2 * np.sin(3/10*np.pi)**-1,
                  'i': Girih_length/2 * np.sin(3/10*np.pi)**-1 * np.cos(1/10*np.pi),
                  'da': Girih_length/2 * np.tan(1/10*np.pi)**-1,
                  'db': Girih_length/2 * np.sin(1/10*np.pi)**-1,
                  'x3': x3,
                  'x4': x4,
                  }

    tiledict = {'quad_v2-1':{'base_angle': 3*np.pi/10.,
                             'length_a': 'l/2',
                             'length_b':'s',
                             'd_2':5.
                             },
                'quad_v2-2':{'base_angle': 2*np.pi/10.,
                             'length_a': 'l/2',
                             'length_b':'q',
                             'd_2':5.
                                         },
                'penta_v2':{'base_angle': 2*np.pi/10.,
                             'length_a': 't',
                             'length_b':'u',
                             'd_2':9.5
                                         },
                'r-hex_v3-edge-1':{'base_angle': 3*np.pi/10.,
                             'length_a': 'r',
                             'length_b':'m',
                             'd_2':5.
                                         },
                'r-hex_v3-edge-4':{'base_angle': 1*np.pi/10.,
                             'length_a': 'i',
                             'length_b':'j',
                             'd_2':5.5
                                         },
                'deca-edge-1':{'base_angle': 1*np.pi/10.,
                             'length_a': 'da',
                             'length_b':'db',
                             'd_2': 23
                                         },
                'hex_v4-edge-1':{'base_angle': myxi, 
                             'length_a': 'x3',
                             'length_b': 'x4',
                             'd_2':5.5
                                         },
                'hex_v4-edge-2':{'base_angle': mygamma,
                             'length_a': 'x3',
                             'length_b': 'm',
                             'd_2':5.5
                                         },
                'hex_v4-edge-3':{'base_angle': 3*np.pi/10.,
                             'length_a': 'r',
                             'length_b': 'm',
                             'd_2':4.5
                                         },
                }
    
    for tk, dct in tiledict.items():
        lakey = dct['length_a']
        lbkey = dct['length_b']
        laval = lengthdict[lakey]
        lbval = lengthdict[lbkey]
        tiledict[tk]['length_a'] = laval
        tiledict[tk]['length_b'] = lbval
    return tiledict

def get_matchdict_Girih():
    mydict = {
                'quad_v2-1': 'quad_v2-2',
                'quad_v2-2': 'quad_v2-1',
                'penta_v2': 'penta_v2',
                'r-hex_v3-edge-1': 'r-hex_v3-edge-1',
                'r-hex_v3-edge-4': 'r-hex_v3-edge-4',
                'deca-edge-1': 'deca-edge-1',
                'hex_v4-edge-1': 'hex_v4-edge-2',
                'hex_v4-edge-2': 'hex_v4-edge-1',
                'hex_v4-edge-3': 'hex_v4-edge-3'
                }
    
    return mydict

def tesselate_Girih_cells(newtiledict, tiledict):
    dict_cells_to_draw = {}
    zeropt = np.array([0.,0.])
    
    ###########################################################################
    #Quad cell
    quad1_points = newtiledict['quad_v2-1']
    quad2_points = newtiledict['quad_v2-2']
    #mirror to match edges
    quad2_points_mirrored = mirror_points(quad2_points, mirror_point=zeropt, mirror_angle=np.pi/2.)
    #create rotated copies
    q1p_1 = deepcopy(quad1_points)
    q2p_1 = deepcopy(quad2_points_mirrored)
    q1p_2 = mirror_points(q1p_1, mirror_point=zeropt, mirror_angle=np.pi/2-2*np.pi/10.)
    q2p_2 = mirror_points(q2p_1, mirror_point=zeropt, mirror_angle=np.pi/2-2*np.pi/10.)
    q1p_3 = mirror_points(q1p_1, mirror_point=zeropt, mirror_angle=-2*np.pi/10.)
    q2p_3 = mirror_points(q2p_1, mirror_point=zeropt, mirror_angle=-2*np.pi/10.)
    q1p_4 = transform_points(q1p_1, np.array([0,0]), rotation_point=zeropt, rotation_angle=np.pi)
    q2p_4 = transform_points(q2p_1, np.array([0,0]), rotation_point=zeropt, rotation_angle=np.pi)
    reghex = [q1p_1, q2p_1, q1p_2, q2p_2,q1p_3, q2p_3, q1p_4, q2p_4]
    dict_cells_to_draw['quad_v2'] = reghex
    
    ###########################################################################
    #Penta cell
    penta_1 = newtiledict['penta_v2']
    penta_m_1 = mirror_points(penta_1, mirror_point=penta_1[0], mirror_angle=np.pi/2.)
    penta_2 = transform_points(penta_1, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_m_2 = transform_points(penta_m_1, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_3 = transform_points(penta_2, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_m_3 = transform_points(penta_m_2, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_4 = transform_points(penta_3, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_m_4 = transform_points(penta_m_3, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_5 = transform_points(penta_4, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_m_5 = transform_points(penta_m_4, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)

    dict_cells_to_draw['corner-penta'] = [penta_1, penta_m_1,
                                          penta_2, penta_m_2,
                                          penta_3, penta_m_3,
                                          penta_4, penta_m_4,
                                          penta_5, penta_m_5]
    
    ###########################################################################
    #R-hex cell
    rhexa = newtiledict['r-hex_v3-edge-1']
    rhexa_m = mirror_points(rhexa, zeropt, np.pi/2.)
    rhexb = transform_points(newtiledict['r-hex_v3-edge-4'], zeropt, zeropt, np.pi)
    rhexb_m = transform_points(mirror_points(rhexb, zeropt, np.pi/2.), zeropt, zeropt, 0.)
    rhexa2 = transform_points(rhexa, zeropt, zeropt, -3*np.pi/5)
    rhexa2_m = transform_points(rhexa_m, zeropt, zeropt, 3*np.pi/5) 
    rhexa3 = transform_points(rhexa, zeropt, zeropt, 3*np.pi/5)
    rhexa3_m = transform_points(rhexa_m, zeropt, zeropt, -3*np.pi/5) 
    reghex = [rhexa, rhexa_m, rhexa2, rhexa2_m, rhexa3, rhexa3_m, rhexb, rhexb_m]
    #mirror the lot
    mhexs = []
    for hexi in reghex:
        zeropt = np.array([0.,0.])
        fliphex = transform_points(hexi, zeropt, rhexb[1], np.pi)
        mhexs.append(fliphex)
    dict_cells_to_draw['r-hex_v3-edge'] = reghex + mhexs
    
    ###########################################################################
    #Hex cell
    rhexb = newtiledict['hex_v4-edge-3']
    rhexb_m = transform_points(mirror_points(rhexb, zeropt, np.pi/2.), zeropt, zeropt, 0.)
    rhexb2 = transform_points(rhexb, zeropt, zeropt, -3*np.pi/10)
    rhexb2_m = transform_points(rhexb_m, zeropt, zeropt, -3*np.pi/10) 
    rhexb3 = transform_points(rhexb, zeropt, zeropt, 3*np.pi/10)
    rhexb3_m = transform_points(rhexb_m, zeropt, zeropt, 3*np.pi/10) 
    #big asymmetric tiles
    rhexa = newtiledict['hex_v4-edge-1']
    rhexa_match = newtiledict['hex_v4-edge-2']
    rhexa_m = mirror_points(rhexa_match, zeropt, np.pi/2.)
    base_2 = tiledict['hex_v4-edge-2']['base_angle']
    rhexa2 = transform_points(rhexa, zeropt, zeropt, (base_2+6*np.pi/10))
    rhexa2_m = transform_points(rhexa_m, zeropt, zeropt, (base_2+6*np.pi/10)) 
    rhexa3 = mirror_points(rhexa2, zeropt, np.pi/2.)
    rhexa3_m = mirror_points(rhexa2_m, zeropt, np.pi/2.) 
    reghex = [rhexa2, rhexa2_m, rhexa3, rhexa3_m, rhexb2, rhexb2_m, rhexb3, rhexb3_m]
    #mirror the lot
    mhexs = []
    for hexi in reghex:
        zeropt = np.array([0.,0.])
        fliphex = transform_points(hexi, zeropt, rhexa2[2], np.pi)
        mhexs.append(fliphex)
    dict_cells_to_draw['hex_v4-edge'] =  reghex + mhexs
    
    ###########################################################################
    #Deca cell
    deca_1 = newtiledict['deca-edge-1']
    deca_1m = mirror_points(deca_1, zeropt, np.pi/2.)
    reghex = []
    for i in range(10):
        mydeca = transform_points(deca_1, zeropt, zeropt, i*np.pi/5.)
        mydeca_m = transform_points(deca_1m, zeropt, zeropt, i*np.pi/5.)
        reghex.extend([mydeca, mydeca_m])
    dict_cells_to_draw['deca-edge'] = reghex
    
    return dict_cells_to_draw
    
def generate_tiledict_Girih_corner(Girih_length=20.):
    l=Girih_length
    gammaprime = np.arctan(1/(2*np.sin(2*np.pi/10.)))
    tiledict = {
                'corner-quad-1':{'base_angle': 3*np.pi/10.,
                             # 'length_a': l*np.sqrt(1/2-1/2*np.cos(4*np.pi/10.)),
                             # 'length_a': l*np.sin(2*np.pi/10.),
                             'length_a': l*np.cos(3*np.pi/10.),
                             'length_b': l/2.,
                             'd_2':5.3
                             },
                'corner-quad-2':{'base_angle': 2*np.pi/10.,
                             'length_a': l*np.sqrt(1/2-1/2*np.cos(6*np.pi/10.)),
                             'length_b': l/2.,
                             'd_2':7.7
                             },
                'corner-penta':{'base_angle': 2*np.pi/10.,
                             'length_a': l/(2*np.sin(2*np.pi/10)),
                             'length_b': l/(2*np.tan(2*np.pi/10)),
                             'd_2':10.6
                             },
                'corner-deca':{'base_angle': np.pi/10.,
                             'length_a': l/(2*np.sin(np.pi/10)),
                             'length_b': l/(2*np.tan(np.pi/10)),
                             'd_2':29
                             },
                'corner-rhex-1':{'base_angle': 3*np.pi/10.,
                              'length_a': l/(2*np.sin(3*np.pi/10)),
                              'length_b': l/(2*np.tan(3*np.pi/10)),
                              'd_2':3.7
                              },
                'corner-rhex-2':{'base_angle': np.pi/10.,
                              'length_a': l/(2*np.sin(3*np.pi/10)),
                              'length_b': l/(2*np.sin(3*np.pi/10)) * np.cos(np.pi/10.),
                              'd_2':7.5
                              },
                'corner-rhex-3':{'base_angle': 3*np.pi/10.,
                              'length_a': l/(2*np.sin(3*np.pi/10)),
                              'length_b': l/(2*np.tan(3*np.pi/10)),
                              'd_2':7.5
                              },
                'corner-hexv2-1':{'base_angle': 3*np.pi/20.,
                             'length_a': l*np.sqrt( (np.sin(2*np.pi/10)/np.sin(7*np.pi/20))**2 + np.sin(2*np.pi/10)/np.tan(7*np.pi/20) + 1/4 ),
                             'length_b': l*np.sin(2*np.pi/10.)/np.sin(7*np.pi/20),
                             'd_2':11
                             },
                'corner-hexv2-2':{'base_angle': 5*np.pi/20,
                             'length_a': l*np.sqrt( (np.sin(2*np.pi/10)/np.sin(7*np.pi/20))**2 - np.sin(2*np.pi/10)/np.tan(7*np.pi/20) + 1/4 ),
                             'length_b': l*np.sin(2*np.pi/10.)/np.sin(7*np.pi/20),
                             'd_2':8.5
                             },
                'corner-hexv2-3':{'base_angle':  7*np.pi/20,
                             'length_a': l*np.sin(2*np.pi/10.)/np.tan(7*np.pi/20),
                             'length_b': l*np.sin(2*np.pi/10.)/np.sin(7*np.pi/20),
                             'd_2': 4.7
                             },
                'corner-hex-1':{'base_angle': np.pi/10,
                             'length_a': l*(1/2+np.cos(2*np.pi/10)),
                             'length_b': l*np.sqrt(1/2+1/2*np.cos(2*np.pi/10)),
                             'd_2':18.2
                             },
                'corner-hex-2':{'base_angle': 4*np.pi/10-gammaprime,
                             'length_a': l*np.sqrt(1/4+np.sin(2*np.pi/10)**2),
                             'length_b': l*np.sqrt(1/2+1/2*np.cos(2*np.pi/10)),
                             'd_2':15
                             },
                'corner-hex-3':{'base_angle': gammaprime,
                             'length_a': l*np.sqrt(1/4+np.sin(2*np.pi/10)**2),
                             'length_b': l*np.sin(2*np.pi/10.),
                             'd_2':12
                             },
                }
    
    return tiledict

def get_matchdict_Girih_corner():
    mydict = {
                'corner-quad-1': 'corner-quad-1',
                'corner-quad-2': 'corner-quad-2',
                'corner-penta': 'corner-penta',
                'corner-deca': 'corner-deca',
                'corner-rhex-1': 'corner-rhex-1',
                'corner-rhex-2': 'corner-rhex-3',
                'corner-rhex-3': 'corner-rhex-2',
                'corner-hex-1': 'corner-hex-1',
                'corner-hex-2': 'corner-hex-3',
                'corner-hex-3': 'corner-hex-2',
                'corner-hexv2-1': 'corner-hexv2-1',
                'corner-hexv2-2': 'corner-hexv2-2',
                'corner-hexv2-3': 'corner-hexv2-3',
                }
    return mydict

def tesselate_Girih_cells_corner(newtiledict, tiledict):
    dict_cells_to_draw = {}
    zeropt = np.array([0.,0.])
    
    ###########################################################################
    #Quad cell
    quad1_points = newtiledict['corner-quad-1']
    quad1_points_paired = mirror_points(quad1_points, mirror_point=zeropt, mirror_angle=np.pi/2)
    q1p_2m = mirror_points(quad1_points, mirror_point=zeropt, mirror_angle=np.pi)
    q1p_2pm = mirror_points(quad1_points_paired, mirror_point=zeropt, mirror_angle=np.pi)
    quad2_points = newtiledict['corner-quad-2']
    quad2_points_paired = mirror_points(quad2_points, mirror_point=zeropt, mirror_angle=np.pi/2)
    q2p_2 = transform_points(quad2_points, np.array([0,0]), rotation_point=zeropt, rotation_angle=np.pi/2.)
    q2p_2p = transform_points(quad2_points_paired, np.array([0,0]), rotation_point=zeropt, rotation_angle=np.pi/2.)
    q2p_2m = mirror_points(q2p_2, mirror_point=zeropt, mirror_angle=np.pi/2.)
    q2p_2pm = mirror_points(q2p_2p, mirror_point=zeropt, mirror_angle=np.pi/2.)

    dict_cells_to_draw['corner-quad'] = [quad1_points, quad1_points_paired, 
                                         q1p_2m, q1p_2pm, q2p_2, q2p_2p,
                                         q2p_2m, q2p_2pm]
    
    ###########################################################################
    #Penta
    penta_1 = newtiledict['corner-penta']
    penta_m_1 = mirror_points(penta_1, mirror_point=penta_1[0], mirror_angle=np.pi/2.)
    penta_2 = transform_points(penta_1, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_m_2 = transform_points(penta_m_1, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_3 = transform_points(penta_2, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_m_3 = transform_points(penta_m_2, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_4 = transform_points(penta_3, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_m_4 = transform_points(penta_m_3, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_5 = transform_points(penta_4, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)
    penta_m_5 = transform_points(penta_m_4, np.array([0,0]), rotation_point=zeropt, rotation_angle=2*np.pi/5.)

    dict_cells_to_draw['corner-penta'] = [penta_1, penta_m_1,
                                          penta_2, penta_m_2,
                                          penta_3, penta_m_3,
                                          penta_4, penta_m_4,
                                          penta_5, penta_m_5]
    
    ###########################################################################
    #R-hex
    rhex_1 = newtiledict['corner-rhex-1']
    rhex_2 = newtiledict['corner-rhex-2']
    #construct matching tiles
    rhex_1_paired = mirror_points(rhex_1, zeropt, np.pi/2.)
    rhex_2_paired = mirror_points(newtiledict['corner-rhex-3'], zeropt, np.pi/2.)
    rh1l = transform_points(rhex_1, zeropt, zeropt, 3*np.pi/10)
    rh1lp = transform_points(rhex_1_paired, zeropt, zeropt, 3*np.pi/10)
    rh1r = transform_points(rhex_1, zeropt, zeropt, -3*np.pi/10)
    rh1rp = transform_points(rhex_1_paired, zeropt, zeropt, -3*np.pi/10)
    rh2 = transform_points(rhex_2_paired, zeropt, zeropt, np.pi-np.pi/10.)
    rh2p = transform_points(rhex_2, zeropt, zeropt, np.pi-np.pi/10.)
    rh3 = mirror_points(rh2, zeropt, np.pi/2)
    rh3p = mirror_points(rh2p, zeropt, np.pi/2)
    #flip
    reghex = [rh1l, rh1lp, rh1r, rh1rp, rh2, rh2p, rh3, rh3p]
    mhexs = []
    for hexi in reghex:
        fliphex = transform_points(hexi, zeropt, rh3p[2], np.pi)
        mhexs.append(fliphex)
    dict_cells_to_draw['corner-rhex'] = reghex + mhexs
    
    ###########################################################################
    #Hex v1
    hex_1 = newtiledict['corner-hex-1']
    hex_2 = newtiledict['corner-hex-2']
    hex_3 = newtiledict['corner-hex-3']
    #construct matching tiles
    hex_1_paired = mirror_points(hex_1, zeropt, np.pi/2.)
    hex_2_paired = mirror_points(hex_3, zeropt, np.pi/2.)
    #construct rotated versions
    h2 = transform_points(hex_2, zeropt, zeropt, - tiledict['corner-hex-2']['base_angle']-np.pi/10.)
    h2p = transform_points(hex_2_paired, zeropt, zeropt, - tiledict['corner-hex-2']['base_angle']-np.pi/10.)
    h3= mirror_points(h2, zeropt, np.pi/2.)
    h3p = mirror_points(h2p, zeropt, np.pi/2.)
    reghex = [hex_1, hex_1_paired, h2, h2p, h3, h3p]
    mhexs = []
    for hexi in reghex:
        fliphex = transform_points(hexi, zeropt, zeropt, np.pi)
        mhexs.append(fliphex)
    dict_cells_to_draw['corner-hexv1'] = reghex + mhexs
    
    ###########################################################################
    #hex v2
    zeropt = np.array([0.,0.])
    hexv2_1 = newtiledict['corner-hexv2-1']
    hexv2_2 = newtiledict['corner-hexv2-2']
    hexv2_3 = newtiledict['corner-hexv2-3']

    #construct matching tiles
    hexv2_1_paired = mirror_points(hexv2_1, zeropt, np.pi/2.)
    hexv2_2_paired = mirror_points(hexv2_2, zeropt, np.pi/2.)
    hexv2_3_paired = mirror_points(hexv2_3, zeropt, np.pi/2.)
    hexv2_2l = transform_points(hexv2_2, zeropt, zeropt, 4*np.pi/10.)
    hexv2_2m = transform_points(hexv2_2_paired, zeropt, zeropt, 4*np.pi/10.)
    hexv2_2r = mirror_points(hexv2_2l, zeropt, np.pi/2.)
    hexv2_2s = mirror_points(hexv2_2m, zeropt, np.pi/2.)
    hexv3_2l = mirror_points(hexv2_3, zeropt, np.pi)
    hexv3_2m = mirror_points(hexv2_3_paired, zeropt, np.pi)

    mhexs = []
    for hexi in [hexv2_1, hexv2_1_paired, hexv2_2l, hexv2_2m, hexv2_2r, hexv2_2s, hexv3_2l, hexv3_2m]:
        fliphex = transform_points(hexi, zeropt, hexv3_2l[1], np.pi)
        mhexs.append(fliphex)
    reghex = [hexv2_1, hexv2_1_paired, hexv2_2l, hexv2_2m, hexv2_2r, hexv2_2s, hexv3_2l, hexv3_2m]
    dict_cells_to_draw['corner-hexv2'] = reghex + mhexs
    
    ###########################################################################
    #Deca
    zeropt = np.array([0.,0.])
    deca_1 = newtiledict['corner-deca']
    deca_1m = mirror_points(deca_1, zeropt, np.pi/2.)
    reghex = []
    for i in range(10):
        mydeca = transform_points(deca_1, zeropt, zeropt, i*np.pi/5.)
        mydeca_m = transform_points(deca_1m, zeropt, zeropt, i*np.pi/5.)
        reghex.extend([mydeca, mydeca_m])

    dict_cells_to_draw['corner-deca'] = reghex
    
    return dict_cells_to_draw

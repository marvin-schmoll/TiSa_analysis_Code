# -*- coding: utf-8 -*-
"""
Ti:Sa HHG Analysis Software
(based on existing Code for K04) 

@author: Marvin Schmoll
marvin.schmoll@physik.uni-freiburg.de
"""


import h5py

import numpy as np
from matplotlib import pyplot as plt
import scipy.signal

from os import listdir
import tkinter as tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename


#TODO: Check if all constants still vaqlid for Ti:Sa
planck = 4.135667516 * 10**(-15) # plancks constant [eV*s]
lightspeed = 299792458           # speed of light [m/s]
g = 1/1200 * 10**(-3)            # grating constant [m]
dist_g_MCP = 469 * 10**(-3)      # distance between grating and MCP [m]
distance_from_focus = 1.37       #*** distance between grating and VMI focus [m]
alpha = 85.3 / 180 * np.pi       # grating incidence angle [rad]
spatial_scale = 43.3 * 10**(-6)  # camera spatial scale [m/pixel]
lambda_IR = 786 * 10**(-9)       # wavelength [m]
#X_0 = -9310                      #*** offset between spectrometer zero and actual zero-order [pixels]



def my_tuple(array):
    '''emulates the python tuple() typecasting from arrays,
        but the tuple with len=1 is replaced by the element itself'''
    my_tuple = tuple(array)
    if len(my_tuple)>1:     return my_tuple
    elif len(my_tuple)==1:  return my_tuple[0]
    else:                   return None



def plot_2D(intensity_data, plot_title='', HH_locs=[], HH_numbers=[], saving=False, path=''):
    '''creates a 2D colormap of the input data'''

    plt.figure()
    plt.imshow(intensity_data, cmap = 'nipy_spectral')
    plt.colorbar(pad = .01, shrink = .81, aspect = 40).set_label('Intensity [arb. units]')

    # calculate divergence for y-label
    dimy, dimx = intensity_data.shape
    scale = spatial_scale / distance_from_focus * dimy   # see Excel-sheet
    ylabels = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ypositions    =  (-ylabels/scale + dimy/2)
    plt.yticks(ypositions, ylabels)
    plt.ylabel('divergence [mrad]')

    if len(HH_locs) > 0 and len(HH_numbers) > 0:   # label the harmonics
        labels = ['HH' + str(HH_numbers[i]) for i in range(len(HH_numbers))]
        plt.xticks(HH_locs, labels, rotation=90)

    if plot_title[-4:] == '.lvm':
        plt.suptitle('HHG-Spectrum', size=14)
        plt.title(plot_title, size=10, linespacing=1)

    else:   plt.title(plot_title)

    if saving == True:
        plt.savefig((path + plot_title + '_2D.pdf'), dpi=600)

    plt.show()



def read_MCP_data(folder, file, background_file=None, plotting=True):
    '''Reads in the MCP data from the h5-files provided by LabVIEW 
        and returns it as NumPy-array
        This method averages over multiple images into one.
    '''
    
    # read file
    f = h5py.File(folder + file)
    raw_MCP_data = np.array(f['Images']).mean(axis=2)
    
    # subtract background
    if background_file is None:
        MCP_data = raw_MCP_data - np.array(f['Background Image'])
    else:
        b = h5py.File(folder + background_file)
        MCP_data = raw_MCP_data - np.array(b['Images']).mean(axis=2)
            
        
    # plot image if requested
    if plotting == True:
        plot_2D(MCP_data, file)

    return MCP_data


def save_MCP_data(folder, file, MCP_data,  HH_locs=[], HH_numbers=[]):

    file = file[:-3] + 'txt'
    
    dimy, dimx = MCP_data.shape
    scale = spatial_scale / distance_from_focus * dimx
    divergence_axis = (int(dimy/2)-np.arange(dimy))*scale
    composite_array = np.concatenate((np.array([divergence_axis]), MCP_data.T)).T

    headertext = 'MCP Spectrum \n first column: divergence axis [mrad] \n other collumns: pixel intensity [arb. units] (value 1 = detector saturated) \n'
    if len(HH_locs) > 0 and len(HH_numbers) > 0:   # add indicator where harmonics are
        headertext += 'Found harmonics (via peak finder): \n Harmonic order:         \t'
        for i in range(len(HH_numbers)):
            headertext += str(HH_numbers[i]) + '   \t'
        headertext += ' \n Peak position [pixels]: \t'
        for i in range(len(HH_locs)):
            headertext += str(HH_locs[i]) + ' \t'

    np.savetxt((folder + file), composite_array, header=headertext)

    return


def calculate_spectrum(MCP_data, integrate=True, plotting=True, saving=False,
                       plot_title='', HH_locs=[], HH_numbers=[], path='', left_side=[], right_side=[]):
    '''obtaines a normalized 1D XUV-spectrum from provided 2D MCP data
        if integrating is set to true the function integrates over y,
        else it just cuts out the central line'''

    if integrate==True: XUV_spectrum=np.sum(MCP_data, axis=0)   # intensity integrated over y
    else:               XUV_spectrum=MCP_data[600,:]            # intensity in the middle of the sensor

    if plotting == True:
        plt.plot(XUV_spectrum/np.max(XUV_spectrum), linewidth=2, c='k')
        plt.grid(True)

        if len(HH_locs) > 0 and len(HH_numbers) > 0:   # label the harmonics
            labels = ['HH' + str(HH_numbers[i]) for i in range(len(HH_numbers))]
            plt.xticks(HH_locs, labels, rotation=90)

        if len(left_side) > 0 and len(right_side) > 0:   # show window boundaries
                for h in left_side:
                    plt.plot([h, h], [0, 1], 'g', lw=0.7)
                for h in right_side:
                    plt.plot([h, h], [0, 1], 'g', lw=0.7)

        if plot_title[-4:] == '.lvm':
            plt.suptitle('HHG-Spectrum', size=14, y=1.00)
            plt.title(plot_title, size=7, linespacing=0.8, y=0.98)

        if saving == True:
            plt.savefig((path + plot_title + '_1D.pdf'), dpi=600)

        plt.show()

    return(XUV_spectrum)


def FWHM_divergence(full_data, x_left, x_right, plotting=3, title=''):
    '''calculates y-direction FWHM of a single harmonic by integrating over x;
        plotting allows to choose the number of plots <= 3 shown in the process'''

    # cut out the selected x-range
    selected_data = full_data.T[x_left:x_right]
    selected_data = selected_data.T

    if plotting == 3: plot_2D(full_data, title)
    if plotting >= 2: plot_2D(selected_data, title)

    intX = np.sum(selected_data, axis = 1)   # integrate over x
    intX=intX/np.max(intX)                   # normalize
    max_set = np.max(intX)                   # maximum of integrated data (=1)
    max_x = np.where(intX==max_set)[0][0]    # position of the maximum

    if plotting >=1:
        plt.plot(intX, label = ('maximum at y = ' + str(max_x)))
        plt.grid(True)
        plt.xlabel('$y$')
        plt.ylabel('Intensity [normalized]')
        plt.xlim([0, 1200])
        plt.title(title)
        plt.legend()
        plt.show()

    # find FWHM
    xx1 = np.where((intX[max_x:len(intX)]<=max_set/2))
    xx2 = np.where((intX[:max_x]<=max_set/2))
    if len(xx1[0])==0 or len(xx2[0])==0:
        print('FWHM could not be calculated; curve does not drop below half maximum')
        return None
    x1 = np.min(xx1) + max_x
    x2 = np.max(xx2)
    max_x1 = (x1-1)*(max_set/2-intX[x1])/(intX[x1-1]-intX[x1]) + (x1)*(intX[x1-1]-max_set/2)/(intX[x1-1]-intX[x1])
    max_x2 = (x2+1)*(max_set/2-intX[x2])/(intX[x2+1]-intX[x2]) + (x2)*(intX[x2+1]-max_set/2)/(intX[x2+1]-intX[x2])
    #print(max_x1, max_x2)
    fwhm = abs(max_x2-max_x1)

    return fwhm


def sideband_remover(*position_information):
    '''gaps between the peaks should become larger with increasing y.
        if this is not the case this removes sidebands in between
        the first element of tuple sideband_information should be its pixel position'''

    if len(position_information[0]) < 3:   return my_tuple(position_information)  # gaps cannot be compared

    differences = position_information[0][1:] - position_information[0][:-1]
    diff_diff = differences[1:] - differences[:-1] #   should be > 0
    test = np.where(diff_diff < 0)
    if len(test[0])==0: return my_tuple(position_information)   # done
    else:   # recursion
        fault = np.min(test) + 2
        return sideband_remover(*tuple([np.delete(info, fault) for info in position_information]))


def peak_finder(MCP_data, height=0.05, width=20, distance=100, rel_height=0.5,
                prominence=None, window_size=2, integrate=True, plotting=True, remove=True):
    '''uses a peak finding method to evaluate the positions of the visible harmonics,
        window_size scales how big the window for each harmonic is;
        outputs peak positions and left and right bound for the resulting window'''

    MCP_spectrum = calculate_spectrum(MCP_data, integrate=integrate, plotting=False)
    MCP_spectrum = MCP_spectrum/np.max(MCP_spectrum) # normalize

    harmonics, properties = scipy.signal.find_peaks(MCP_spectrum, height=height, prominence=prominence,
                                                    width=width, distance=distance, rel_height=rel_height)
    left_side=(np.floor(properties.get('left_ips')) - harmonics) * window_size + harmonics
    right_side=(np.ceil(properties.get('right_ips')) - harmonics) * window_size + harmonics

    if remove==True:
        harmonics, left_side, right_side = sideband_remover(harmonics, left_side, right_side)
    
    print('positions of the harmonics:', harmonics)

    # Indicate positions of the chosen parts of the spectrum
    for h in harmonics:
        plt.plot([h, h], [0, 1], 'g')
    for h in left_side:
        plt.plot([h, h], [0, 1], 'g', lw=0.5)
    for h in right_side:
        plt.plot([h, h], [0, 1], 'g', lw=0.5)

    if plotting == True:
        plt.plot(MCP_spectrum, linewidth=2, c='k')
        plt.show()

    return harmonics, left_side.astype(int), right_side.astype(int)


def HH_energy_scale(peaks):
    '''assigns the harmonic orders to each peak based on their distances
        via a curve fitting method based on the linearity of the wavelength scale'''
    
    def position(n, a, b):
        '''fittable function returning the pixel value of a harmonic of order (b-2*n)'''
        xx = lambda_IR / ((b - 2*n) * g) - np.sin(alpha)
        return -1 * dist_g_MCP / spatial_scale * np.tan(np.arccos(xx)) - a

    nn = np.arange(0, len(peaks), 1)

    popt, pcov = scipy.optimize.curve_fit(position, nn, peaks, p0=[0,41])
    print('optimal parameters', popt)

    plt.figure()
    plt.plot(nn, peaks, alpha=0.5, color='r', linestyle=None, marker='o')
    plt.plot(nn, position(nn, *popt), lw=0.8, ls='-.', color='k')
    plt.xlabel('peak index')
    plt.ylabel('MCP pixel')
    plt.show()

    return int(np.round(popt[1])) - 2*nn, popt, pcov




#%%

if __name__ == "__main__":

    folder = 'D:/2024-09-10/' 
    file = '1240_05harmonics_power_1p4W_9832.h5'
    background_file = '1244_34harmonics_power_1p4W_9832_laserblocked.h5'
    
    data = read_MCP_data(folder, file, background_file, plotting=True)
    HH, left, right = peak_finder(data, distance=40, window_size=2, width=10, height=0.03)
    harm_number = HH_energy_scale(HH)

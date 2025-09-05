# -*- coding: utf-8 -*-
"""
Marvins Utility Library
(utility functions to non data-specific tasks) 

@author: Marvin Schmoll
marvin.schmoll@physik.uni-freiburg.de
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.signal
import warnings

#%% Data Structures and typecasting

def my_tuple(array):
    '''emulates the python tuple() typecasting from arrays,
        but the tuple with len=1 is replaced by the element itself'''
    my_tuple = tuple(array)
    if len(my_tuple)>1:     return my_tuple
    elif len(my_tuple)==1:  return my_tuple[0]
    else:                   return None


#%% Simple data processing

def normalized(array, normalization='max'):
    '''Shorthand for normalizing arrays.
        "Normalization" decides, if the maximum or the sum of all values is set to one.'''

    if normalization in ['max', 'maximum', 'Maximum']:
        return array / np.nanmax(np.abs(array))

    if normalization in ['sum', 'Sum', 'int', 'integral', 'Integral']:
        return array / np.nansum(array)

    raise ValueError('Specify normalization convention as "maximum" or "sum"')


def find_FWHM(dataset):
    """
    Find the full with at half maximum of a dataset with a peak

    Parameters
    ----------
    dataset : 1D np.array
        One-dimensional array containing a peak.

    Returns
    -------
    float
        Full width at half maximum in bins
    (float, int, float)
        (left half-maximum, maximum, right half-maximum)
        
    Notes
    -----
    If the dataset has a more complicated multi-peaked structure
    the peak will be the highest singular value in the dataset,
    and the FWHM will be from the first point where the intensity is more than
    half this value to the last.

    """
    
    data = normalized(dataset)
    max_x = np.where(data==1)[0][0]    # position of the maximum
    
    # find FWHM
    xx1 = np.where((data[max_x:len(data)]<=1/2))
    xx2 = np.where((data[:max_x]<=1/2))
    if len(xx1[0])==0 or len(xx2[0])==0:
        warnings.warn('FWHM could not be calculated; curve does not drop below half maximum')
        return None
    x1 = np.min(xx1) + max_x
    x2 = np.max(xx2)
    max_x1 = (x1-1)*(1/2-data[x1])/(data[x1-1]-data[x1]) + (x1)*(data[x1-1]-1/2)/(data[x1-1]-data[x1])
    max_x2 = (x2+1)*(1/2-data[x2])/(data[x2+1]-data[x2]) + (x2)*(data[x2+1]-1/2)/(data[x2+1]-data[x2])
    #print(max_x1, max_x2)
    return abs(max_x2-max_x1) ,(x2, max_x, x1)

#%% Smoothing and Averaging

def smooth_1D(input_data, kernel_size=1, window='flat'):
    '''
    smoothes 2D data along the time/phase axis 

    Parameters
    ----------
    input_data : 2D numpy array
        The data to be smoothed.
    kernel_size : int, optional
        Window size of the smoothing kernel. The default is 1.
    window : TYPE, optional
        Function type of the cmoothing window. The default is 'flat'.

    Returns
    -------
    2D numpy array
        The smoothed dataset.
        
    Notes
    -----
    The working core of the code has been heavily adapted from:
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    Check for more documentation and a minimalistic example.
    '''
    
    input_data[np.isnan(input_data)] = 0 # TODO: Test and improve (interpolation?)
    
    # point symmetric interoplation over each endpoint (to minimize boundary effects)
    data_unsmoothed = np.concatenate((2*input_data[0]-input_data[kernel_size:0:-1],
                                      input_data, 2*input_data[-1]-input_data[-2:-kernel_size-2:-1]))

    if window == 'flat': # Create flat kernel
        kernel = np.ones([2*kernel_size + 1])

    elif window in ['hanning', 'hamming', 'bartlett', 'blackman']:
        kernel = np.array([getattr(np, window)(2*kernel_size + 1)]).T

    elif window == 'gauss':
        kernel = np.array([np.exp(-(np.arange(-kernel_size,kernel_size+1)**2 / kernel_size))]).T

    kernel = kernel / np.sum(kernel)
    return scipy.signal.convolve(data_unsmoothed, kernel, mode='valid')


def smooth_2D(input_data, k_x=1, k_y=3):
    '''
    smoothes 2D data along the both dimensions using a gaussian window

    Parameters
    ----------
    input_data : TYPE
        DESCRIPTION.
    k_x : int, optional
        Kernel size along x direction. The default is 1.
    k_y : int, optional
        Kernel size along y direction. The default is 3.

    Returns
    -------
    2D numpy array
        The smoothed dataset.
    
    Notes
    -----
    The working core of the code has been heavily adapted from:
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    Check for more documentation and a minimalistic example.
    '''
    
    input_data[np.isnan(input_data)] = 0 # TODO: Test and improve (interpolation?)
    
    # point symmetric interoplation over each endpoint (to minimize boundary effects)
    data_between = np.concatenate((2*input_data[0]-input_data[k_x:0:-1], input_data, 2*input_data[-1]-input_data[-2:-k_x-2:-1])).T
    data_unsmoothed = np.concatenate((2*data_between[0]-data_between[k_y:0:-1], data_between, 2*data_between[-1]-data_between[-2:-k_y-2:-1])).T

    def gauss_kernel(size, sizey=None):
        """ Returns a normalized 2D gauss kernel array for convolutions """
        size = int(size)
        if not sizey:
            sizey = size
        else:
            sizey = int(sizey)
        x, y = np.mgrid[0-size:size+1, 0-sizey:sizey+1]
        g = np.exp(-(x**2/size + y**2/sizey))
        return g / np.sum(g)

    return scipy.signal.convolve(data_unsmoothed, gauss_kernel(k_x, k_y), mode='valid')


#%% Plotting

def rainbow_colors(length, darken=1):
    """
    Creates a set of colors linearly sampled from the matplotlib rainbow colormap.

    Parameters
    ----------
    length : int
        Amount of colors to be generated.
    darken : float, optional
        Specify a value >1 to darken the colors for use in plotting (e.g. 1.3).
        The default is 1, which chooses the colors directly from the matplotlib 
        rainbow colormap.

    Returns
    -------
    colors_sat : arr of shape (length, 4)
        Array containing the colors.

    """

    colors = plt.get_cmap('rainbow')(np.linspace(1,0,length))
    colors_sat = colors
    colors_sat[:,:3] = colors[:,:3] / darken   # darken colors for better visibility
    return colors_sat

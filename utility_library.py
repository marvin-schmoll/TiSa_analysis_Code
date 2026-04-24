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
import os
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

#%% Physical constants

ionization_energies = {'He': 24.587, 'Ne': 21.565, 'Ar': 15.760, 'Kr': 14.000, 'Xe': 12.13,
                       'CH4': 13.6, 'CH3': 14.8, 'CH2': 15.8, 'CH': 22.9} # [eV]


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
    and the FWHM will be from the last point before thge max where the intensity 
    is less than half its value to the first one after the max.

    """
    
    data = normalized(dataset)
    max_x = np.where(data==1)[0][0]    # position of the maximum
    
    # find FWHM
    xx1 = np.where((data[max_x:len(data)]<=1/2))
    xx2 = np.where((data[:max_x]<=1/2))
    if len(xx1[0])==0:
        warnings.warn('FWHM could not be calculated properly; curve does not drop below half maximum')
        xx1 = len(data) - max_x - 1
    if len(xx2[0])==0:
        warnings.warn('FWHM could not be calculated properly; curve does not drop below half maximum')
        xx2 = 0
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


def select_ranges(plot_func, x_axis=None, *args, **kwargs):
    '''
    Select ranges from a plot

    Parameters
    ----------
    plot_func : python function producing a plot
        The function is required to have an argument show_external which will
        be used to show the figure within this function.
    x_axis : np.array
        If None as per default, the selected values in axis units will be returned.
        If specified, the bins along the specified axis will be returned.
    *args, **kwargs : to be passed on to plot_func

    Returns
    -------
    2 tuple of np.array
        The left and right bounds of the selected regions 
        (one array for left one for right).
        Depending on whether x_axis is specified or not this will be in axis
        units or in bins.

    '''
    fig, axs = plot_func(*args, **kwargs, show_external=True)

    selected_ranges = []
    clicks = []
    
    def onclick(event):
        if len(axs) == 1:
            if event.inaxes != axs:
                return
        else:
            if event.inaxes not in axs:
                return
        
        clicks.append(event.xdata)
        if len(clicks) == 2:
            x1, x2 = sorted(clicks)
            selected_ranges.append((x1, x2))
            axs[0].axvspan(x1, x2, color='orange', alpha=0.3)
            plt.draw()
            print(f"Selected x-range: ({x1:.2f}, {x2:.2f})")
            clicks.clear()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)

    X = np.array(selected_ranges)
    X.sort(axis=0)
    
    if x_axis is None: # return in axis units
        return X.T[0], X.T[1]
        
    else: # Convert to data bins
        left = np.array([np.argmin(np.abs(x_axis - X.T[0,i])) for i in range(len(X))])
        right = np.array([np.argmin(np.abs(x_axis - X.T[1,i])) for i in range(len(X))])
        return left, right


#%% Filedialog

def select_file(operation="open", file=None, title='Select file', 
              defaultextension=".h5", 
              filetypes=[('HDF5 dataset','*.h5')]):
    """
    Wrapper around `tkinter.askopenfilename`. 
    If a file is specified will return it, otherwise open a tkinter file
    selection dialog.

    Parameters
    ----------
    operation : str, optional
        Operation to choose from tkinter. Options are "open", "save".
        The default is "open".
    file : str, optional
        Option to skip the dialog by directly providing a file.
    title : str, optional
        Title for the window. The default is 'Open file'.
    defaultextension : str, optional
        Default file extension to look for. The default is ".h5".
    filetypes : list of 2-tuple of str, optional
        List of 2-tuples where the first element is a description and the
        second is a file extension. The default is [('HDF5 dataset','*.h5')].

    Returns
    -------
    file : str
        The file to open.

    """
    
    if file is None:
        root = tk.Tk()
        root.withdraw()
        if operation == "open":
            file = askopenfilename(title=title, defaultextension=defaultextension, 
                                   filetypes=filetypes)
        elif operation == "save":
            file = asksaveasfilename(title=title, defaultextension=defaultextension, 
                                     filetypes=filetypes)
            print("Saving at: " + file)
        else:
            raise AttributeError("Unrecognized file operation. Choose 'open' or 'save'.")
        root.destroy()
    
    if not os.path.isfile(file):
        raise FileNotFoundError("File " + file  + " does not exist.")
    
    return file


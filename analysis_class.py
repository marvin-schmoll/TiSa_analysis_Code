# -*- coding: utf-8 -*-
"""
Ti:Sa RABBITT Analysis Software
Version 1.0
01.08.2024

@author: Marvin Schmoll
marvin.schmoll@physik.uni-freiburg.de
"""


import h5py

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image
import matplotlib.cm as cm
import cmasher as cmr # makes better colormaps available, comment out if not installed
import scipy.signal
from scipy.optimize import curve_fit

import abel

import tkinter as tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from tqdm import tqdm


c = 2.99792458 * 10**8 # velocity of light [m/s]
h = 4.135667696        # planck constant [eV*fs]
omega_IR = 2.35        # [1/fs] (for 800nm)

E_IR = h / (2*np.pi) * omega_IR   # [eV]

ionization_energies = {'He+': 24.587, 'Ne+': 21.565, 'Ar+': 15.760, 'Kr+': 14.000, 'Xe+': 12.13,
                       'CH4+': 13.6, 'CH3+': 14.8, 'CH2+': 15.8, 'CH+': 22.9} # in eV



def normalized(array, normalization='max'):
    '''Shorthand for normalizing arrays.
        "Normalization" decides, if the maximum or the sum of all values is set to one.'''

    if normalization in ['max', 'maximum', 'Maximum']:
        return array / np.max(np.abs(array))

    if normalization in ['sum', 'Sum', 'int', 'integral', 'Integral']:
        return array / np.sum(array)

    raise ValueError('Specify normalization convention as "maximum" or "sum"')



class RABBITT_scan():
    
    def __init__(self):
        '''currently empty as functionality is tranferred to the class'''
        
        self.scan = self.inverted_scan = None                                   # collection of 2D images before and after Abel inversion
        self.speed_distributions = self.speed_distribution = None               # speed distributions obtained from angular integration of inverted images, single speed distribution integrated over array
        self.speed_distributions_jacobi = self.speed_distribution_jacobi = None # same, multiplied by jacobi determinant
        self.speed_axis = self.energies = None                                  # axes for the photoelectron spectrum, speed in samples, energy in eV
        self.min_energy, self.max_energy = 0, 20 # TODO: possibility to change  # energy limits in eV used for plotting
        self.times = None                                                       # time axis [fs]
 
      
    def _rainbow_colors(self, length, darken=1):
        '''creates a set of "length" colors from red to blue
            "darken">1 darkens colors for usage in plotting'''

        colors = cm.get_cmap('rainbow')(np.linspace(1,0,length))
        colors_sat = colors
        colors_sat[:,:3] = colors[:,:3] / darken   # darken colors for better visibility
        return colors_sat


    def _delay_axis(self, unit='n'):
        '''private subfunction used in all plotting functions using a delay axis to allow
            for different units on said axis'''

        if unit.lower() in {'n', 'step', 'steps', 'number'}:
            return np.arange(len(self.scan)), 'delay steps', True

        elif unit.lower() in {'s', 'fs', 'as', 'second', 'seconds', 'time', 'times', 't', 'delay'}:
            if self.times is None: # time scale has not jet been calculated
                self.time_scale() # calculate the time scale
            return self.times, 'delay [fs]', True

        else: raise ValueError('Given axis type not supported, try e.g. "step" or "time"')
        

    def _energy_axis(self, unit='n'):
        '''private subfunction to allow
            for different units on the energy axis'''

        if unit.lower() in {'n', 'step', 'steps', 'number', 'speed'}:
            return self.speed_axis, 'speed [pixels]', True

        elif unit.lower() in {'e', 'energy', 'ev', 'j'}:
            return self.energies, 'energy [eV]', False
        
        #TODO: one could implement a velocity axis as well

        else: raise ValueError('Given axis type not supported, try e.g. "speed" or "energy"')


    def read_scan_files(self):
        '''reads a selection of h5-files corresponding to a scan
            and averages them'''
    
        root = tk.Tk()
        root.withdraw()
        files = askopenfilenames(title='Select image files')
        bfile = askopenfilename(title='Select background file')
        root.destroy()
        
        if bfile: # only read if background was selected
            print('Reading background...')
            bf = h5py.File(bfile, 'r')
            bimage = np.array(bf['Images']).mean(axis=2)
        
        print('Reading shape...')       # get correct shape of single scan
        f = h5py.File(files[0], 'r')
        scan = np.zeros_like(np.array(f['Images']))
        n_files = len(files)
    
        for file in tqdm(files, desc='Reading scans'):
            f = h5py.File(file, 'r')
            scan += np.array(f['Images'])
       
        if bfile: self.scan = scan.T - bimage.T * n_files
        else:     self.scan = scan.T
    

    def plot_VMI_image(self, image, cmap='viridis', saving=False):
        '''plots a single VMI image'''
        
        plt.figure()
        plt.matshow(image, cmap=cmap)
        plt.colorbar()
        plt.clim(0,None)
        
        if saving is True or saving == "pdf":
            plt.savefig('vmi_image.pdf')
        elif  saving == "png":
            plt.savefig('vmi_image.png', dpi=300)
        
        plt.show()


    def save_scan_images_Daniel(self):
        '''saves the single images belonging to the scan as tsv-files
            (this is a legacy function for compatibility with Daniel script)'''
            
        root = tk.Tk()
        root.withdraw()
        path = askdirectory(title='Select folder to save TSV-files in')
        root.destroy()
        
        for i,image in enumerate(tqdm(self.scan)):
            np.savetxt(path+'/step'+str(i)+'.tsv', image.T, delimiter='\t', fmt='%d')


    def save_scan_images(self):
        '''Saves the raw VMI images of the scan'''
            
        filetypes = [('HDF5 dataset','*.h5'), ('Numpy array','*.npy')]
            
        root = tk.Tk()
        root.withdraw()
        path = asksaveasfilename(title='Save as', defaultextension=".h5", 
                                 filetypes=filetypes)
        root.destroy()    
        print("Saving at: " + path)
        
        if path.split(".")[-1] == "npy": # Save as numpy binary file
            np.save(path, self.scan)
        
        elif path.split(".")[-1] == "h5": # Save as h5 dataset
            with h5py.File(path, "w") as f:
                f.create_dataset("scan", data=self.scan)


    def read_scan_images(self):
        '''Reads h5 or npy files containing the raw VMI images of the scan'''
            
        filetypes = [('HDF5 dataset','*.h5'), ('Numpy array','*.npy')]
            
        root = tk.Tk()
        root.withdraw()
        path = askopenfilename(title='Open scan file containing raw VMI images', 
                               defaultextension=".h5", filetypes=filetypes)
        root.destroy()    
        
        if path.split(".")[-1] == "npy": # Read numpy binary file
            self.scan = np.load(path)
        
        elif path.split(".")[-1] == "h5": # Read from h5 dataset
            with h5py.File(path, "r") as f:
                self.scan = np.array(f['scan'])        


    def check_oscillation_preliminary(self):
        '''checks the oscillation of a (as of now hardcoded) region in the image'''
        
        intensities = np.zeros(len(self.scan))
        
        for i, image in enumerate(self.scan):        
            intensities[i] = image[920:1050,335:360].sum()
        
        plt.figure()
        plt.plot(intensities)
        plt.show()


    def perform_abel_inversion(self):
        '''performs an abel inversion of the individual VMI immages 
            to obtain the spped distributions'''
        
        if self.scan is None:
            message = "No scan loaded to perform Abel inversion on."
            raise AttributeError(message)
        
        self.inverted_scan = np.zeros((len(self.scan),1920,1199))
        self.speed_distributions = np.zeros((len(self.scan),600))
        
        origin = (998,610)  # TODO: hardcoded bad bad bad
        
        for i, VMI_image in tqdm(enumerate(self.scan), total=len(self.scan)):
            recon = abel.Transform(VMI_image, direction='inverse', method='rbasex',
                                   origin=origin, verbose=False)
            self.inverted_scan[i] = recon.transform
        
            speeds = abel.tools.vmi.angular_integration_3D(self.inverted_scan[i])
            self.speed_distributions[i] = speeds[1][:600]
 
        
    def energy_scale(self):
        
        if self.speed_distributions is None:
            message = "Perform Abel inversion first to get speed distribution."
            raise AttributeError(message)
        
        def velocity(n, a, b):   # n, b in [harm. orders]; a in [samples^2/harm. order]
            return np.sqrt(a * (n+b))   # output in [samples]
        
        self.speed_distribution = normalized(self.speed_distributions.sum(axis=0))
        peaks, properties = scipy.signal.find_peaks(self.speed_distribution, 
                                                    height=0.1, prominence=0.1, width=10)
        
        plt.figure(num='Speed distribution', clear=True)
        plt.plot(self.speed_distribution)
        plt.plot(peaks, self.speed_distribution[peaks], 'd')
        plt.xlabel('speed (samples)')
        plt.ylabel('intensity (normalized)')
        plt.show()
        
        nn = np.arange(len(peaks))
        popt, pcov = curve_fit(velocity, nn, peaks, p0=[1e4,1])
        plotrange = np.arange(-popt[1],len(peaks),0.01)
        print(popt)
        
        plt.figure(num='Speed curve-fit', clear=True)
        plt.plot(peaks, 'x')
        plt.plot(plotrange, velocity(plotrange, *popt))
        plt.ylabel('speed (samples)')
        plt.show()
        
        self.speed_axis = np.arange(len(self.speed_distribution))  # [samples]
        self.energies = self.speed_axis**2 * E_IR / popt[0]        # [eV]
        
        # Multiplying by Jacobi determinant for plotting of PES
        self.speed_distribution_jacobi = self.speed_distribution / self.speed_axis
        self.speed_distributions_jacobi = self.speed_distributions / self.speed_axis
    
    
    def time_scale(self, step):
        '''define the time axis given the steps size for the piezo in microns'''
        
        delta_t = step*1e-6 * 2 / c * 1e15   # step size in fs
        self.times = np.arange(0, len(self.scan)*delta_t, delta_t)
        
    
    def plot_RABBITT_trace(self, data_2D, fig_number=None, clabel='counts', cmap='jet', 
                           delay_unit='n', energy_unit='n', clim=None, saving=False, figsize=None):
        '''plots the RABBITT-trace as colormap;
            no interpolation between datapoints is used to show the real resolution'''

        x_axis, x_label, _ = self._delay_axis(delay_unit)
        y_axis, y_label, _ = self._energy_axis(energy_unit)
        _, ax = plt.subplots(num=fig_number, clear=True, figsize=figsize)

        # works with nonuniform x-axis
        im = matplotlib.image.NonUniformImage(ax, interpolation='nearest', cmap=cmap, clim=clim)
        im.set_data(x_axis, y_axis, data_2D.T)
        ax.images.append(im)
        ax.set_xlim(x_axis[0], x_axis[-1])
        ax.set_ylim(y_axis[0], y_axis[-1])
        ima = matplotlib.image.AxesImage(ax)
        if clim is None:
            ima.set_clim(np.min(data_2D), np.max(data_2D))
        else:
            ima.set_clim(clim)
        ima.set_cmap(cmap)
        plt.colorbar(mappable=ima, pad = .01, shrink = .81, aspect = 40).set_label(clabel)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        if saving is True or saving == "pdf":
            plt.savefig('trace.pdf')
        elif  saving == "png":
            plt.savefig('trace.png', dpi=300)
        
        plt.show()     

        
        
        
        
        
#%%

if __name__ == "__main__":

    hasi = RABBITT_scan()
    hasi.read_scan_files()
    hasi.perform_abel_inversion()
    hasi.energy_scale()
    hasi.time_scale(0.01)
    hasi.plot_RABBITT_trace(hasi.speed_distributions_jacobi, delay_unit='fs', energy_unit='eV')
        
        
        
        
        
        
        
        
        
        

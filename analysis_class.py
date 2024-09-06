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


c = 2.99792458 * 10**8  # velocity of light [m/s]
h = 4.135667696         # planck constant [eV*fs]
omega_IR = 2.35         # [1/fs] (for 800nm)
m_e = 5.68563 * 10**-12 # electron mass [eV/(m/s)^2]

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
    
    def __init__(self, name=None):
        '''currently empty as functionality is tranferred to the class'''
        
        self.name = name
        
        self.scan = self.inverted_scan = None                                   # collection of 2D images before and after Abel inversion
        self.speed_distributions = self.speed_distribution = None               # speed distributions obtained from angular integration of inverted images, single speed distribution integrated over array
        self.speed_distributions_jacobi = self.speed_distribution_jacobi = None # same, multiplied by jacobi determinant
        self.speed_axis = self.energies = self.velocity_axis = None             # axes for the photoelectron spectrum, speed in samples, energy in eV, velocity in m/s
        self.min_energy, self.max_energy = 0, 20 # TODO: possibility to change  # energy limits in eV used for plotting
        self.times = None                                                       # time axis [fs]
        
        self.data_norm = self.data_diff = None                                  # speed distributions normalized and speed distribution differences from average
        
        self.phase_by_energy = self.phase_by_energy_error = np.array([])        # oscillation phase and uncertainty
        self.depth_by_energy = self.depth_by_energy_error = np.array([])        # oscillation depth and uncertainty
        self.slope_by_energy = self.slope_by_energy_error = np.array([])        # slope for oscillation reconstruction and uncertainty
        self.contrast_by_energy = self.contrast_by_energy_error = np.array([])  # oscillation contrast and uncertainty
 
    
      
    def _rainbow_colors(self, length, darken=1):
        '''creates a set of "length" colors from red to blue
            "darken">1 darkens colors for usage in plotting'''

        colors = cm.get_cmap('rainbow')(np.linspace(1,0,length))
        colors_sat = colors
        colors_sat[:,:3] = colors[:,:3] / darken   # darken colors for better visibility
        return colors_sat
    
    
    
    def _prefix(self):
        '''changes the name like "name: " to create separate plots for each intance of the class'''
        if self.name is None or self.name == '':
            return ''
        else:
            return str(self.name) + ': '



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
        
        elif unit.lower() in {'v', 'velocity', 'm/s', 'km/s'}:
            return self.velocity_axis, 'velocity [km/s]', False

        else: raise ValueError('Given axis type not supported, try e.g. "speed" or "energy"')
    
    
    
    def set_energy_limit(self, limit=None, left_limit=None):
        '''allows to set an energy limit up to which structure is visible in the spectrum
            this will be used as axis limit in all plots'''

        if limit is None:
            self.max_energy = float(np.max(self.energies))
        else:
            self.max_energy = float(limit)
        assert isinstance(self.max_energy, float), "energy limit has to be float"

        if left_limit is None:
            self.min_energy = float(0)
        else:
            self.min_energy = float(left_limit)
        assert isinstance(self.min_energy, float), "left energy limit has to be float"



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
    


    def plot_VMI_image(self, image, cmap='viridis', saving=False, upper_clim=None):
        '''plots a single VMI image'''
        
        plt.matshow(image, cmap=cmap)
        plt.xlabel('pixels')
        plt.ylabel('pixels')
        plt.colorbar()
        plt.clim(0, upper_clim)
        
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
        '''perform curve fit to determine energy axis'''
        
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
        plt.xlabel('energy [eV]')
        plt.ylabel('speed (samples)')
        plt.show()
        
        self.speed_axis = np.arange(len(self.speed_distribution))   # [samples]
        self.energies = self.speed_axis**2 * E_IR / popt[0]         # [eV]
        self.velocity_axis = np.sqrt(2 * self.energies / m_e) / 1e3 # [km/s]
        
        # Multiplying by Jacobi determinant for plotting of PES
        self.speed_distribution_jacobi = self.speed_distribution / self.speed_axis
        self.speed_distributions_jacobi = self.speed_distributions / self.speed_axis
    
    
    
    def prepare_analysis(self):
        '''normalizes data in a way that is useful for the rabbitt-analysis'''
        
        # Normalize signal for each delay step
        self.data_norm = (self.speed_distributions_jacobi.T / np.nansum(self.speed_distributions_jacobi, axis=1)).T
        
        #
        self.data_diff = self.data_norm - normalized(np.nansum(self.data_norm, axis=0), 'sum')
    
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



    def plot_phase_diagram(self, indicator='points', show_amplitude=False, 
                           left=[], right=[], show_errors=False, saving=False):
        '''plots the phase by energy
            indicator ("points", "range", or "none") specifies which indication for sideband
            positions should be shown'''

        if len(self.phase_by_energy) == 0:   # "self.phase_by_energy" was never defined
            self.do_fourier_transform(plotting=False)

        fig, ax1 = plt.subplots(num=(self._prefix() + 'Phases'), clear=True)
        ax1.plot(self.energies, self.phase_by_energy, 'x-', color='k')
        ax1.tick_params(axis='both', which='major')
        ax1.set_xlabel('photoelectron energy [eV]')
        ax1.set_ylabel('phase [rad]')

        if show_amplitude ==  True:
            ax2 = ax1.twinx()
            ax2.plot(self.energies, self.depth_by_energy, 'x-', color='grey')
            ax2.tick_params(axis='both', which='major')
            ax2.set_ylabel('modul. amp. (a.u.)', color='grey')
            highest = np.nanmax(self.depth_by_energy)
            ax1.set_ylim([-1.8*np.pi, 1.05*np.pi])  # avoid overlap
            ax2.set_ylim([0, 2.5*highest])  # avoid overlap
            
        if show_errors == True:
            upper_bound = self.phase_by_energy + self.phase_by_energy_error
            lower_bound = self.phase_by_energy - self.phase_by_energy_error
            ax1.fill_between(self.energies, upper_bound, lower_bound, color='gray')

        # TODO: reactivate when harmonic selection is implemented
        #if indicator == 'points':   # draw points for the harmonic and sideband locations
        #    ax1.plot(self.energies[self.harmonics], self.phase_by_energy[self.harmonics], 'o', color='orange', label='HH')
        #    ax1.plot(self.energies[self.sidebands], self.phase_by_energy[self.sidebands], 'o', color='green', label='SB')
        #
        #if indicator == 'range':   # color points ascribed to each sideband in different colors
        #    colors = self._rainbow_colors(len(left), 1.0)   # spectral colormap from red to blue
        #    colors_sat = self._rainbow_colors(len(left), 1.3)   # spectral colormap from red to blue
        #    for i in range(len(left)):
        #        ax1.plot(self.energies[left[i]:right[i]], self.phase_by_energy[left[i]:right[i]],
        #                 'x-', label=self.SB_names[i], color=colors_sat[i])
        #        if show_amplitude  == True:
        #            ax2.plot(self.energies[left[i]:right[i]], self.depth_by_energy[left[i]:right[i]],
        #                     'x-', color=colors[i], alpha=0.5)

        plt.xlim([self.min_energy, self.max_energy])
        plt.legend(loc='upper right')
        fig.tight_layout()
        plt.savefig('favorite_plot.png', dpi=400)
        fig.show()


    def do_fourier_transform(self, plotting=True):
        '''does a fourier transform for each energy bin and extracts the phase of the 2-omega-component'''
        
        if self.data_diff is None:
            self.prepare_analysis()  # Calculate difference dataset expected by curve fit

        self.phase_by_energy = np.array([])  # oscillation phase
        self.depth_by_energy = np.array([])  # oscillation amplitude
        self.phase_by_energy_error = np.array([])  # oscillation phase
        self.depth_by_energy_error = np.array([])  # oscillation amplitude
        
        # Perform all the Fourier transforms
        fouriers = [np.fft.fft(single_line) for single_line in self.data_diff.T]
        fourier_map = np.abs(fouriers)
        fourier_phases = np.angle (fouriers)
        fourier_spectrum = np.nansum(fourier_map, axis=0)
        
        # Find oscillation frequency and extract phase there
        peak = np.argmax(fourier_spectrum[3:]) + 3
        self.phase_by_energy = -fourier_phases.T[peak]
        self.depth_by_energy = fourier_map[peak]

        # show corresponding plot
        if plotting == True:
            self.plot_phase_diagram(indicator='points',  show_amplitude=True)

        return self.phase_by_energy


    def do_cosine_fit(self, plotting=True, omega=False, integrate=1):
        '''does a cosine fit for each energy bin and extracts the phase of the 2-omega-component
            omega=True tries to fit the 1-omega-component instead'''
        
        if self.data_diff is None:
            self.prepare_analysis()  # Calculate difference dataset expected by curve fit

        self.phase_by_energy = np.array([])  # oscillation phase
        self.depth_by_energy = np.array([])  # oscillation amplitude
        self.slope_by_energy = np.array([])  # slope of the background
        self.phase_by_energy_error = np.array([])
        self.depth_by_energy_error = np.array([])
        self.slope_by_energy_error = np.array([])
        
        if omega is True:
            def cos(t, phi, a, b): # fittable cosine with linear background
                return a * np.cos(omega_IR * t - phi) + b * t
        else:
            def cos(t, phi, a, b): # fittable cosine with linear background
                return a * np.cos(2 * omega_IR * t - phi) + b * t

        if integrate == 1:
            for single_line in self.data_diff.T:
                
                try:
                    ### perform cosine fit ###
                    popt, pcov = scipy.optimize.curve_fit(cos, self.times, single_line)
                    perr = np.sqrt(np.diag(pcov))
                    print(popt)
        
                    ### write down phase parameters ###
                    if popt[1] > 0:
                        self.phase_by_energy = np.append(self.phase_by_energy, (popt[0]+np.pi)%(2*np.pi)-np.pi)
                    else:
                        self.phase_by_energy = np.append(self.phase_by_energy, (popt[0])%(2*np.pi)-np.pi)
                    self.phase_by_energy_error = np.append(self.phase_by_energy_error, perr[0])
        
                    self.depth_by_energy = np.append(self.depth_by_energy, np.abs(popt[1]))
                    self.depth_by_energy_error = np.append(self.depth_by_energy_error, perr[1])
        
                    self.slope_by_energy = np.append(self.slope_by_energy, popt[2])
                    self.slope_by_energy_error = np.append(self.slope_by_energy_error, perr[2])
                
                except ValueError:
                    self.phase_by_energy = np.append(self.phase_by_energy, np.nan)
                    self.phase_by_energy_error = np.append(self.phase_by_energy_error, np.nan)
        
                    self.depth_by_energy = np.append(self.depth_by_energy, np.nan)
                    self.depth_by_energy_error = np.append(self.depth_by_energy_error, np.nan)
        
                    self.slope_by_energy = np.append(self.slope_by_energy, np.nan)
                    self.slope_by_energy_error = np.append(self.slope_by_energy_error, np.nan)
                
        if integrate > 1:
            for i in range(int(len(self.data_diff.T)/integrate)):
                single_line = (self.data_diff.T[i*integrate:(i+1)*integrate]).sum(axis=0)
                
                try:
                    ### perform cosine fit ###
                    popt, pcov = scipy.optimize.curve_fit(cos, self.times, single_line)
                    perr = np.sqrt(np.diag(pcov))
                    print(popt)
        
                    ### write down phase parameters ###
                    if popt[1] > 0:
                        self.phase_by_energy = np.append(self.phase_by_energy, (popt[0]+np.pi)%(2*np.pi)-np.pi)
                    else:
                        self.phase_by_energy = np.append(self.phase_by_energy, (popt[0])%(2*np.pi)-np.pi)
                    self.phase_by_energy_error = np.append(self.phase_by_energy_error, perr[0])
        
                    self.depth_by_energy = np.append(self.depth_by_energy, np.abs(popt[1]))
                    self.depth_by_energy_error = np.append(self.depth_by_energy_error, perr[1])
        
                    self.slope_by_energy = np.append(self.slope_by_energy, popt[2])
                    self.slope_by_energy_error = np.append(self.slope_by_energy_error, perr[2])
                    
                except ValueError:
                    self.phase_by_energy = np.append(self.phase_by_energy, np.nan)
                    self.phase_by_energy_error = np.append(self.phase_by_energy_error, np.nan)
        
                    self.depth_by_energy = np.append(self.depth_by_energy, np.nan)
                    self.depth_by_energy_error = np.append(self.depth_by_energy_error, np.nan)
        
                    self.slope_by_energy = np.append(self.slope_by_energy, np.nan)
                    self.slope_by_energy_error = np.append(self.slope_by_energy_error, np.nan)
                    
        # TODO: reactivate when ready
        #self.contrast_by_energy = self.depth_by_energy / self.energy_graph_norm
        #self.contrast_by_energy_error = self.depth_by_energy_error / self.energy_graph_norm

        # show corresponding plot
        if plotting == True:
            self.plot_phase_diagram(indicator='points', show_amplitude=True, show_errors=True)

        return self.phase_by_energy

        
        
        
        
        
#%%

if __name__ == "__main__":

    hasi = RABBITT_scan()
    hasi.read_scan_files()
    hasi.perform_abel_inversion()
    hasi.energy_scale()
    hasi.time_scale(0.01)
    
    hasi.plot_RABBITT_trace(hasi.speed_distributions, delay_unit='fs', energy_unit='v')
    hasi.plot_RABBITT_trace(hasi.speed_distributions_jacobi, delay_unit='fs', energy_unit='eV')
        
        
        
        
        
        
        
        
        
        

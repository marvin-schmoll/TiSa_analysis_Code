# -*- coding: utf-8 -*-
"""
Ti:Sa VMI Analysis Software
(based on existing Code for K04) 

@author: Marvin Schmoll
marvin.schmoll@physik.uni-freiburg.de
"""


import h5py

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
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

ionization_energies = {'He': 24.587, 'Ne': 21.565, 'Ar': 15.760, 'Kr': 14.000, 'Xe': 12.13,
                       'CH4': 13.6, 'CH3': 14.8, 'CH2': 15.8, 'CH': 22.9} # in eV



def normalized(array, normalization='max'):
    '''Shorthand for normalizing arrays.
        "Normalization" decides, if the maximum or the sum of all values is set to one.'''

    if normalization in ['max', 'maximum', 'Maximum']:
        return array / np.nanmax(np.abs(array))

    if normalization in ['sum', 'Sum', 'int', 'integral', 'Integral']:
        return array / np.nansum(array)

    raise ValueError('Specify normalization convention as "maximum" or "sum"')




class RABBITT_scan():
    
    def __init__(self, gas, name=None):
        '''currently empty as functionality is tranferred to the class'''
        
        self.gas, self.Ip = gas, ionization_energies[gas]                       # gas used in the VMI, its ionization potential
        self.name = name
        
        self.scan = self.inverted_scan = None                                   # collection of 2D images before and after Abel inversion
        self.speed_distributions = self.speed_distribution = None               # speed distributions obtained from angular integration of inverted images, single speed distribution integrated over array
        self.speed_distributions_jacobi = self.speed_distribution_jacobi = None # same, multiplied by jacobi determinant
        self.speed_distribution_norm = None                                     # normalized speed distribution (integral is 1) 
        self.speed_axis = self.energies = self.velocity_axis = None             # axes for the photoelectron spectrum, speed in samples, energy in eV, velocity in m/s
        self.min_energy, self.max_energy = 0, 20                                # energy limits in eV used for plotting
        self.times = None                                                       # time axis [fs]
        
        self.harmonics = self.sidebands = None                                  # pixel positions of HH/SB-peaks
        self.n_harmonics = self.n_sidebands = None                              # order of HH/SB
        self.left = self.right = None                                           # left and right edges of sidebands
        
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



    def read_scan_files(self, files=None, bfile=''):
        """
        Reads a selection of h5-files corresponding to a scan and averages them.

        Parameters
        ----------
        files : str or list of str, optional
            Specify the file or files containing the scan data. 
            The default is None, which opens a dialog to select the files.
        bfile : str, optional
            Specify the file containing a background scan. 
            Only relevant when 'files' is not None.
            The default is '', which disables background subtraction.

        Returns
        -------
        None.

        """
        if type(files) is str:
            files = [files]
        
        if files is None:
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
        """
        Checks the oscillation of a (as of now hardcoded) region in the image.
        
        Legacy function for quick checks of newly acquired data without Abel inversion.
        

        Returns
        -------
        None.

        """
        
        intensities = np.zeros(len(self.scan))
        
        for i, image in enumerate(self.scan):        
            intensities[i] = image[920:1050,335:360].sum()
        
        plt.figure()
        plt.plot(intensities)
        plt.show()



    def perform_abel_inversion(self, origin=(998,610)):
        """
        Performs an Abel inversion of the individual VMI images to obtain the speed distributions.
        
        Uses the PyAbel-implementation of the rbasex-method.

        Parameters
        ----------
        origin : 2-tuple of int, optional
            Image center in pixels. The default is (998,610).

        Returns
        -------
        None.

        """
        
        if self.scan is None:
            message = "No scan loaded to perform Abel inversion on."
            raise AttributeError(message)
        
        self.inverted_scan = np.zeros((len(self.scan),1920,1199))
        self.speed_distributions = np.zeros((len(self.scan),600))
        
        for i, VMI_image in tqdm(enumerate(self.scan), total=len(self.scan)):
            recon = abel.Transform(VMI_image, direction='inverse', method='rbasex',
                                   origin=origin, verbose=False)
            self.inverted_scan[i] = recon.transform
        
            speeds = abel.tools.vmi.angular_integration_3D(self.inverted_scan[i])
            self.speed_distributions[i] = speeds[1][:600]
 
    
        
    def energy_scale(self, max_pixel=550):
        """
        Performs curve fit to determine energy axis.
        
        Using the known ionization potential, the harmonics and sidebands are assigned.

        Parameters
        ----------
        max_pixel : int, optional
            Pixel up to which peaks can will be recognized as harmonics/sidebands. 
            The default is 550.

        Returns
        -------
        None.

        """
        
        if self.speed_distributions is None:
            message = "Perform Abel inversion first to get speed distribution."
            raise AttributeError(message)
        
        def velocity(n, a, b):   # n, b in [harm. orders]; a in [samples^2/harm. order]
            return np.sqrt(a * (n+b))   # output in [samples]
        
        self.speed_distribution = normalized(self.speed_distributions.sum(axis=0))
        peaks, properties = scipy.signal.find_peaks(self.speed_distribution[0:max_pixel], 
                                                    height=0.1, prominence=0.1, width=5)
        
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
        
        # finding and assigning harmonics and sidebands
        lowest_peak_energy = self.energies[peaks][0] + self.Ip
        lowest_peak_order = np.argmin(np.abs(lowest_peak_energy - E_IR*np.arange(20)))
        peak_orders = lowest_peak_order + np.arange(len(peaks))
        off = lowest_peak_order % 2
        self.harmonics = peaks[(1-off)::2]
        self.n_harmonics = peak_orders[(1-off)::2]
        self.harmonic_names = np.array(['H' + H for H in np.array(self.n_harmonics, dtype=str)])
        self.sidebands = peaks[off::2]
        self.n_sidebands = peak_orders[off::2]
        self.sideband_names = np.array(['SB' + S for S in np.array(self.n_sidebands, dtype=str)])
        
    
    
    def time_scale(self, step):
        """
        Define the time axis given the steps size for the piezo in microns.

        Parameters
        ----------
        step : float
            Step size used by the piezo in µm.

        Returns
        -------
        None.

        """
        
        delta_t = step*1e-6 * 2 / c * 1e15   # step size in fs
        self.times = np.arange(0, len(self.scan)*delta_t, delta_t)
    
    
    
    def prepare_analysis(self):
        '''normalizes data in a way that is useful for the rabbitt-analysis'''
        
        self.speed_distribution_norm = normalized(self.speed_distribution_jacobi, 'sum')
        
        # Normalize signal for each delay step
        self.data_norm = (self.speed_distributions_jacobi.T / np.nansum(self.speed_distributions_jacobi, axis=1)).T
        
        # Calculate changes from average signal
        self.data_diff = self.data_norm - normalized(np.nansum(self.data_norm, axis=0), 'sum')
    
    
    
<<<<<<< HEAD
    def plot_oscillation(self, oscillation, labels=None, fig_number=None, delay_unit='fs', #TODO: Cleanup
=======
    def plot_oscillation(oscillation, labels=None, fig_number=None, delay_unit='fs', #TODO: Cleanup
>>>>>>> refs/remotes/origin/main
                         size_hor=10, size_ver=8, saving=False):
        '''plots multiple oscillations in seperate subplots with line coloring showing their energies,
            each having a seperate axis indicating their relative intensity
            if only one is given, the function also works'''

        x_axis, x_label, x_linarity = hasi._delay_axis(delay_unit)
        plt.figure(num=fig_number, clear=True, figsize=(size_hor, size_ver))


        # plot multiple oscillations in one figure
        if len(np.shape(oscillation)) == 2:
            gs = gridspec.GridSpec(len(oscillation), 1)
            colors = hasi._rainbow_colors(len(oscillation), 1.3) # spectral colormap from red to blue
            for i in range(len(oscillation)):
                if i==0:
                    ax0 = plt.subplot(gs[i])

                    pl, = ax0.plot(x_axis, oscillation[len(oscillation)-1-i]*1e3, 'x-', 
                                   lw=0.8, ms=6, color=colors[len(oscillation)-1-i])
                    ax0.plot([x_axis[0],x_axis[-1]], [0,0], color='grey', lw=0.5)
                    axl=ax0
                else:
                    axi = plt.subplot(gs[i], sharex=axl)

                    pl, = axi.plot(x_axis, oscillation[len(oscillation)-1-i]*1e3, 'x-', 
                                   lw=0.8, ms=6, color=colors[len(oscillation)-1-i])
                    axi.plot([x_axis[0],x_axis[-1]], [0,0], color='grey', lw=0.5)

                    yticks = axi.yaxis.get_major_ticks()
                    yticks[-1].label1.set_visible(False)
                    if i != len(oscillation) - 1:
                        axi.tick_params(axis='x', labelbottom=False)
                    axl=axi

                if i==int(len(oscillation-1)/2): # put ylabel only on the middle plot
                    plt.ylabel('count difference (a.u.) \n')

                plt.grid(axis='x')
                plt.legend([labels[len(oscillation)-1-i]], frameon=False, loc='upper left',
                           bbox_to_anchor=(0.07-0.01*len(oscillation),1.03)) # change label position here !!

            plt.setp(ax0.get_xticklabels(), visible=False)
            plt.subplots_adjust(hspace=.0)

        # plot just one oscillation
        if len(np.shape(oscillation)) == 1:
            plt.plot(x_axis, oscillation/np.max(np.abs(oscillation)), 'x-', color='b')

        plt.xlabel(x_label)
        plt.xlim([x_axis[0], x_axis[-1]])

        if saving: plt.savefig('rabbitt_oscillation.pdf', dpi=400)
        plt.show()        
    
       
    
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
        ax.add_image(im)
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

        if indicator == 'points':   # draw points for the harmonic and sideband locations
            ax1.plot(self.energies[self.harmonics], self.phase_by_energy[self.harmonics], 'o', color='orange', label='HH')
            ax1.plot(self.energies[self.sidebands], self.phase_by_energy[self.sidebands], 'o', color='green', label='SB')
        
        if indicator == 'range':   # color points ascribed to each sideband in different colors
            colors = self._rainbow_colors(len(left), 1.0)   # spectral colormap from red to blue
            colors_sat = self._rainbow_colors(len(left), 1.3)   # spectral colormap from red to blue
            for i in range(len(left)):
                ax1.plot(self.energies[left[i]:right[i]], self.phase_by_energy[left[i]:right[i]],
                         'x-', label=self.sideband_names[i], color=colors_sat[i])
                if show_amplitude  == True:
                    ax2.plot(self.energies[left[i]:right[i]], self.depth_by_energy[left[i]:right[i]],
                             'x-', color=colors[i], alpha=0.5)

        plt.xlim([self.min_energy, self.max_energy])
        ax1.legend(loc='upper right')
        fig.tight_layout()
        if saving: plt.savefig('favorite_plot.png', dpi=400)
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
                    
        self.contrast_by_energy = self.depth_by_energy / self.speed_distribution_norm
        self.contrast_by_energy_error = self.depth_by_energy_error / self.speed_distribution_norm

        # show corresponding plot
        if plotting == True:
            self.plot_phase_diagram(indicator='points', show_amplitude=True, show_errors=True)

        return self.phase_by_energy

        
        
    def calculate_sideband_ranges(self, extent=None):
        """
        Calculates the positions and ranges of the sideband oscillations in terms of their modulation amplitude.

        Parameters
        ----------
        extent : int, optional
            If specified, the corresponding number of bins towards each side is used instead of the fitted ranges.

        Raises
        ------
        AttributeError
            If the modulation has not yet been characterized.

        Returns
        -------
        np.array
            Left bounds of the sidebands.
        np.array
            Right bounds of the sidebands.

        """
        '''calculates the FWHM-ranges of the sideband oscillations in terms of their modulation amplitude
            if extent (integer) is specified, the corresponding number of bins towards each side is used instead'''

        if len(self.phase_by_energy) == 0:   # "self.phase_by_energy" was never defined
            message = "Perform cosine fit or fourier transform to obtain the oscillation amplitude."
            raise AttributeError(message)
        
        # find maxima in the modulation amplitude of the oscillations
        peaks, properties = scipy.signal.find_peaks(normalized(self.depth_by_energy),  #TODO: maybe shoothing makes this more robust (?)
                                                    height=0.25, width=10, rel_height=0.75)
        
        off = int(np.abs(peaks[0] - self.sidebands[0]) > np.abs(peaks[1] - self.sidebands[0]))
        self.sidebands = peaks[off::2]

        if extent is None: # use FWHM ranges for the sidebands
            left_ips = np.array(np.ceil(properties['left_ips']), dtype=int)
            right_ips = np.array(np.ceil(properties['right_ips']), dtype=int)
            self.left = left_ips[off::2]
            self.right = right_ips[off::2]

        else:  # use specified ranges
            self.left  = self.sidebands - extent
            self.right = self.sidebands + extent + 1

        #TODO: this could eventually be used in nice plots
        cutted = np.split(self.data_diff, np.sort((self.left,self.right), axis=None), axis=1)[1::2]
        self.SB_oscillation = np.array([np.sum(cutted[i], axis=1) for i in range(len(cutted))])
        #
        #self.SB = np.array(self.lowest_harmonic + np.linspace(1, 2*len(self.sidebands)-1, len(self.sidebands)), dtype=int)
        #self.SB_names = np.array(['SB' + S for S in np.array(self.SB, dtype=str)])
        
        self.plot_phase_diagram('range', True, self.left, self.right)

        return self.left, self.right        



        
#%%

if __name__ == "__main__":

    hasi = RABBITT_scan('Ar')
    hasi.read_scan_files()
    hasi.perform_abel_inversion()
    hasi.energy_scale()
    hasi.time_scale(0.01)
    
    hasi.plot_RABBITT_trace(hasi.speed_distributions, delay_unit='fs', energy_unit='v')
    hasi.plot_RABBITT_trace(hasi.speed_distributions_jacobi, delay_unit='fs', energy_unit='eV')
    
    hasi.do_cosine_fit(plotting=False)
    hasi.calculate_sideband_ranges()
    hasi.plot_oscillation(hasi.SB_oscillation, hasi.sideband_names, hasi._prefix() + 'Sideband Oscillation')    
        

    
        
        

# -*- coding: utf-8 -*-
"""
Ti:Sa VMI Analysis Software
(based on my older Code for K04) 

@author: Marvin Schmoll
marvin.schmoll@physik.uni-freiburg.de
"""


import h5py

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.image
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import cmasher as cmr # makes better colormaps available, comment out if not installed
import scipy.signal
from scipy.optimize import curve_fit
import warnings
from enum import Enum

import abel

import tkinter as tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename
from tqdm import tqdm

import utility_library as util
from utility_library import normalized


c = 2.99792458 * 10**8  # velocity of light [m/s]
h = 4.135667696         # planck constant [eV*fs]
omega_IR = 2.35         # [rad/fs] (for 800nm)
lambda_IR = 800e-9      # [m]
m_e = 5.68563 * 10**-12 # electron mass [eV/(m/s)^2]

CEP_factor = 9.6406e-4  # calibration factor wedge distance to CEP distance

E_IR = h / (2*np.pi) * omega_IR   # [eV]

default_origin = (967, 607)  # Change (!) here if VMI camera was moved


def plot_VMI_image(image, cmap='viridis', saving=False, 
                   lower_clim=None, upper_clim=None, logscale=False):
    '''plots a single VMI image'''
    
    if logscale:
        image = np.where(image < 0.1, np.ones_like(image)*0.1, image)
        plt.matshow(image, norm=LogNorm(), cmap=cmap)
    else:
        plt.matshow(image, cmap=cmap)
    plt.xlabel('pixels')
    plt.ylabel('pixels')
    plt.colorbar()
    plt.clim(lower_clim, upper_clim)
    
    if saving is True or saving == "pdf":
        plt.savefig('vmi_image.pdf')
    elif  saving == "png":
        plt.savefig('vmi_image.png', dpi=300)
    
    plt.show()


def vmi_radial_intensity(kind, IM, origin=None, dr=1, dt=None, 
                         theta_low=-np.pi, theta_high=np.pi):
    """
    Calculate the one-dimensional radial intensity profile by angular
    integration or averaging of the image, treated either as a two-dimensional
    distribution or as a central slice of a cylindrically symmetric
    three-dimensional distribution.

    Parameters
    ----------
    kind : str
        operation to perform:

        ``'int2D'``:
            integration in 2D over polar angles
        ``'int3D'``:
            integration in 3D over solid angles
        ``'avg2D'``:
            averaging in 2D over polar angles
        ``'avg3D'``:
            averaging in 3D over solid angles

    IM : 2D numpy.array
        the image data

    origin : tuple of float or None
        image origin in the (row, column) format. If ``None``, the geometric
        center of the image (``rows // 2, cols // 2``) is used.

    dr : float
        radial grid spacing in pixels (default 1). ``dr=0.5`` may reduce pixel
        granularity of the radial profile.

    dt : float or None
        angular grid spacing in radians.
        If ``None``, the number of theta values will be set to largest
        dimension (the height or the width) of the image, which should
        typically ensure good sampling.
    
    theta_low : float
        angle to start integration (radians).
        The angle range is parametrized from -pi to +pi.
        Default is -pi, which corresponds to no lower limit.
        
    theta_high : float
        angle to end integration (radians).
        The angle range is parametrized from -pi to +pi.
        Default is +pi, which corresponds to no upper limit.

    Returns
    -------
    r : 1D numpy.array
        radial coordinates

    intensity : 1D numpy.array
        intensity profile as a function of the radial coordinate
        
    Notes
    -----
    This is a clone of the abel.tools.vmi.radial_intensity function from thy 
    PyAbel library with added capability to only integrate a slice of the image
    """
    polarIM, R, T = abel.tools.polar.reproject_image_into_polar(IM, origin, dr=dr, dt=dt)
    # apply necessary Jacobian/normalization
    if kind == 'int2D':
        polarIM *= R
    elif kind == 'int3D':
        polarIM *= np.pi * R**2 * np.abs(np.sin(T))
    elif kind == 'avg2D':
        polarIM /= 2 * np.pi
    elif kind == 'avg3D':
        polarIM *= np.abs(np.sin(T)) / 4
    else:
        raise ValueError('Incorrect kind={}'.format(kind))

    # integrate over theta
    dt = T[0, 1] - T[0, 0]  # get the actual number, if dt=None was passed
    mask = np.logical_and(theta_low < T, T < theta_high)
    intensity = polarIM.sum(axis=1, where=mask) * dt

    return R[:, 0], intensity



class RABBITT_scan():
    
    def __init__(self, gas, name=None):
        '''currently empty as functionality is tranferred to the class'''
        
        self.types = Enum('scan_type', [('NONE', None), ('DELAY', 0), ('CEP', 1)])
        self.scan_type = self.types.NONE                                        # type of scan performed
        
        self.gas, self.Ip = gas, util.ionization_energies[gas]                  # gas used in the VMI, its ionization potential
        self.name = name
        
        self.scan = self.inverted_scan = None                                   # collection of 2D images before and after Abel inversion
        self.speed_distributions = self.speed_distribution = None               # speed distributions obtained from angular integration of inverted images, single speed distribution integrated over array
        self.speed_distributions_jacobi = self.speed_distribution_jacobi = None # same, multiplied by jacobi determinant
        self.speed_distribution_norm = None                                     # normalized speed distribution (integral is 1) 
        self.speed_axis = self.energies = self.velocity_axis = None             # axes for the photoelectron spectrum, speed in samples, energy in eV, velocity in m/s
        self.min_energy, self.max_energy = 0, 20                                # energy limits in eV used for plotting
        self.times = self.angles = self.distances = None                        # x axis [fs], [rad] and [mm] of 800nm
        self.nsteps = None                                                      # number of delay steps
        
        self.harmonics = self.sidebands = None                                  # pixel positions of HH/SB-peaks
        self.n_harmonics = self.n_sidebands = None                              # order of HH/SB
        self.left = self.right = None                                           # left and right edges of sidebands
        self.HH_oscillation = self.SB_oscillation = None                        # signal oscillation averaged over each HH/SB
        
        self.data_norm = self.data_diff = None                                  # speed distributions normalized and speed distribution differences from average
        
        self.phase_by_energy = self.phase_by_energy_error = np.array([])        # oscillation phase and uncertainty
        self.depth_by_energy = self.depth_by_energy_error = np.array([])        # oscillation depth and uncertainty
        self.slope_by_energy = self.slope_by_energy_error = np.array([])        # slope for oscillation reconstruction and uncertainty
        self.contrast_by_energy = self.contrast_by_energy_error = np.array([])  # oscillation contrast and uncertainty
 
    
      
    def _prefix(self):
        '''changes the name like "name: " to create separate plots for each intance of the class'''
        if self.name is None or self.name == '':
            return ''
        else:
            return str(self.name) + ': '
    
    
    def _legend_name(self, order, pre='SB'):
        '''returns names as 'SB14' for plot legends'''
        if type(order) in (int, float):
            return pre + str(np.round(order, 1))
        elif type(order) in (np.ndarray, list, tuple):
            return ['SB' + str(np.round(o, 1)) for o in order]


    def _phase_axis(self, unit='n'):
        '''Private subfunction used in all plotting functions using a delay axis
        to allow for different units on said axis.'''
    
        # dispatch table:
        # unit sets → (attribute_name, (delay label, CEP label), flag, warning message)
        axis_map = {
            ('n', 'step', 'steps', 'number'): 
                (None, 'steps', True, None),
            ('s', 'fs', 'as', 'second', 'seconds', 'time', 'times', 't', 'delay'):
                ('times', ('delay [fs]', 'CEP delay [fs]'), True, 'Time scale not yet calculated'),
            ('rad', 'mrad', 'phase', 'phi'):
                ('angles', ('phase delay [rad]', 'CEP [rad]'), True, 'Angle scale not yet calculated'),
            ('m', 'mm', 'um', 'x', 'dist', 'distance'):
                ('distances', ('stage distance [mm]', 'wedge distance [mm]'), True, 'Distance scale not yet calculated'),
        }
    
        for keys, (attr_name, labels, flag, warn_msg) in axis_map.items():
            if unit.lower() in keys:
                if attr_name is None:  # step axis
                    return np.arange(self.nsteps), labels, flag
    
                axis = getattr(self, attr_name)
                if axis is None:
                    warnings.warn(f'{warn_msg}, using steps for the axis instead.')
                    return self._phase_axis('n')
                return axis, labels[self.scan_type.value], flag
    
        raise ValueError(
            f'Given axis type "{unit}" not supported, try e.g. "step" or "time"'
        )

        
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



    def read_scan_files(self, files=None, bfile='', use_steps=slice(None)):
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
            The default is '', which uses the internal background image saved 
            in the h5 file if available for subtraction.
        use_steps : slice
            Specify which delay steps include. Default is to use all.

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
        if not bfile: bimage = np.zeros_like(np.array(f['Background Image']))
        n_files = len(files)
    
        for file in tqdm(files, desc='Reading scans'):
            f = h5py.File(file, 'r')
            scan += np.array(f['Images'])
            bimage += np.array(f['Background Image'])
       
        if bfile: self.scan = scan.T - bimage.T * n_files
        else:     self.scan = scan.T - bimage.T
        
        self.scan = self.scan[use_steps]
        self.nsteps = len(self.scan)



    def save_scan_images_Daniel(self):
        '''saves the single images belonging to the scan as tsv-files
            (this is a legacy function for compatibility with Daniel script)'''
            
        root = tk.Tk()
        root.withdraw()
        path = askdirectory(title='Select folder to save TSV-files in')
        root.destroy()
        
        for i,image in enumerate(tqdm(self.scan)):
            np.savetxt(path+'/step'+str(i)+'.tsv', image.T, delimiter='\t', fmt='%d')



    def save_scan_images(self, include_inverted=False):
        '''Saves the raw or inverted VMI images of the scan'''
            
        filetypes = [('HDF5 dataset','*.h5'), ('Numpy array','*.npy')]
            
        root = tk.Tk()
        root.withdraw()
        path = asksaveasfilename(title='Save as', defaultextension=".h5", 
                                 filetypes=filetypes)
        root.destroy()    
        print("Saving at: " + path)
        
        if path.split(".")[-1] == "npy": # Save as numpy binary file
            if include_inverted: np.save(path, self.scan)
            else:                np.save(path, self.inverted_scan)
        
        elif path.split(".")[-1] == "h5": # Save as h5 dataset
            with h5py.File(path, "w") as f:
                f.create_dataset("scan", data=self.scan)
                if include_inverted:
                    f.create_dataset("inverted_scan", data=self.inverted_scan)
                    f.create_dataset("speed_distributions", data=self.speed_distributions)



    def read_scan_images(self, files=None):
        '''Reads h5 or npy files containing the raw VMI images of the scan'''
        
        if type(files) is str:
            files = [files]
        
        if files is None:
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
        
        self.nsteps = len(self.scan)
    
    
    
    def clone_image_half(self, side, origin=default_origin):
        """
        Function to delete half of the image 
        and replace it by a mirrored copy of the other image half.

        Parameters
        ----------
        side : str
            Chose which half of the image ("top", "bottom", "left", "right")
            to be kept. The opposite half will be deleted.
        origin : 2-tuple of int, optional
            Image center in pixels. The default can be set globally.

        Returns
        -------
        None.

        """
        
        dims = self.scan.shape
        
        if side == 'top': # upper side on camera, right in numpy plotting
            self.scan[:,:,:origin[1]] = 0
            i = min(origin[1], dims[2]-origin[1])
            self.scan[:,:,origin[1]-i:origin[1]] = np.flip(self.scan[:,:,origin[1]:origin[1]+i], axis=2)
            
        elif side == 'bottom': # lower side on camera, left in numpy plotting
            self.scan[:,:,origin[1]:] = 0
            i = min(origin[1], dims[2]-origin[1])
            self.scan[:,:,origin[1]:origin[1]+i] = np.flip(self.scan[:,:,origin[1]-i:origin[1]], axis=2)
            
        elif side == 'right': # right side on camera, lower in numpy plotting
            self.scan[:,origin[0]:,:] = 0
            i = min(origin[0], dims[1]-origin[0])
            self.scan[:,origin[0]:origin[0]+i,:] = np.flip(self.scan[:,origin[0]-i:origin[0],:], axis=1)
        
        elif side == 'left': # left side on camera, upper in numpy plotting
            self.scan[:,:origin[0],:] = 0
            i = min(origin[0], dims[1]-origin[0])
            self.scan[:,origin[0]-i:origin[0],:] = np.flip(self.scan[:,origin[0]:origin[0]+i,:], axis=1)
        
        else:
            raise ValueError("Side must be one of 'left', 'right', 'top', 'bottom'")



    def perform_abel_inversion(self, origin=default_origin,
                               theta=(-np.pi,+np.pi), order=6, odd_orders=True):
        """
        Performs an Abel inversion of the individual VMI images to obtain the speed distributions.
        
        Uses the PyAbel-implementation of the rbasex-method.
    
        Parameters
        ----------
        origin : 2-tuple of int, optional
            Image center in pixels. The default can be set globally.
            
        theta : 2-tuple of float
            Angle range for the integration (radians).
            Angles are parametrized from -pi to +pi.
            Default is (-np.pi, +np.pi), which corresponds to the full range.
            
        order : int
            Highest angular order for rbasex evaluation, ≥ 0 (by default, 6). 
            Working with very high orders (≳ 15) can result in excessive noise,
            especially at small radii and for narrow peaks.
        
        odd_orders : bool
            Include odd angular orders (by default is True)

        Returns
        -------
        None.
    
        """
        
        if self.scan is None:
            message = "No scan loaded to perform Abel inversion on."
            raise AttributeError(message)
        
        self.inverted_scan = np.zeros((self.nsteps,1920,1200))
        self.speed_distributions = np.zeros((self.nsteps,600))
        
        for i, VMI_image in tqdm(enumerate(self.scan), total=self.nsteps):
            recon = abel.rbasex.rbasex_transform(self.scan[i].T, origin=origin[::-1], 
                                                     order=order, odd=odd_orders)
            self.inverted_scan[i] = recon[0].T
        
            #speeds = abel.tools.vmi.angular_integration_3D(self.inverted_scan[i])
            speeds = vmi_radial_intensity('int3D', self.inverted_scan[i], origin=origin,
                                          theta_low=theta[0], theta_high=theta[1])
            self.speed_distributions[i] = speeds[1][:600]
            self.speed_distribution = normalized(self.speed_distributions.sum(axis=0))



    def read_inverted_images(self, read_raw=True):
        """
        Reads h5 or npy files containing the inverted VMI images of the scan.

        Parameters
        ----------
        read_raw : bool, optional
            Choose wether to read the raw (uninverted) images as well.
            Not doing so will speed up the process and use less RAM.
            The option gets ignored for .npy-format which does not save the raw data.
            Default is False.

        Returns
        -------
        None.

        """
        filetypes = [('HDF5 dataset','*.h5'), ('Numpy array','*.npy')]
            
        root = tk.Tk()
        root.withdraw()
        path = askopenfilename(title='Open scan file containing inverted VMI images', 
                               defaultextension=".h5", filetypes=filetypes)
        root.destroy()    
        
        if path.split(".")[-1] == "npy": # Read numpy binary file
            self.inverted_scan = np.load(path)
            self.speed_distributions = np.zeros((self.nsteps,600))
            for i in range(self.nsteps):
                speeds = abel.tools.vmi.angular_integration_3D(self.inverted_scan[i])
                self.speed_distributions[i] = speeds[1][:600]
        
        elif path.split(".")[-1] == "h5": # Read from h5 dataset
            with h5py.File(path, "r") as f:
                if read_raw:
                    self.scan = np.array(f['scan'])
                self.inverted_scan = np.array(f['inverted_scan'])
                self.speed_distributions = np.array(f['speed_distributions'])
        
        self.nsteps = len(self.inverted_scan)
    
        
    
    def energy_scale(self, max_pixel=550, peak_distance=2/3,
                     height=0.1, prominence=0.1, width=5):
        """
        Performs curve fit to determine energy axis.
        
        Using the known ionization potential, the harmonics and sidebands are assigned.

        Parameters
        ----------
        max_pixel : int, optional
            Pixel up to which peaks can will be recognized as harmonics/sidebands. 
            The default is 550.
            
        peak_distance : int or float, optional
            Distance between peaks in harmonic orders.
            Change this if very strong sidebands get recognized by the peak finder.
            The default is 2. This corresponds to no sidebands.
        
        height : float, optional
            Minimum peak height for the the peak finder. The default is 0.1.
        
        prominence : float, optional
            Minimum peak prominence for the the peak finder. The default is 0.1.
            
        width : float, optional
            Minimum peak width for the the peak finder. The default is 5.

        Returns
        -------
        None.
        
        Notes
        -----
        * 'peak_distance'=2 will assume odd order harmonics
        * 'peak_distance'=1 will assume odd order harmonics with sidebands 
            between at the locations of even orders
        * 'peak_distance'=2/3 will assume odd order harmonics with 
            two equidistant sidebands between
        other values expressable as 2/n should work along similar lines 
        but are not explicitly supported

        """
 
        if self.speed_distributions is None:
            message = "Perform Abel inversion first to get speed distribution."
            raise AttributeError(message)
        
        def velocity(n, a, b):   # n, b in [harm. orders]; a in [samples^2/harm. order]
            return np.sqrt(a * (n+b))   # output in [samples]
        
        peaks, properties = scipy.signal.find_peaks(self.speed_distribution[0:max_pixel], 
                                                    height=height, prominence=prominence,
                                                    width=width)
        
        plt.figure(num='Speed distribution', clear=True)
        plt.plot(self.speed_distribution)
        plt.plot(peaks, self.speed_distribution[peaks], 'd')
        plt.xlabel('speed (samples)')
        plt.ylabel('intensity (normalized)')
        plt.show()
        
        nn = np.arange(len(peaks))*peak_distance
        popt, pcov = curve_fit(velocity, nn, peaks, p0=[1e4,1])
        plotrange = np.linspace(-popt[1]/peak_distance,len(peaks),10000)
        print(popt)
        
        plt.figure(num='Speed curve-fit', clear=True)
        plt.plot(peaks, 'x')
        plt.plot(plotrange, velocity(plotrange*peak_distance, *popt))
        plt.xlabel('harmonic peak number')
        plt.ylabel('speed (samples)')
        plt.show()
        
        self.speed_axis = np.arange(len(self.speed_distribution))   # [samples]
        self.energies = self.speed_axis**2 * E_IR / popt[0]         # [eV]
        self.velocity_axis = np.sqrt(2 * self.energies / m_e) / 1e3 # [km/s]
        
        # Multiplying by Jacobi determinant for plotting of PES
        self.speed_distribution_jacobi = self.speed_distribution / self.speed_axis
        self.speed_distributions_jacobi = self.speed_distributions / self.speed_axis
        
        # finding and assigning harmonics and sidebands
        expected_peak_orders = (peak_distance * np.arange(50)) + 1
        expected_peak_energies = expected_peak_orders * E_IR
        lowest_peak_energy = self.energies[peaks][0] + self.Ip
        lowest_peak_index = np.argmin(np.abs(lowest_peak_energy - expected_peak_energies))
        peak_orders = expected_peak_orders[lowest_peak_index:lowest_peak_index+len(peaks)]
        harmonics = []
        sidebands = []
        n_harmonics = []
        n_sidebands = []
        for i in range(len(peaks)):
            peak, n_peak = peaks[i], peak_orders[i]
            if n_peak%2 == 1: #odd orders
                harmonics.append(peak)
                n_harmonics.append(n_peak)
            else:
                sidebands.append(peak)
                n_sidebands.append(n_peak)
        self.harmonics = np.array(harmonics)
        self.n_harmonics = np.array(n_harmonics)
        self.sidebands = np.array(sidebands)
        self.n_sidebands = np.array(n_sidebands)
    
    
    def save_energy_scale(self):
        '''saves the energy scale and peak locations of a scan'''
        
        filetypes = [('HDF5 dataset','*.h5')]
            
        root = tk.Tk()
        root.withdraw()
        path = asksaveasfilename(title='Save as', defaultextension=".h5", 
                                 filetypes=filetypes)
        root.destroy()    
        print("Saving at: " + path)
        
        if path.split(".")[-1] == "h5": # Save as h5 dataset
            with h5py.File(path, "w") as f:
                f.create_dataset("speed_axis", data=self.speed_axis)
                f.create_dataset("energy_axis", data=self.energies)
                f.create_dataset("velocity_axis", data=self.velocity_axis)
                f.create_dataset("harmonic_locations", data=self.harmonics)
                f.create_dataset("harmonic_orders", data=self.n_harmonics)
                f.create_dataset("sideband_locations", data=self.sidebands)
                f.create_dataset("sideband_orders", data=self.n_sidebands)
    
    def read_energy_scale(self):
        '''Reads h5 files containing the energy calibration'''
            
        filetypes = [('HDF5 dataset','*.h5')]
            
        root = tk.Tk()
        root.withdraw()
        path = askopenfilename(title='Open file containing energy scale', 
                               defaultextension=".h5", filetypes=filetypes)
        root.destroy()    
        
        if path.split(".")[-1] == "h5": # Read from h5 dataset
            with h5py.File(path, "r") as f:
                self.speed_axis = np.array(f['speed_axis'])
                self.energies = np.array(f['energy_axis'])
                self.velocity_axis = np.array(f['velocity_axis'])
                self.harmonics = np.array(f['harmonic_locations'])
                self.n_harmonics = np.array(f['harmonic_orders'])
                self.sidebands = np.array(f['sideband_locations'])
                self.n_sidebands = np.array(f['sideband_orders'])
        
        # Multiplying by Jacobi determinant for plotting of PES
        self.speed_distribution_jacobi = self.speed_distribution / self.speed_axis
        self.speed_distributions_jacobi = self.speed_distributions / self.speed_axis
    
    
    
    def time_scale(self, step, step_unit='um'):
        """
        Define the time axis given the steps size for the piezo in microns or radians.
        Legacy alias - replaced by phase_scale.

        """
        
        return self.phase_scale(step, step_unit)


    def phase_scale(self, step, step_unit='um'):
        """
        Define the axis for the scan parameter of the delay stage or CEP wedge.

        Parameters
        ----------
        step : float
            Step size used by the piezo in µm,
            phase step by the stabilization system in mrad,
            or CEP wedge step distance in mm.
            Choose which to specify with step_unit.
        
        step_unit: 'um', 'mrad', 'mm'
            Choose which one to specify. Default is 'um'.
        
        Notes
        -----
        If 'um' or 'mrad' is specified, the class will assume a delay scan,
        if 'mm' is specified, the class will treat this scan as a CEP scan.
        

        Returns
        -------
        None.

        """
        
        if step_unit == 'um':
            self.scan_type = self.types.DELAY
            print('Delay scan with ' + str(self.nsteps) + ' delay steps')
            delta_x = step                       # step size in µm
            delta_t = step*1e-6 * 2 / c * 1e15   # step size in fs
            delta_phi = omega_IR * delta_t       # step size in rad
    
        elif step_unit == 'mrad':
            self.scan_type = self.types.DELAY
            print('Delay scan with ' + str(self.nsteps) + ' delay steps')
            delta_phi = step * 1e-3              # step size in rad
            delta_t = delta_phi / omega_IR       # step size in fs
            delta_x = delta_t/1e15 * c * 1e6/2   # step size in µm
        
        elif step_unit == 'mm':
            self.scan_type = self.types.CEP
            print('CEP scan with ' + str(self.nsteps) + ' CEP steps')
            delta_x = step                                           # step size in mm
            delta_phi = step*CEP_factor / (lambda_IR*1e3) * 2*np.pi  # step size in rad
            delta_t = delta_phi / omega_IR
            
        else: raise ValueError('Given step type not supported, try e.g. "um" or "mrad"')
        
        self.distances = np.linspace(0, self.nsteps*delta_x, self.nsteps)
        self.times = np.linspace(0, self.nsteps*delta_t, self.nsteps)
        self.angles = np.linspace(0, self.nsteps*delta_phi, self.nsteps)
    
    
    
    def plot_oscillation(self, oscillation, labels=None, popts=None, fig_number=None, 
                         delay_unit='fs', size_hor=10, size_ver=8, saving=False):
        '''plots multiple oscillations in seperate subplots with line coloring showing their energies,
            each having a seperate axis indicating their relative intensity
            if only one is given, the function also works'''
    
        x_axis, x_label, x_linarity = self._phase_axis(delay_unit)
        plt.figure(num=fig_number, clear=True, figsize=(size_hor, size_ver))
        scaling = 1e3  # to stretch all to make numbers with less decimals
        
        if popts is not None:
            fit_x_axis = np.linspace(x_axis[0], x_axis[-1], 1000)
            def cos(t, omega, phi, a, b): # fittable cosine with linear background
                return a * np.cos(omega*omega_IR * t - phi) + b * t
    
        # plot multiple oscillations in one figure
        if len(np.shape(oscillation)) == 2:
            n_subfigs = len(oscillation)
            gs = gridspec.GridSpec(n_subfigs, 1)
            colors = util.rainbow_colors(n_subfigs, 1.3) # spectral colormap from red to blue
            darker_colors = util.rainbow_colors(n_subfigs, 2) # spectral colormap from red to blue
            for i in range(n_subfigs):
                if i==0:
                    ax0 = plt.subplot(gs[i])
    
                    pl, = ax0.plot(x_axis, oscillation[n_subfigs-1-i]*scaling, 'x-', 
                                   lw=0.8, ms=6, color=colors[n_subfigs-1-i])
                    if popts is not None:
                        pl, = ax0.plot(fit_x_axis, cos(fit_x_axis, *popts[n_subfigs-1-i])*scaling,
                                        lw=0.6, color=darker_colors[n_subfigs-1-i])
                    axl=ax0
                else:
                    axi = plt.subplot(gs[i], sharex=axl)
    
                    pl, = axi.plot(x_axis, oscillation[n_subfigs-1-i]*scaling, 'x-', 
                                   lw=0.8, ms=6, color=colors[n_subfigs-1-i])
                    if popts is not None:
                        pl, = axi.plot(fit_x_axis, cos(fit_x_axis, *popts[n_subfigs-1-i])*scaling,
                                        lw=0.6, color=darker_colors[n_subfigs-1-i])
    
                    yticks = axi.yaxis.get_major_ticks()
                    yticks[-1].label1.set_visible(False)
                    if i != n_subfigs - 1:
                        axi.tick_params(axis='x', labelbottom=False)
                    axl=axi
    
                if i==int(len(oscillation-1)/2): # put ylabel only on the middle plot
                    plt.ylabel('count difference (a.u.) \n')
    
                plt.grid(axis='both')
                if labels is not None:
                    plt.legend([labels[n_subfigs-1-i]], loc='upper left',
                               bbox_to_anchor=(0.07-0.01*n_subfigs,1.03)) # change label position here !!
    
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

        x_axis, x_label, _ = self._phase_axis(delay_unit)
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



    def _calculate_asymmetry_parameter(self, origin=default_origin):
        """
        Calculates signal difference between top and bottom half of the image.
        TODO: this method is still in development

        Parameters
        ----------
        origin : 2-tuple of int, optional
            Image center in pixels. The default can be set globally.

        Returns
        -------
        None.

        """
        
        for i, inverted_image in tqdm(enumerate(self.inverted_scan), total=self.nsteps):
            top_half = vmi_radial_intensity('int3D', inverted_image, origin=origin,
                                            theta_low=0, theta_high=np.pi)[1]
            low_half = vmi_radial_intensity('int3D', inverted_image, origin=origin,
                                            theta_low=-np.pi, theta_high=0)[1]
            parameter = (top_half - low_half)
            self.speed_distributions[i] = parameter[:600]
            
            self.speed_distribution_jacobi = self.speed_distribution / self.speed_axis
            self.speed_distributions_jacobi = self.speed_distributions / self.speed_axis
            
    
    
    def prepare_analysis(self, integral_width=2, smoothE=None, smoothT=None):
        '''
        Normalizes data in a way that is useful for the RABBITT-analysis
        and extracts the integrals of sidband and harmonic signal.

        Parameters
        ----------
        integral_width : int, optional
            Specifies how many bins either side of the sideband/harmonic maximum
            are taken into account for the integral. The default is 2.
        
        smoothE : int, optional
            Can be specified to smooth the data along the energy axis.
            The integer will specify the size of the smoothing kernel.
            The default is None, which deactivates smoothing completely.
        
        smoothT : int, optional
            If smoothing is done along the energy axis,
            this can be specified to smooth the data along the time axis.
            The integer will specify the size of the smoothing kernel.
            The default is None, which deactivates smoothing along this axis.

        Returns
        -------
        None.

        '''
        
        self.speed_distribution_norm = normalized(self.speed_distribution_jacobi, 'sum')
        
        # Normalize signal for each delay step
        self.data_norm = (self.speed_distributions_jacobi.T / np.nansum(self.speed_distributions_jacobi, axis=1)).T
        
        # Smooth data if specified
        if (smoothE is None) or (smoothE == 0):
            self.data_smooth = self.data_norm
        elif (smoothT is None) or (smoothT == 0):
            self.data_smooth = util.smooth_1D(self.data_norm.T, smoothE, 'hanning').T
        else:
            self.data_smooth = util.smooth_2D(self.data_norm, smoothT, smoothE)
        
        # Calculate changes from average signal
        self.data_diff = self.data_smooth - normalized(np.nansum(self.data_smooth, axis=0), 'sum')
        
        if self.left is None:  # don't overwrite if ranges have been determined already
            self.left  = self.sidebands - integral_width
            self.right = self.sidebands + integral_width+1
        
        self.HH_oscillation = np.sum(np.array(np.split(self.data_diff, np.sort((self.harmonics-integral_width,self.harmonics+integral_width+1), 
                                                                               axis=None), axis=1)[1::2]), axis=2)
        cutted = np.split(self.data_diff, np.sort((self.left,self.right), axis=None), axis=1)[1::2]
        self.SB_oscillation = np.array([np.sum(cutted[i], axis=1) for i in range(len(cutted))])
        
    
    
    def plot_phase_diagram(self, indicator='points', show_amplitude=False, 
                           left=None, right=None, show_errors=False, saving=False,
                           show_external=False):
        """
        Plots the phase by energy.

        Parameters
        ----------
        indicator : {'none', 'points', 'range'}, optional
            Specifies which indication for sideband positions should be shown. 
            The default is 'points'.
        show_amplitude : bool, optional
            If true an additional axis is added to the same plot to show the modulation depth by energy. 
            The default is False.
        left : arr of int, optional
            Only read when indicator is 'range'.
            Left bounds of ranges in pixels. The default is None,
            which defaults to the left bounds saved in local variable self.left.
        right : arr of int, optional
            Only read when indicator is 'range'.
            Left bounds of ranges in pixels. The default is None,
            which defaults to the left bounds saved in local variable self.right.
        show_errors : bool, optional
            If true shows shaded regions depicting the range of error. 
            The default is False.
        saving : bool, optional
            If true saves the plot as pdf-format. The default is False.
        show_external : bool, optional
            Do not show the figure inside this function (can be showed externally).
            The default is False.

        Returns
        -------
        matplitlib.Figure
            The figure object of the plot.
        2-tuple of matplotlib.axes.Axes
            The axis objects of the plot

        """
        if len(self.phase_by_energy) == 0:   # "self.phase_by_energy" was never defined
            self.do_fourier_transform(plotting=False)

        fig, ax1 = plt.subplots(num=(self._prefix() + 'Phases'), clear=True)
        ax1.plot(self.energies, self.phase_by_energy, 'x-', color='k')
        ax1.tick_params(axis='both', which='major')
        ax1.set_xlabel('photoelectron energy [eV]')
        ax1.set_ylabel('phase [rad]')

        if show_amplitude:
            ax2 = ax1.twinx()
            ax2.plot(self.energies, self.depth_by_energy, 'x-', color='grey')
            ax2.tick_params(axis='both', which='major')
            ax2.set_ylabel('modul. amp. (a.u.)', color='grey')
            highest = np.nanmax(self.depth_by_energy)
            ax1.set_ylim([-1.8*np.pi, 1.05*np.pi])  # avoid overlap
            ax2.set_ylim([0, 2.5*highest])  # avoid overlap
            
        if show_errors:
            upper_bound = self.phase_by_energy + self.phase_by_energy_error
            lower_bound = self.phase_by_energy - self.phase_by_energy_error
            ax1.fill_between(self.energies, upper_bound, lower_bound, color='gray')

        if indicator == 'points':   # draw points for the harmonic and sideband locations
            ax1.plot(self.energies[self.harmonics], self.phase_by_energy[self.harmonics], 'o', color='orange', label='HH')
            ax1.plot(self.energies[self.sidebands], self.phase_by_energy[self.sidebands], 'o', color='green', label='SB')
        
        if indicator == 'range':   # color points ascribed to each sideband in different colors
            if left is None: left = self.left
            if right is None: right = self.right
            colors = util.rainbow_colors(len(left), 1.0)   # spectral colormap from red to blue
            colors_sat = util.rainbow_colors(len(left), 1.3)   # spectral colormap from red to blue
            for i in range(len(left)):
                ax1.plot(self.energies[left[i]:right[i]], self.phase_by_energy[left[i]:right[i]],
                         'x-', label=self._legend_name(self.n_sidebands[i]), color=colors_sat[i])
                if show_amplitude:
                    ax2.plot(self.energies[left[i]:right[i]], self.depth_by_energy[left[i]:right[i]],
                             'x-', color=colors[i], alpha=0.5)
                if show_errors:
                    ax1.fill_between(self.energies[left[i]:right[i]], upper_bound[left[i]:right[i]], 
                                     lower_bound[left[i]:right[i]], color=colors[i])

        plt.xlim([self.min_energy, self.max_energy])
        ax1.legend(loc='upper right')
        fig.tight_layout()
        if saving: plt.savefig('favorite_plot.png', dpi=400)
        if not show_external: plt.show()
        
        return fig, (ax1, ax2) 
        


    def do_fourier_transform(self, plotting=True):
        """
        Does a fourier transform for each energy bin 
        and extracts the phase of the oscillating component.

        Parameters
        ----------
        plotting : bool, optional
            Whether to directly plot the result. The default is True.

        Returns
        -------
        np.array
            Array containing the oscillation phases at each energy.

        """
        
        if self.data_diff is None:
            self.prepare_analysis()  # Calculate difference dataset expected by curve fit

        self.phase_by_energy = np.array([])  # oscillation phase
        self.depth_by_energy = np.array([])  # oscillation amplitude
        self.phase_by_energy_error = np.array([])  # oscillation phase
        self.depth_by_energy_error = np.array([])  # oscillation amplitude
        
        # Perform all the Fourier transforms
        fouriers = [np.fft.fft(single_line) for single_line in self.data_diff.T]
        fourier_map = np.abs(fouriers)
        fourier_phases = np.angle(fouriers)
        fourier_spectrum = np.nansum(fourier_map, axis=0)
        
        # Find oscillation frequency and extract phase there
        peak = np.argmax(fourier_spectrum[3:]) + 3
        print('Used fourier bin ' + str(peak))
        self.phase_by_energy = -fourier_phases.T[peak]
        self.depth_by_energy = fourier_map.T[peak]

        # show corresponding plot
        if plotting == True:
            self.plot_phase_diagram(indicator='points', show_amplitude=True)

        return self.phase_by_energy


    def do_cosine_fit(self, plotting=True, omega=2, average=0):
        """
        Does a cosine fit for each energy bin 
        and extracts the phase of the oscillating component

        Parameters
        ----------
        plotting : bool, optional
            Whether to directly plot the result. The default is True.
        omega : int or float, optional
            The frequency of the angular component to be fitted in units of omega_IR.
            The default is 2, which captures RABBITT with harmonics spaced 2*E_IR.
            Use 1 for harmonics spaced 1*E_IR or the Ti:Sa CEP scan.
        average : int, optional
            Specify >0 to average neighboring pixels when fitting for less noisy fits.
            The default is 0, meaning no averaging.

        Returns
        -------
        np.array
            Array containing the oscillation phases at each energy.

        """
        
        if self.data_diff is None:
            self.prepare_analysis()  # Calculate difference dataset expected by curve fit

        self.phase_by_energy = np.array([])  # oscillation phase
        self.depth_by_energy = np.array([])  # oscillation amplitude
        self.slope_by_energy = np.array([])  # slope of the background
        self.phase_by_energy_error = np.array([])
        self.depth_by_energy_error = np.array([])
        self.slope_by_energy_error = np.array([])
        
        def cos(t, phi, a, b): # fittable cosine with linear background
            return a * np.cos(omega*omega_IR * t - phi) + b * t

        for i in range(len(self.data_diff.T)):
            if average == 0:
                single_line = self.data_diff.T[i]
            if average > 0:
               single_line = (self.data_diff.T[i-average:i+average+1]).sum(axis=0)
            
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

        
        
    def select_sideband_ranges(self, manual_selection=False, 
                               integral_width=None, dist=None):
        """
        Calculates the positions and ranges of the sideband oscillations in terms of their modulation amplitude.
    
        Parameters
        ----------
        manual_selection: bool, optional
            Choose whether to open an interactive plot to manually select. 
            The default is False.
        integral width : int or None, optional
            Specify to just use an integer number of bins either side of the
            sideband intensity maximum.
            The default is None which does look at the modulation amplitude.
            Only read if  manual_election is False.
        dist : int or None, optional
            How many pixels off the pre-given walues to look for max.
            Default is None, which looks inside the previous integration ranges.
    
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
    
        if len(self.phase_by_energy) == 0:   # "self.phase_by_energy" was never defined
            message = "Perform cosine fit or fourier transform to obtain the oscillation amplitude."
            raise AttributeError(message)
        
        if manual_selection:
            self.left, self.right = util.select_ranges(self.plot_phase_diagram, self.energies,
                                                       show_amplitude=True, indicator='range')
        elif integral_width is not None:
            self.left  = self.sidebands - integral_width
            self.right = self.sidebands + integral_width+1            
        else:    
            if dist is None:   # look for maximum inside the pre-made selection
                left = self.left
                right = self.right
            else:   # look for maximum extent bins to the left and right
                left = self.sidebands - dist
                right = self.sidebands + dist
        
            for i, sb in enumerate(self.sidebands):
                fwhm_data = util.find_FWHM(self.depth_by_energy[left[i]:right[i]])
                self.left[i] = fwhm_data[1][0] + left[i]
                self.right[i] = fwhm_data[1][2] + left[i]
            
    
        cutted = np.split(self.data_diff, np.sort((self.left,self.right), axis=None), axis=1)[1::2]
        self.SB_oscillation = np.array([np.sum(cutted[i], axis=1) for i in range(len(cutted))])
        
        self.plot_phase_diagram('range', True, self.left, self.right)
    
        return self.left, self.right



    def phases_cosine(self, oscillation=None, labels=None, omega=2):
        """
        Fit the phases of sidebands (or harmonics) alredy integrated over a region.
    
        Parameters
        ----------
        oscillation : list of np.arrays, optional
            List of arrays each containing data values along the common time axis
            representing an oscillation to fit. Typical choices are the variables
            self.SB_oscillation and self.HH_oscillation from within this library,
            the former is selected by default / if None is passed.
        labels : list of str, optional
            Labels for the plots representing the fits. 
            Per default or for None, sideband labels are generated and used.
        omega : int or float, optional
            The frequency of the angular component to be fitted in units of omega_IR.
            The default is 2, which captures RABBITT with harmonics spaced 2*E_IR.
            Use 1 for harmonics spaced 1*E_IR or the Ti:Sa CEP scan.
    
        Returns
        -------
        self.phases : np.array
            Array of the phases determined by each fit.
        self.phase_errors : np.array
            Array of the phases uncertainties determined by each fit.
    
        """
            
        if oscillation is None:
            oscillation = self.SB_oscillation
        if labels is None:
            labels = self._legend_name(self.n_sidebands)
    
        def cos(t, phi, a, b): # fittable cosine with linear background
            return a * np.cos(omega*omega_IR * t - phi) + b * t
        
        self.phases = np.array([])
        self.phase_errors = np.array([])
        cos_fit_popts = []
    
        try: tt = np.arange(0, self.times[-1], 0.0001) # finer time array for plotting
        except AttributeError: # time scale has not jet been calculated
            self.time_steps() # calculate the time scale
            tt = np.arange(0, self.times[-1], 0.0001) # finer time array for plotting
    
    
        for i in range(len(oscillation)):
    
            # perform cosine fit
            normalization = np.max(np.abs(oscillation[i])) # to make initial guess closer
            popt, pcov = scipy.optimize.curve_fit(cos, self.times, oscillation[i]/normalization)
            popt[1:] *= normalization
            # write down phase parameters
            cos_fit_popts.append(np.append(omega, popt))
            if popt[1] > 0:
                self.phases = np.append(self.phases, (popt[0])%(2*np.pi))
            else:
                self.phases = np.append(self.phases, (popt[0]+np.pi)%(2*np.pi))
            print(self._prefix() + labels[i] + ': ' + str(self.phases[-1]*180/np.pi) + ' +- ' + str(pcov[0][0]*180/np.pi))
            self.phase_errors = np.append(self.phase_errors, pcov[0][0])
    
            # show corresponding plot
            plt.figure((self._prefix() + labels[i]), clear=True)
            plt.plot(self.times, oscillation[i], 'x-', color='r', label='measurement')
            plt.plot(tt, cos(tt, *popt), color='b', label='fit')
            plt.xlabel('assumed time [fs]')
            plt.ylabel('normalized count difference')
            plt.legend()
            plt.show()
    
        self.cos_fit_popts = np.array(cos_fit_popts)
        return self.phases, self.phase_errors



        
#%% Example usage

if __name__ == "__main__":

    hasi = RABBITT_scan('Ar')                      # initialize scan object
    hasi.read_scan_files()                         # read .h5 measurement data 
    hasi.perform_abel_inversion(theta=(0, np.pi))  # abel invert, integrate upper image half
    
#%%%
    hasi.energy_scale()                            # calibrate energy axis 
    hasi.phase_scale(100, 'mrad')                  # apply phase axis
    
    hasi.plot_RABBITT_trace(hasi.speed_distributions, delay_unit='fs', energy_unit='v')
    hasi.plot_RABBITT_trace(hasi.speed_distributions_jacobi, delay_unit='fs', energy_unit='eV')
    
    hasi.do_cosine_fit(plotting=False)             # cosine-fit for every energy
    hasi.select_sideband_ranges(dist=10)           # select sb integration ranges
    
    legend_names = [hasi._legend_name(n_SB) for n_SB in hasi.n_sidebands]
    hasi.plot_oscillation(hasi.SB_oscillation, legend_names, hasi._prefix() + 'Sideband Oscillation')    
        

    
        
        

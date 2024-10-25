"""
==================================================

This script takes processed Raman spectra and fits the organic D&G band region for every point-spectrum using a standardised fit regime optimised for DUV Raman spectra. The fit uses pseudo-voigt functions and is based on the work of Kouketsu et al (2014), and includes the following peaks:
    D1 - major organic defect mode at 1375 cm-1
    D2 - minor organic defect mode at 1660 cm-1
    D3 - minor organic defect mode at 1470 cm-1
    D4 - minor organic defect mode at 1275 cm-1
    G - major organic graphitic mode at 1605 cm-1
    O2 - atmospheric O2 at 1556 cm-1, uses fermi-dirac function with predetermined width, position
    silicate - Si-O stretching mode of plagioclase/pyroxene/maskelynite at ~1000 cm-1
    carbonate - CO3 v1 mode at 1080 cm-1
    unknown - unassigned peak at 1175 cm-1
    
Fit also calculates the following intensity ratio parameters defined by Beyssac (2004), Rahl (2005), Lahfid (2010):
    R1: amplitude ratio D1/G
    R2: integrated intensity ratio D1/(G+D1+D2)
    RA1: integrated intensity ratio (D1+D4)/(G+D1+D2+D3+D4)
    RA2: integrated intensity ratio (D1+D4)/(G+D2+D3)

This script is designed to accept files outputted by the MOB_R-mapping script.

References:
- Beyssac, O., Bollinger, L., Avouac, J. P., and Goffe, B. (2004). Earth and Planetary Science Letters, 225(1-2), 233-241. https://doi.org/10.1016/j.epsl.2004.05.023
- Rahl, J. M., Anderson, K. M., Brandon, M. T., and Fassoulas, C. (2005). Earth and Planetary Science Letters, 240(2), 339-354. https://doi.org/10.1016/j.epsl.2005.09.055
- Lahfid, A., Beyssac, O., Deville, E., Negro, F., Chopin, C., and Goffe, B. (2010). Terra Nova, 22(5), 354-360. https://doi.org/10.1111/j.1365-3121.2010.00956.x
- Kouketsu, Y., Mizukami, T., Mori, H., Endo, S., Aoya, M., Hara, H., et al. (2014). Island Arc, 23(1), 33-50. https://doi.org/10.1111/iar.12057

==================================================
"""

import os
import glob
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import lmfit as lmfit

from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy.special import expit
from scipy.integrate import trapz
from itertools import combinations
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==================================================
# initial variables

# specify which scan to import
Scan = '*'

# define the directories for input/output:
Data_dir = './data/MOBIUS/Raman/'
Output_dir = './output/MOBIUS/'
Figure_dir = './figures/MOBIUS/'

# specify data pre-processing
Smooth = True           # use smoothed data?
Normalise = True        # use normalised data?

# specify fitting regime parameters
Fit_DG = True           # run the fit?
DG_fit_function = 'Pseudo-Voigt'    # function to use for fitting, choose from 'pseudo-voigt', 'gaussian', 'lorentzian', or 'fermi-dirac'
DG_peaks = {'peak': ['silicate', 'carbonate', 'mystery', 'D4', 'D1', 'D3', 'O2', 'G', 'D2'], 'center': [1000, 1080, 1175, 1275, 1376, 1470, 1556, 1605, 1660]}      # specify peak names and positions
DG_window = 30          # acceptable tolerance for peak position changes

# specify additional data to plot
plot_scan_avs = True    # plot average spectra for each scan?
plot_std = False        # plot +/-1 standard deviation?

# color parameters for plotting
Cmap = 'viridis'        # default matplotlib colormap to use for false color maps
Color_list =  ['orange', 'limegreen', 'deeppink', 'darkgoldenrod', 'cornflowerblue', 'red', 'green', 'tab:purple', 'slategray', 'm', 'tab:brown', 'c', 'b', 'tab:orange']            # matplotlib single colors for line and scatter plots
Marker_list = ['o', 's', 'v', '^', 'D', '*']    # matplotlib marker styles for scatter plots

# ==================================================
# define functions for finding minima/maxima

def find_max(x, y, x_start, x_end):
    # function for finding the maximum from a slice of input data
    x_slice = x[np.where((x_start <= x) & (x <= x_end))]       # create slice
    y_slice = y[np.where((x_start <= x) & (x <= x_end))]
    i = np.argmax(y_slice)
    return np.asarray([x_slice[i], y_slice[i]]) # return x,y position of the maximum

def find_min(x, y, x_start, x_end):
    # function for finding the maximum from a slice of input data
    x_slice = x[np.where((x_start <= x) & (x <= x_end))]       # create slice
    y_slice = y[np.where((x_start <= x) & (x <= x_end))]
    i = np.argmin(y_slice)
    return np.asarray([x_slice[i], y_slice[i]]) # return x,y position of the minimum

# ==================================================
# functions for handling shift/wavelength conversion in figure axes

def wavelength2shift(wavelength, excitation=248.5794):
    # convert wavelength (nm) to Raman shift (cm-1)
    shift = ((1./excitation) - (1./wavelength)) * (10**7)
    return shift

def shift2wavelength(shift, excitation=248.5794):
    # convert Raman shift (cm-1) to wavelength (nm)
    wavelength = 1./((1./excitation) - shift/(10**7))
    return wavelength

def get_shift_ticks(x_range, major_spacing=None, major_ticks=6, minor_ticks=4, debug=False):
    # this function generates major and minor ticks for a secondary x axis expressed in Raman shift
    
    # convert x_range limits
    shift_min = wavelength2shift(x_range[0])
    shift_max = wavelength2shift(x_range[1])
    
    # determine major spacing (if not specified)
    if major_spacing == None:
        spacing_temp = (shift_max-shift_min)/(major_ticks-1)
        print spacing_temp
        if spacing_temp <= 1.:
            major_spacing = 1.
        else:
            scales = np.array([1000., 2000., 5000.])
            for i in [0, 1, 2]:
                check = spacing_temp > scales
                print scales, check
                if np.any(check) == True:
                    major_spacing = scales[check][-1]
                    print "major tick spacing set to %0.f" % major_spacing
                    break
                else:
                    scales /= 10.
    
    # determine first and last major tick
    tick_0 = np.floor(shift_min // major_spacing)
    tick_n = np.ceil(shift_max // major_spacing)
    if debug == True:
        print "plot min: %0.f cm / %0.2f nm" % (shift_min, x_range[0])
        print "plot max: %0.f cm / %0.2f nm" % (shift_max, x_range[1])
        print "%s major ticks needed from %0.f to %0.f" % (tick_n-(tick_0+1), tick_0+1, tick_n)
        
    # generate arrays for major ticks (including 0th tick, beyond left edge of plot)
    major_shifts = major_spacing * np.arange(tick_0, tick_n+1)
    major_wavelengths = shift2wavelength(major_shifts)
    major_locations = (major_wavelengths - x_range[0])/(x_range[1] - x_range[0])
    
    # generate array for minor ticks
    minor_spacing = major_spacing/(minor_ticks+1)
    minor_shifts = minor_spacing * np.arange((minor_ticks+1)*tick_0, (minor_ticks+1)*(tick_n+1))
    minor_shifts = np.delete(minor_shifts, slice(0, None, minor_ticks+1))
    minor_wavelengths = shift2wavelength(minor_shifts)
    minor_locations = (minor_wavelengths - x_range[0])/(x_range[1] - x_range[0])
        
    # create boolean array based on which ticks are located between 0 and 1
    majors_plot = np.logical_and(major_locations >= 0., major_locations <= 1.)
    minors_plot = np.logical_and(minor_locations >= 0., minor_locations <= 1.)
    if debug == True:
        print major_shifts[majors_plot]
        print minor_shifts[minors_plot]
    
    # only return ticks that are plottable, making sure that shift labels are strings with 0 decimal places
    return ["%.0f" % x for x in major_shifts[majors_plot]], major_locations[majors_plot], minor_locations[minors_plot]


def get_wavelength_ticks(x_range, laser=248.5794, major_spacing=5., minor_ticks=4, debug=False):
    # this function generates major and minor ticks for a secondary x axis expressed in wavelength
    
    # convert x_range limits and determine bounding major ticks
    wavelength_min = shift2wavelength(x_range[0])
    wavelength_max = shift2wavelength(x_range[1])
    tick_0 = np.floor(wavelength_min // major_spacing)
    tick_n = np.ceil(wavelength_max // major_spacing)
    if debug == True:
        print "plot min: %0.f nm / %0.1f cm" % (wavelength_min, x_range[0])
        print "plot max: %0.f nm / %0.1f cm" % (wavelength_max, x_range[1])
        print "%s major ticks needed from %0.f to %0.f" % (tick_n-(tick_0+1), tick_0+1, tick_n)
        
    # generate arrays for major ticks (including 0th tick, beyond left edge of plot)
    major_wavelengths = major_spacing * np.arange(tick_0, tick_n+1)
    major_shifts = wavelength2shift(major_wavelengths)
    major_locations = (major_shifts - x_range[0])/(x_range[1] - x_range[0])
    if debug == True:
        print major_wavelengths
        print major_shifts
        
    # generate array for minor ticks
    minor_spacing = major_spacing/(minor_ticks+1)
    minor_wavelengths = minor_spacing * np.arange((minor_ticks+1)*tick_0, (minor_ticks+1)*(tick_n+1))
    minor_wavelengths = np.delete(minor_wavelengths, slice(0, None, minor_ticks+1))
    minor_shifts = wavelength2shift(minor_wavelengths)
    minor_locations = (minor_shifts - x_range[0])/(x_range[1] - x_range[0])
        
    # create boolean array based on which ticks are located between 0 and 1
    majors_plot = np.logical_and(major_locations >= 0., major_locations <= 1.)
    minors_plot = np.logical_and(minor_locations >= 0., minor_locations <= 1.)
    if debug == True:
        print major_wavelengths[majors_plot]
        print minor_wavelengths[minors_plot]
    
    # only return ticks that are plottable, making sure that shift labels are strings with 0 decimal places
    if major_spacing >= 1:
        return ["%.0f" % x for x in major_wavelengths[majors_plot]], major_locations[majors_plot], minor_locations[minors_plot]
    elif major_spacing >= 0.1:
        return ["%.1f" % x for x in major_wavelengths[majors_plot]], major_locations[majors_plot], minor_locations[minors_plot]
    else:
        return ["%.2f" % x for x in major_wavelengths[majors_plot]], major_locations[majors_plot], minor_locations[minors_plot]

# ==================================================
# functions for fitting the D&G peak region

def DG_curve(x, params, maxima={'peak': [], 'center': []}, function='g', peak=None):
    # function for generating D&G curve using stated parameters
    global DG_peaks
    if len(maxima['peak']) == 0:
        maxima = DG_peaks
    gradient = params['gradient']
    intercept = params['intercept']
    model = gradient * x + intercept
    indices = range(0, len(maxima['center']))
    # check if outputting single peak or not
    if peak != None:
        if peak in maxima['peak']:
            indices = [maxima['peak'].index(peak)]
    for i in indices:
        peak = maxima['peak'][i]
        A = params['%s_amplitude' % peak]
        mu = params['%s_center' % peak]
        if peak == 'O2':
            width = params['%s_fwhm' % peak] / 2.
            rounding = params['%s_round' % peak]
            model += A * expit(-((np.absolute(x - mu) - width) / rounding))
        else:
            if function.lower() in ['fd', 'fermidirac', 'fermi-dirac']:
                width = params['%s_fwhm' % peak] / 2.
                rounding = params['%s_round' % peak]
                model += A * expit(-((np.absolute(x - mu) - width) / rounding))
            elif function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt']:
                sigma = params['%s_fwhm' % peak] / 2.355
                eta = params['%s_eta' % peak]
                model += A * (eta * (sigma**2)/((x - mu)**2 + sigma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
            elif function.lower() in ['l', 'lorentz', 'lorentzian']:
                gamma = params['%s_fwhm' % peak] / 2.
                model += A * (gamma**2)/((x - mu)**2 + gamma**2)
            else:
                sigma = params['%s_fwhm' % peak] / 2.355
                model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    return model

def DG_fit(params, x, y, maxima={'peak': [], 'center': []}, function='g'):
    # function for comparing fitted D&G peaks to actual spectrum, called by lmfit.minimize
    global DG_peaks
    if len(maxima['peak']) == 0:
        maxima = DG_peaks
    gradient = params['gradient']
    intercept = params['intercept']
    model = gradient * x + intercept
    for peak, position in zip(maxima['peak'], maxima['center']):
        A = params['%s_amplitude' % peak]
        mu = params['%s_center' % peak]
        if peak == 'O2':
            width = params['%s_fwhm' % peak] / 2.
            rounding = params['%s_round' % peak]
            model += A * expit(-((np.absolute(x - mu) - width) / rounding))
        else:
            if function.lower() in ['fd', 'fermidirac', 'fermi-dirac']:
                width = params['%s_fwhm' % peak] / 2.
                rounding = params['%s_round' % peak]
                model += A * expit(-((np.absolute(x - mu) - width) / rounding))
            elif function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt']:
                sigma = params['%s_fwhm' % peak] / 2.355
                eta = params['%s_eta' % peak]
                model += A * (eta * (sigma**2)/((x - mu)**2 + sigma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
            elif function.lower() in ['l', 'lorentz', 'lorentzian']:
                gamma = params['%s_fwhm' % peak] / 2.
                model += A * (gamma**2)/((x - mu)**2 + gamma**2)
            else:
                sigma = params['%s_fwhm' % peak] / 2.355
                model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    return y - model

def DG_fit_script(x, y, maxima={'peak': [], 'center': []}, function='g', free_baseline=True, window=30, max_fwhm=100., debug=False):
    # script for preparing data for D&G peak fitting and collating results
    global DG_peaks
    if len(maxima['peak']) == 0:
        maxima = DG_peaks
    if debug == True:
        print
        print "    input x:", np.shape(x)
        print "    input y:", np.shape(y)
        print "    input peaks:", maxima['peak']
        print "    input peak positions:", maxima['center']
    params = lmfit.Parameters()
    params.add('gradient', value=0., vary=free_baseline)
    params.add('intercept', value=np.amin(y), vary=free_baseline)
    y_max = np.amax(y)
    for peak, position in zip(maxima['peak'], maxima['center']):
        if peak == 'O2':
            params.add('%s_center' % peak, value=1556, min=1551, max=1561)
            params.add('%s_amplitude' % peak, value=y_max - np.amin(y), min=0.)
            params.add("%s_fwhm" % peak, value=32., vary=False)
            params.add("%s_round" % peak, value=3.5, vary=False)
        else:
            params.add('%s_center' % peak, value=position, min=position-window, max=position+window)
            params.add('%s_amplitude' % peak, value=y_max - np.amin(y), min=0.)
            params.add('%s_fwhm' % peak, value=40., min=20., max=max_fwhm)
            if function.lower() in ['fd', 'fermidirac', 'fermi-dirac']:
                params.add('%s_round' % peak, value=2., min=0.)
            elif function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt']:
                params.add('%s_eta' % peak, value=0.5, min=0., max=1.) 
    if debug == True:
        print "        initial parameters:"
        print params.pretty_print()
    fit_output = lmfit.minimize(DG_fit, params, args=(x, y, maxima, function))
    fit_curve = DG_curve(x, fit_output.params, maxima, function=function)
    print "        fit status: ", fit_output.message
    if debug == True:
        print "        fitted parameters:"
        print fit_output.params.pretty_print()
    # create summary of all relevant D,G output properties
    x_temp = np.linspace(np.amin(x)-100, np.amax(x)+100)
    base_integ = trapz(DG_curve(x_temp, fit_output.params, maxima=maxima, peak='baseline', function=function), x_temp)
    properties = {}
    for peak in maxima['peak']:
        properties['%s_center' % peak] = fit_output.params['%s_center' % peak].value
        properties['%s_amplitude' % peak] = fit_output.params['%s_amplitude' % peak].value
        properties['%s_fwhm' % peak] = fit_output.params['%s_fwhm' % peak].value
        if peak == 'O2':
            curve_integ = trapz(DG_curve(x_temp, fit_output.params, maxima=maxima, peak=peak, function='fd'), x_temp)
        else:
            if function.lower() in ['fd', 'fermidirac', 'fermi-dirac']:
                properties['%s_rounding' % peak] = fit_output.params['%s_round' % peak].value
            elif function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt']:
                properties['%s_eta' % peak] = fit_output.params['%s_eta' % peak].value
            curve_integ = trapz(DG_curve(x_temp, fit_output.params, maxima=maxima, peak=peak, function=function), x_temp)
        if debug == True:
            print peak, base_integ, curve_integ, curve_integ-base_integ
        if None not in [fit_output.params['%s_center' % peak].stderr, fit_output.params['%s_amplitude' % peak].stderr, fit_output.params['%s_fwhm' % peak].stderr]:
            properties['%s_center_err' % peak] = fit_output.params['%s_center' % peak].stderr
            properties['%s_amplitude_err' % peak] = fit_output.params['%s_amplitude' % peak].stderr
            properties['%s_fwhm_err' % peak] = fit_output.params['%s_fwhm' % peak].stderr
            if function.lower() in ['fd', 'fermidirac', 'fermi-dirac'] and peak != 'O2':
                properties['%s_rounding_err' % peak] = fit_output.params['%s_round' % peak].stderr
            elif function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt'] and peak != 'O2':
                properties['%s_eta_err' % peak] = fit_output.params['%s_eta' % peak].stderr
        else:
            properties['%s_center_err' % peak] = 0.
            properties['%s_amplitude_err' % peak] = 0.
            properties['%s_fwhm_err' % peak] = 0.
            if function.lower() in ['fd', 'fermidirac', 'fermi-dirac'] and peak != 'O2':
                properties['%s_rounding' % peak] = 0.
            elif function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt'] and peak != 'O2':
                properties['%s_eta' % peak] = 0.
        properties['%s_integrint' % peak] = curve_integ - base_integ
        
    # report results
    if debug == True:
        for key in properties.keys():
            print "    %s: %0.2f" % (key, properties[key])
        
    # calculate additional properties (R1: Rahl 2005, R2: Beyssac 2004, RA1&2: Lahfid 2010)
    if 'D1' in maxima['peak'] and 'G' in maxima['peak']:
        properties['R1_parameter'] = properties['D1_amplitude'] / properties['G_amplitude']
        if debug == True:
            print "R1: %0.2f" % properties['R1_parameter']
        if 'D2' in maxima['peak']:
            properties['R2_parameter'] = properties['D1_integrint'] / (properties['G_integrint'] + properties['D1_integrint'] + properties['D2_integrint'])
            if debug == True:
                print "R2: %0.2f" % properties['R2_parameter']
            if 'D3' in maxima['peak'] and 'D4' in maxima['peak']:
                properties['RA1_parameter'] = (properties['D1_integrint'] + properties['D4_integrint']) / (properties['G_integrint'] + properties['D1_integrint'] + properties['D2_integrint'] + properties['D3_integrint'] + properties['D4_integrint'])
                properties['RA2_parameter'] = (properties['D1_integrint'] + properties['D4_integrint']) / (properties['G_integrint'] + properties['D2_integrint'] + properties['D3_integrint'])
                if debug == True:
                    print "RA1: %0.2f" % properties['RA1_parameter']
                    print "RA2: %0.2f" % properties['RA2_parameter']
    return fit_output, fit_curve, properties

"""
# ==================================================
# import processed Raman spectra
# ==================================================
"""

# search for processed (output) files that match target scan
spec_dirs = sorted(glob.glob("%sSAU-008/SAU-008/*_%s_*/*_Rmap.txt" % (Output_dir, Scan)))

print
print "processed spectra files found:", len(spec_dirs)

# prepare arrays for storing data
samples = []
scans = []
raman_shift = []
map_positions = []
xy_coords = []
y_sub = []
y_sub_av = []
y_sub_std = []

for spec_dir in spec_dirs:
    while True:
        try:
            file_name = spec_dir.split("/")[-1]
            print
            print "attempting to import", file_name
            # import spec array
            spec = np.genfromtxt(spec_dir, delimiter="\t")
            if np.size(spec, axis=0) in [1024,1025,2048,2049]:
                spec = np.transpose(spec)
            shift_temp = spec[1]
            y_sub_temp = spec[2:]
            print "    spectra in map:", np.shape(y_sub_temp)
            # look for matching spatial position array in original data folder
            orig_folder = spec_dir.split("/")[-2]
            print "        importing point position array..."
            positions_dir = glob.glob("%sSAU-008/%s/working_files/points_array.csv" % (Data_dir, orig_folder))
            # import position array
            positions_temp = np.genfromtxt(positions_dir[0], delimiter=',', dtype='int')[1:,1:]
            print "            position array", np.shape(positions_temp)
            # convert to grid
            grid = np.mgrid[0:np.size(positions_temp, axis=0), 0:np.size(positions_temp, axis=1)]
            print "            grid:", np.shape(grid)
            if len(np.ravel(positions_temp)) != np.size(y_sub_temp, axis=0):
                # point array does not match spec array
                print "            point array size does not match y array size, trimming to match y!"
                max_rows = np.size(y_sub_temp, axis=0)/np.size(positions_temp, axis=1)
                print "                complete rows found:", max_rows
                max_spec = max_rows * np.size(positions_temp, axis=1)
                print "                complete scan size: %0.f x %0.f" % (max_rows, np.size(positions_temp, axis=1))
                print "                    %s points in total" % max_spec
                y_sub_temp = y_sub_temp[:max_spec]
                positions_temp = positions_temp[:max_rows,:]
                grid = np.mgrid[0:np.size(positions_temp, axis=0), 0:np.size(positions_temp, axis=1)]
                print "                trimmed spectra:", np.shape(y_sub_temp)
                print "                trimmed positions:", np.shape(positions_temp)
            # determine point-to-point spacing from filename
            spacing_temp = 100.
            for temp in file_name.split("_"):
                if len(temp) > 2:
                    if temp[-2:] == 'um':
                        spacing_temp = float(temp[:-2])
            # convert point array to coordinates
            coords_temp = (spacing_temp/1000.) * np.roll(grid, 1, axis=0)
            print "        dimensions of scan: %0.1f x %0.1f mm" % (coords_temp[0,0,-1], coords_temp[1,-1,0])
            print "    adding data to storage arrays..."
            # add data to storage arrays
            samples.append(file_name.split("_")[0])
            scans.append(file_name.split("_")[1])
            map_positions.append(positions_temp)
            xy_coords.append(coords_temp)
            raman_shift.append(shift_temp)
            y_sub.append(y_sub_temp)
            y_sub_av.append(np.mean(y_sub_temp, axis=0))
            y_sub_std.append(np.std(y_sub_temp, axis=0))
            print "    successfully imported %s" % file_name
            break
        except Exception as e:
            print "    something went wrong! Exception:", e
            break
            
print
print

"""
==================================================
begin data processing
==================================================
"""

# specify Regions of Interest in each scan that will be averaged for fitting
ROIs = {
    'SAU-008_Region1': {'radius': 0.2, 'labels': ['Olivine', 'Plagioclase', 'Pyroxene', 'Glass'], 'centers': [(1.3, 1.8), (2.6, 1.7), (2.3, 0.9), (4.5, 0.5)]},
    'SAU-008_Region2': {'radius': 0.2, 'labels': ['Olivine', 'Plagioclase', 'Pyroxene', 'Carbonate'], 'centers': [(0.8, 0.9), (2.8, 1.6), (2.2, 0.2), (1.7, 0.8)]},
    'SAU-008_Region3': {'radius': 0.2, 'labels': ['Olivine', 'Plagioclase', 'Pyroxene', 'Glass', 'Carbonate', 'Phosphate', 'Phosphate', 'Phosphate', 'Phosphate', 'Phosphate'], 'centers': [(4.0, 2.0), (3.0, 0.4), (1.7, 0.8), (4.0, 0.6), (3.5, 1.3), (0.8, 2.1), (1.1, 2.1), (1.8, 1.6), (0.6, 1.6), (1.3, 1.2)]}
} # each named ROI is a circle centred on specified position with given radius

# specify spectral region to fit
x_start, x_end = (900, 2000)

# specify which calculated parameters to plot as false color maps, and with which colormaps
maps_to_plot = ['R1_parameter', 'R2_parameter', 'RA1_parameter', 'RA2_parameter', 'G_center', 'G_fwhm', 'D1_center']
cmaps = ['viridis', 'inferno', 'viridis', 'inferno', 'plasma', 'cividis']

for i in range(len(samples)):
    # for each imported scan
    print
    print "fitting D,G peaks for %s..." % samples[i]
    
    # prepare data arrays
    start_time = datetime.datetime.now()
    last_time = datetime.datetime.now()
    spec_count = np.size(y_sub[i], axis=0)
    arrtemp = np.zeros(spec_count)
    DG_props = {'R1_parameter': np.copy(arrtemp), 'R2_parameter': np.copy(arrtemp), 'RA1_parameter': np.copy(arrtemp), 'RA2_parameter': np.copy(arrtemp)}
    for peak in DG_peaks['peak']:
        DG_props["%s_center" % peak] = np.copy(arrtemp)
        DG_props["%s_amplitude" % peak] = np.copy(arrtemp)
        DG_props["%s_integrint" % peak] = np.copy(arrtemp)
        DG_props["%s_fwhm" % peak] = np.copy(arrtemp)
        if DG_fit_function.lower() in ['fd', 'fermidirac', 'fermi-dirac'] and peak != 'O2':
            DG_props['%s_rounding' % peak] = np.copy(arrtemp)
        elif DG_fit_function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt'] and peak != 'O2':
            DG_props['%s_eta' % peak] = np.copy(arrtemp)
        
    print "%s spectra to fit" % spec_count
    for i2 in range(spec_count):
        # for each spectrum in scan
        print
        print "fitting spectrum %s of %s..." % (i2, spec_count)
        noise = np.std(y_sub[i][i2][np.where((2000 <= raman_shift[i]) & (raman_shift[i] <= 2100))])
        x_slice = raman_shift[i][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
        y_slice = y_sub[i][i2][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
        fit_output, fit_curve, DG_params = DG_fit_script(x_slice, y_slice, DG_peaks, function=DG_fit_function, free_baseline=True, window=DG_window, max_fwhm=100.)
        # add data to storage arrays
        DG_props['R1_parameter'][i2] = DG_params['R1_parameter']
        DG_props['R2_parameter'][i2] = DG_params['R2_parameter']
        DG_props['RA1_parameter'][i2] = DG_params['RA1_parameter']
        DG_props['RA2_parameter'][i2] = DG_params['RA2_parameter']
        for peak in DG_peaks['peak']:
            DG_props['%s_center' % peak][i2] = DG_params['%s_center' % peak]
            DG_props['%s_amplitude' % peak][i2] = DG_params['%s_amplitude' % peak]
            DG_props['%s_fwhm' % peak][i2] = DG_params['%s_fwhm' % peak]
            DG_props['%s_integrint' % peak][i2] = DG_params['%s_integrint' % peak]
            if DG_fit_function.lower() in ['fd', 'fermidirac', 'fermi-dirac'] and peak != 'O2':
                DG_props['%s_rounding' % peak][i2] = DG_params['%s_rounding' % peak]
            elif DG_fit_function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt'] and peak != 'O2':
                DG_props['%s_eta' % peak][i2] = DG_params['%s_eta' % peak]
        print "    fit %s of %s complete! Time taken: %s (HH:MM:SS)" % (i2, spec_count, datetime.datetime.now() - last_time)
        last_time = datetime.datetime.now()
        print "        %s remaining fits will take %s" % (spec_count-i2, (spec_count-i2)*(last_time-start_time)/(i2+1))
        print
        
    # save fit results to file
    print
    print "all DG fits done for %s, saving to file..." % scans[i]
    cols = ['R1_parameter', 'R2_parameter', 'RA1_parameter', 'RA2_parameter']
    for peak in DG_peaks['peak']:
        for prop in ['center', 'fwhm', 'amplitude', 'integrint']:
            cols += ["%s_%s" % (peak, prop)]
        if DG_fit_function.lower() in ['fd', 'fermidirac', 'fermi-dirac'] and peak != 'O2':
            cols += ['%s_rounding' % peak]
        elif DG_fit_function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt'] and peak != 'O2':
            cols += ['%s_eta' % peak]
    save_data = np.vstack([DG_props[key] for key in cols])
    frame = pd.DataFrame(save_data.transpose(), columns=cols, index=range(spec_count))
    print
    print "columns:", frame.columns
    print "rows:", frame.index
    frame.to_csv("%s%s/DG_properties_SAU-008_%s.csv" % (Output_dir, samples[i], scans[i]))
    
print
print
print "DONE"
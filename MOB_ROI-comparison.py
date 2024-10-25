"""
==================================================

This script takes the ROIs outputted from different MOBIUS scans and compares them to each other. It also fits the organic D&G band region for each ROI based on the standardised fitting method described in the MOB_DG-fitting script.

This script is designed to accept processed files outputted by the MOB_R-mapping and MOB_F-mapping scripts.

==================================================
"""

import os
import glob
import math
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
Plot_maps = True        # plot DG parameters as 2D maps?

# color parameters for plotting
Cmap = 'viridis'        # default matplotlib colormap to use for false color maps
Color_list =  ['orange', 'limegreen', 'deeppink', 'darkgoldenrod', 'cornflowerblue', 'red', 'green', 'tab:purple', 'slategray', 'm', 'tab:brown', 'c', 'b', 'tab:orange']            # matplotlib single colors for line and scatter plots
Marker_list = ['o', 's', 'v', '^', 'D', '*']    # matplotlib marker styles for scatter plots

# ==================================================
# functions

def find_max(x, y, x_start, x_end):
    # function for finding the maximum from a slice of input data.
    x_slice = x[np.where((x_start <= x) & (x <= x_end))]       # create slice
    y_slice = y[np.where((x_start <= x) & (x <= x_end))]
    i = np.argmax(y_slice)
    return np.asarray([x_slice[i], y_slice[i]]) # return x,y position of the maximum

def find_min(x, y, x_start, x_end):
    # function for finding the maximum from a slice of input data.
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
# functions for fitting the D,G peaks

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
            if function.lower() in ['lbwf']:
                if peak == 'G':
                    # use a BWF function for G specifically
                    gamma = params['%s_fwhm' % peak] / 2.
                    model += A * ((1 + 2*(x - mu) / (params['%s_Q' % peak] * gamma))**2) / (1 + (2*(x - mu) / gamma)**2)
                else:
                    # all other peaks get a Lorentzian function
                    gamma = params['%s_fwhm' % peak] / 2.
                    model += A * (gamma**2)/((x - mu)**2 + gamma**2)
            elif function.lower() in ['fd', 'fermidirac', 'fermi-dirac']:
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
            if function.lower() in ['lbwf']:
                if peak == 'G':
                    # use a BWF function for G specifically
                    gamma = params['%s_fwhm' % peak] / 2.
                    model += A * (1 + 2*(x - mu) / (params['%s_Q' % peak] * gamma))**2 / (1 + (2*(x - mu) / gamma)**2)
                else:
                    # all other peaks get a Lorentzian function
                    gamma = params['%s_fwhm' % peak] / 2.
                    model += A * (gamma**2)/((x - mu)**2 + gamma**2)
            elif function.lower() in ['fd', 'fermidirac', 'fermi-dirac']:
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

def DG_fit_script(x, y, maxima={'peak': [], 'center': []}, function='g', free_baseline=True, window=30, max_fwhm=100.):
    # script for preparing data for D&G peak fitting and collating results
    global DG_peaks
    if len(maxima['peak']) == 0:
        maxima = DG_peaks
    print
    print "    input x:", np.shape(x)
    print "    input y:", np.shape(y)
    print "    input peaks:", maxima['peak']
    print "    input peak positions:", maxima['center']
    params = lmfit.Parameters()
    params.add('gradient', value=0., vary=free_baseline)
    params.add('intercept', value=np.amin(y), vary=free_baseline)
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
            if function.lower() in ['lbwf'] and peak == 'G':
                params.add("%s_Q" % peak, value=1)
            if function.lower() in ['fd', 'fermidirac', 'fermi-dirac']:
                params.add('%s_round' % peak, value=2., min=0.)
            elif function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt']:
                params.add('%s_eta' % peak, value=0.5, min=0., max=1.) 
    print "        initial parameters:"
    print params.pretty_print()
    fit_output = lmfit.minimize(DG_fit, params, args=(x, y, maxima, function))
    fit_curve = DG_curve(x, fit_output.params, maxima, function=function)
    print "        fit status: ", fit_output.message
    print "        fitted parameters:"
    print fit_output.params.pretty_print()
    # create summary of all relevant D&G output properties
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
    for key in properties.keys():
        print "    %s: %0.2f" % (key, properties[key])
        
    # calculate additional properties (R1: Rahl 2005, R2: Beyssac 2002, RA1&2: Lahfid 2010)
    if 'D1' in maxima['peak'] and 'G' in maxima['peak']:
        properties['R1_parameter'] = properties['D1_amplitude'] / properties['G_amplitude']
        print "R1: %0.2f" % properties['R1_parameter']
        if 'D2' in maxima['peak']:
            properties['R2_parameter'] = properties['D1_integrint'] / (properties['G_integrint'] + properties['D1_integrint'] + properties['D2_integrint'])
            print "R2: %0.2f" % properties['R2_parameter']
            if 'D3' in maxima['peak'] and 'D4' in maxima['peak']:
                properties['RA1_parameter'] = (properties['D1_integrint'] + properties['D4_integrint']) / (properties['G_integrint'] + properties['D1_integrint'] + properties['D2_integrint'] + properties['D3_integrint'] + properties['D4_integrint'])
                print "RA1: %0.2f" % properties['RA1_parameter']
                properties['RA2_parameter'] = (properties['D1_integrint'] + properties['D4_integrint']) / (properties['G_integrint'] + properties['D2_integrint'] + properties['D3_integrint'])
                print "RA2: %0.2f" % properties['RA2_parameter']
    return fit_output, fit_curve, properties

"""
# ==================================================
# import processed Raman spectral maps
# ==================================================
"""

spec_dirs = sorted(glob.glob("%sSAU-008/SAU-008/*/*_Rmap.txt" % Output_dir))

print "processed spectra files found:", len(spec_dirs)

sample_av = []
scan_av = []
raman_shift_av = []
y_sub = []
y_av_sub = []
std_sub = []

for spec_dir in spec_dirs:
    while True:
        try:
            file_name = spec_dir.split("/")[-1]
            print
            print "attempting to import", file_name
            spec = np.genfromtxt(spec_dir, delimiter="\t")
            if np.size(spec, axis=0) in [1024,1025,2048,2049]:
                spec = np.transpose(spec)
            shift_temp = spec[1]
            y_sub_temp = spec[2:]
            print "    spectra in map:", np.shape(y_sub_temp)
            print "    adding data to storage arrays..."
            sample_av.append(file_name.split("_")[0])
            scan_av.append(file_name.split("_")[1])
            raman_shift_av.append(shift_temp)
            y_sub.append(y_sub_temp)
            y_av_sub.append(np.mean(y_sub_temp, axis=0))
            std_sub.append(np.std(y_sub_temp, axis=0))
            print "    successfully imported %s" % file_name
            break
        except Exception as e:
            print "    something went wrong! Exception:", e
            break
            
print
print

x_start, x_end = (800, 4000)

if len(sample_av) > 0:
    x_temp = np.linspace(x_start, x_end, int(x_end-x_start))
    y_temp = []
    for i in range(0, len(scan_av)):
        x_slice = raman_shift_av[i][np.where((x_start <= raman_shift_av[i]) & (raman_shift_av[i] <= x_end))]
        for i2 in range(0, len(y_sub[i])):
            y_slice = y_sub[i][i2][np.where((x_start <= raman_shift_av[i]) & (raman_shift_av[i] <= x_end))]
            y_temp.append(np.interp(x_temp, x_slice, y_slice))
    y_temp = np.asarray(y_temp)
    y_temp_av = np.mean(y_temp, axis=0)
    y_temp_std = np.std(y_temp, axis=0)
    print np.shape(x_temp), np.shape(y_temp), np.shape(y_temp_av), np.shape(y_temp_std)
    
    plt.figure(figsize=(8,4))
    ax1 = plt.subplot(111)
    ax1.set_xlim(x_start, x_end)
    ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax1.set_ylabel("Average Intensity (counts)")
    # add wavelength ticks to top of ax1
    ax1_top = ax1.twiny()
    x_range = ax1.get_xlim()
    major_labels, major_locations, minor_locations = get_wavelength_ticks(x_range)
    ax1_top.set_xticks(major_locations)
    ax1_top.set_xticklabels(major_labels)
    ax1_top.xaxis.set_minor_locator(plt.FixedLocator(minor_locations))
    ax1_top.set_xlabel("Wavelength (nm)")
    if plot_scan_avs == True:
        # plot individual scan spectra
        for i in range(0, len(scan_av)):
            x_slice = raman_shift_av[i][np.where((x_start <= raman_shift_av[i]) & (raman_shift_av[i] <= x_end))]
            y_slice = y_av_sub[i][np.where((x_start <= raman_shift_av[i]) & (raman_shift_av[i] <= x_end))]
            if plot_std == True:
                std_slice = std_sub[i][np.where((x_start <= raman_shift_av[i]) & (raman_shift_av[i] <= x_end))]
                ax1.fill_between(x_slice, y_slice-std_slice, y_slice+std_slice, color=Color_list[i], alpha=0.1)
            ax1.plot(x_slice, y_slice, Color_list[i], label=scan_av[i])
    # plot average spectrum
    if plot_std == True:
        ax1.fill_between(x_temp, y_temp_av-y_temp_std, y_temp_av+y_temp_std, color='k', alpha=0.1)
    ax1.plot(x_temp, y_temp_av, 'k', label='average')
    y_max = np.amax(y_temp_av)
    ax1.set_ylim(-0.2*y_max, 1.2*y_max)
    ax1.legend()
    plt.savefig("%sSAU-008/SAU-008/SAU-008_Raman_by_scan.png" % Figure_dir, dpi=300)
    plt.savefig("%sSAU-008/SAU-008/SAU-008_Raman_by_scan.svg" % Figure_dir, dpi=300)
    plt.show()

"""
# ==================================================
# import processed Raman spectra for ROIs
# ==================================================
"""

# search for spec files
spec_dirs = sorted(glob.glob("%sSAU-008/SAU-008/*/*_Raman_POIs.txt" % Output_dir))

print "ROI files found:", len(spec_dirs)

samples = []
scans = []
raman_shift = []
ROI_names = []
ROI_av_sub = []
nonROI_av_sub = []
wavelength = []
ROI_fluor = []

# ROI names for each scan, in order
Manual_ROI_names = {
    'Region1': ['Olivine', 'Plagioclase', 'Pyroxene', 'Glass', 'Carbonate'],
    'Region2': ['Olivine', 'Plagioclase', 'Pyroxene', 'Carbonate'],
    'Region3': ['Olivine', 'Plagioclase', 'Pyroxene', 'Glass', 'Carbonate', 'Phosphate', 'Phosphate', 'Phosphate', 'Phosphate', 'Phosphate']
}

ROI_count = 0
for spec_dir in spec_dirs:
    while True:
        try:
            file_name = spec_dir.split("/")[-1]
            print
            print "attempting to import", file_name
            file_name = spec_dir.split("/")[-1]
            file_name_split = file_name.split("_")
            print "    ", file_name_split
            print "    sample:", file_name_split[0]
            print "    scan:", file_name_split[1]
            spec = np.genfromtxt(spec_dir, delimiter="\t")
            if np.size(spec, axis=0) in [1024,1025,2048,2049]:
                spec = np.transpose(spec)
            print "    spec array:", np.shape(spec)
            wavelength_temp = spec[0]
            raman_shift_temp = spec[1]
            nonROI_spec_temp = spec[2]
            ROI_spec_temp = spec[3:]
            print "        ROI spectra found:", np.size(ROI_spec_temp, axis=0)
            if file_name_split[1] in Manual_ROI_names.keys():
                ROI_names_temp = Manual_ROI_names[file_name_split[1]]
            else:
                ROI_names_temp = ["ROI-%0.f" % i for i in range(len(ROI_spec_temp))]
            print "        ROI names:", ROI_names_temp
            if len(ROI_names_temp) == len(ROI_spec_temp):
                print "    adding data to storage arrays..."
                samples.append(file_name_split[0])
                scans.append(file_name_split[1])
                raman_shift.append(raman_shift_temp)
                nonROI_av_sub.append(nonROI_spec_temp)
                ROI_av_sub.append(ROI_spec_temp)
                ROI_names.append(ROI_names_temp)
                ROI_count += len(ROI_spec_temp)
                print "    successfully imported %s!" % file_name
            print "%sSAU-008/SAU-008/%s_%s_*/*_Fluor_ROIs.txt" % (Output_dir, file_name_split[0], file_name_split[1])
            fluor_dirs = sorted(glob.glob("%sSAU-008/SAU-008/%s_%s_*/*_Fluor_ROIs.txt" % (Output_dir, file_name_split[0], file_name_split[1])))
            print "fluorescence ROI files found:", len(fluor_dirs)
            if len(fluor_dirs) > 0:
                fluor_spec = np.genfromtxt(fluor_dirs[0], delimiter="\t")
                if np.size(fluor_spec, axis=0) in [1024,1025,2048,2049]:
                    fluor_spec = np.transpose(fluor_spec)
                print "    fluor spec array:", np.shape(fluor_spec)
                if np.size(fluor_spec, axis=0) == len(ROI_names_temp) + 2:
                    print "        same number of ROIs found!"
                    wavelength.append(fluor_spec[0])
                    ROI_fluor.append(fluor_spec[2:])
                else:
                    print "        number of F ROIs does not match number of R ROIs!"
                    wavelength.append(np.nan)
                    ROI_fluor.append(np.nan)
            else:
                print "        no F ROI file found!"
                wavelength.append(np.nan)
                ROI_fluor.append(np.nan)
            break
        except Exception as e:
            print "    something went wrong! Exception:", e
            break
            
print
print "%s scans imported, with a total of %s ROI spectra" % (len(scans), ROI_count)

# manually specify unique ROI names, or leave empty [] for auto
unique_ROIs = ['Olivine', 'Plagioclase', 'Pyroxene', 'Glass', 'Phosphate', 'Carbonate']

if len(unique_ROIs) == 0:
    unique_ROIs = np.unique(ROI_names)

print
print "ROIs found:", unique_ROIs

"""
# ==================================================
# plot ROIs against each other
# ==================================================
"""

x_start, x_end = (900, 1800)
F_start, F_end = (250, 410)
y_max = 100.

# plot average R and F spectra for each mineral ROI
plt.figure(figsize=(8,8))
# plot R spectra
plt.subplot(211)
plt.title("Normalised Average Spectra by ROI")
plt.xlim(x_start, x_end)
plt.xlabel("Raman Shift (cm$^{-1}$)")
offset = 0
ROI_count = 0
x_temp = np.linspace(x_start, x_end, x_end-x_start)
for ROI in unique_ROIs:
    print ROI
    y_temp = []
    for i in range(len(scans)):
        for i2 in range(len(ROI_names[i])):
            if ROI_names[i][i2] == ROI:
                y_temp.append(np.interp(x_temp, raman_shift[i], ROI_av_sub[i][i2]))
    if len(y_temp) > 1:
        x_slice = x_temp[np.where((x_start <= x_temp) & (x_temp <= x_end))]
        y_slice = np.mean(y_temp, axis=0)[np.where((x_start <= x_temp) & (x_temp <= x_end))]
        if Smooth == True:
            y_slice = savgol_filter(y_slice, 5, 3)
        y_slice /= find_max(x_temp, np.mean(y_temp, axis=0), x_start, x_end)[1]
        """elif ROI == 'Carbonate':
            y_slice /= 10."""
        plt.plot(x_slice, y_slice+offset*ROI_count, c=Color_list[ROI_count], label=ROI)
        ROI_count += 1
plt.ylim(-0.2, 1.2)
plt.ylabel("Normalised Intensity")
plt.legend()
plt.minorticks_on()
# plot F spectra
plt.subplot(212)
plt.xlim(F_start, F_end)
plt.xlabel("Wavelength (nm)")
offset = 0
ROI_count = 0
x_temp = np.linspace(F_start, F_end, F_end-F_start)
for ROI in unique_ROIs:
    print ROI
    y_temp = []
    for i in range(len(scans)):
        for i2 in range(len(ROI_names[i])):
            if ROI_names[i][i2] == ROI:
                y_temp.append(np.interp(x_temp, wavelength[i], ROI_fluor[i][i2]))
    if len(y_temp) > 1:
        x_slice = x_temp[np.where((F_start <= x_temp) & (x_temp <= F_end))]
        y_slice = np.mean(y_temp, axis=0)[np.where((F_start <= x_temp) & (x_temp <= F_end))]
        if Smooth == True:
            y_slice = savgol_filter(y_slice, 5, 3)
        y_slice /= y_slice[np.argmin(np.abs(x_slice - 370))]
        plt.plot(x_slice, y_slice+offset*ROI_count, c=Color_list[ROI_count], label=ROI)
        ROI_count += 1
plt.ylim(-0.2, 2.5)
plt.ylabel("Intensity (Normalised to 370 nm)")
plt.legend()
plt.minorticks_on()
plt.savefig("%sSAU-008/SAU-008/SAU-008_ROIspectra.png" % Figure_dir, dpi=300)
plt.savefig("%sSAU-008/SAU-008/SAU-008_ROIspectra.svg" % Figure_dir, dpi=300)
plt.show()

x_start, x_end = (800, 1300)
y_max = 10.

# plot ROIs for each separate scan
fig, axs = plt.subplots(nrows=len(scans), sharey=True, figsize=(8,2+2*len(scans)))
for i, ax in enumerate(axs):
    ax.set_xlim(x_start, x_end)
    for i2 in range(len(ROI_names[i])):
        x_slice = raman_shift[i][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
        y_slice = ROI_av_sub[i][i2][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
        if Smooth == True:
            y_slice = savgol_filter(y_slice, 5, 3)
        if Normalise == True:
            y_slice /= find_max(raman_shift[i], ROI_av_sub[i][i2], x_start, x_end)[1]
        ax.plot(x_slice, y_slice, c=Color_list[i2], label=ROI_names[i][i2])
    x_slice = raman_shift[i][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
    y_slice = y_av_sub[i][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
    if Normalise == True:
        y_slice /= find_max(raman_shift[i], y_av_sub[i], x_start, x_end)[1]
    ax.plot(x_slice, y_slice, 'k', label=scans[i]+" mean")
    ax.legend(loc=2)
    if Normalise == True:
        ax.set_ylim(-0.4, 1.2)
        ax.set_ylabel("Norm. Int.")
    else:
        ax.set_ylim(-0.4*y_max, 1.2*y_max)
        ax.set_ylabel("Rel. Int.")
    if i == len(scans)-1:
        ax.set_xlabel("Raman Shift (cm$^{-1}$)")
plt.minorticks_on()
plt.savefig("%sSAU-008/SAU-008/SAU-008_ROIs_by_scan.png" % Figure_dir, dpi=300)
plt.savefig("%sSAU-008/SAU-008/SAU-008_ROIs_by_scan.svg" % Figure_dir, dpi=300)
plt.show()

# plot ROIs for each scan
for i in range(len(scans)):
    plt.figure(figsize=(8,4))
    plt.title(scans[i])
    plt.xlim(x_start, x_end)
    ROI_count = 0
    if Normalise == True:
        offset = 1.2
    else:
        offset = 1.2*y_max
    labels = []
    for ROI in unique_ROIs:
        count = 0
        for i2 in range(len(ROI_names[i])):
            if ROI_names[i][i2] == ROI:
                count += 1
                x_slice = raman_shift[i][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
                y_slice = ROI_av_sub[i][i2][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
                if Smooth == True:
                    y_slice = savgol_filter(y_slice, 5, 3)
                if Normalise == True:
                    y_slice /= find_max(raman_shift[i], ROI_av_sub[i][i2], x_start, x_end)[1]
                """elif ROI == 'Carbonate':
                    y_slice /= 10."""
                label = "_%s" % (ROI)
                if label not in labels:
                    labels.append(label)
                    label = label[1:]
                plt.plot(x_slice, y_slice+offset*ROI_count, c=Color_list[ROI_count], label=label)
        if count > 0:
            ROI_count += 1
    plt.legend(loc=2)
    if Normalise == True:
        plt.ylim(-0.4, 0.4+offset*ROI_count)
        plt.ylabel("Normalised Intensity (offset)")
    else:
        plt.ylim(-0.4*y_max, 0.4*y_max+offset*ROI_count)
        plt.ylabel("Relative Intensity (offset)")
    if offset > 0:
        plt.yticks([])
    plt.minorticks_on()
    plt.savefig("%sSAU-008/SAU-008/SAU-008_%s_ROIs.png" % (Figure_dir, scans[i]), dpi=300)
    plt.savefig("%sSAU-008/SAU-008/SAU-008_%s_ROIs.svg" % (Figure_dir, scans[i]), dpi=300)
    plt.show()

# plot specific ROIs against each other from across scans
fig, axs = plt.subplots(nrows=len(unique_ROIs), sharey=True, figsize=(8,2+2*len(unique_ROIs)))
count = 0
for ROI in unique_ROIs:
    print ROI
    axs[count].set_xlim(x_start, x_end)
    x_temp = np.linspace(x_start, x_end)
    y_temp = []
    scan_count = 0
    for i in range(len(scans)):
        for i2 in range(len(ROI_names[i])):
            if ROI_names[i][i2] == ROI:
                x_slice = raman_shift[i][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
                y_slice = ROI_av_sub[i][i2][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
                if Smooth == True:
                    y_slice = savgol_filter(y_slice, 5, 3)
                if Normalise == True:
                    y_slice /= find_max(raman_shift[i], ROI_av_sub[i][i2], x_start, x_end)[1]
                label = "_%s %s" % (ROI, scans[i])
                axs[count].plot(x_slice, y_slice, c=Color_list[count], label=label)
                y_temp.append(np.interp(x_temp, raman_shift[i], ROI_av_sub[i][i2]))
    if len(y_temp) > 1:
        x_slice = x_temp[np.where((x_start <= x_temp) & (x_temp <= x_end))]
        y_slice = np.mean(y_temp, axis=0)[np.where((x_start <= x_temp) & (x_temp <= x_end))]
        if Normalise == True:
            y_slice /= find_max(raman_shift[i], ROI_av_sub[i][i2], x_start, x_end)[1]
        axs[count].plot(x_slice, y_slice, 'k', label=ROI+" mean")
    if Normalise == True:
        axs[i].set_ylim(-0.4, 1.2)
        axs[i].set_ylabel("Norm. Int.")
    else:
        axs[i].set_ylim(-0.4*y_max, 1.2*y_max)
        axs[i].set_ylabel("Rel. Int.")
    axs[count].legend(loc=2)
    if i == len(scans)-1:
        axs[count].set_xlabel("Raman Shift (cm$^{-1}$)")
    count += 1
plt.minorticks_on()
plt.savefig("%sSAU-008/SAU-008/SAU-008_ROIs_by_mineral.png" % Figure_dir, dpi=300)
plt.savefig("%sSAU-008/SAU-008/SAU-008_ROIs_by_mineral.svg" % Figure_dir, dpi=300)
plt.show()

# plot unique Raman ROIs against each other from across scans, in a single plot with offset
plt.figure(figsize=(8,2+1*len(unique_ROIs)))
count = 0
if Normalise == True:
    offset = 1.2
else:
    offset = 1.2*y_max
plt.xlim(x_start, x_end)
plt.xlabel("Raman Shift (cm$^{-1}$)")
labels = []
x_temp = np.linspace(x_start, x_end, x_end-x_start)
y_temp = []
for ROI in unique_ROIs:
    print ROI
    scan_count = 0
    y_temp.append([])
    for i in range(len(scans)):
        for i2 in range(len(ROI_names[i])):
            if ROI_names[i][i2] == ROI:
                x_slice = raman_shift[i][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
                y_slice = ROI_av_sub[i][i2][np.where((x_start <= raman_shift[i]) & (raman_shift[i] <= x_end))]
                if Smooth == True:
                    y_slice = savgol_filter(y_slice, 5, 3)
                if Normalise == True:
                    y_slice /= find_max(raman_shift[i], ROI_av_sub[i][i2], x_start, x_end)[1]
                elif ROI == 'Carbonate':
                    y_slice /= 10.
                label = "_%s" % (ROI)
                if label not in labels:
                    labels.append(label)
                    label = label[1:]
                plt.plot(x_slice, y_slice+offset*count, c=Color_list[count], label=label)
                y_temp[count].append(np.interp(x_temp, raman_shift[i], ROI_av_sub[i][i2]))
    if len(y_temp[count]) > 1:
        x_slice = x_temp[np.where((x_start <= x_temp) & (x_temp <= x_end))]
        y_slice = np.mean(y_temp[count], axis=0)[np.where((x_start <= x_temp) & (x_temp <= x_end))]
        if Smooth == True:
            y_slice = savgol_filter(y_slice, 5, 3)
        if Normalise == True:
            y_slice /= find_max(x_temp, np.mean(y_temp[count], axis=0), x_start, x_end)[1]
        elif ROI == 'Carbonate':
            y_slice /= 10.
        plt.plot(x_slice, y_slice+offset*count, 'k')
    count += 1
if Normalise == True:
    plt.ylim(-0.4, 0.4++offset*count)
    plt.ylabel("Normalised Intensity (offset)")
else:
    plt.ylim(-0.4*y_max, 0.4*y_max+offset*count)
    plt.ylabel("Relative Intensity (offset)")
if offset > 0:
    plt.yticks([])
plt.legend(loc=2)
plt.minorticks_on()
plt.savefig("%sSAU-008/SAU-008/SAU-008_ROIs_by_mineral_v2.png" % Figure_dir, dpi=300)
plt.savefig("%sSAU-008/SAU-008/SAU-008_ROIs_by_mineral_v2.svg" % Figure_dir, dpi=300)
plt.show()

# plot mean spectra for each ROI type
plt.figure(figsize=(8,4))
plt.xlim(x_start, x_end)
count = 0
for ROI in ['Olivine', 'Plagioclase', 'Pyroxene']:
    x_slice = x_temp[np.where((x_start <= x_temp) & (x_temp <= x_end))]
    y_slice = np.mean(y_temp[count], axis=0)[np.where((x_start <= x_temp) & (x_temp <= x_end))]
    if Smooth == True:
        y_slice = savgol_filter(y_slice, 5, 3)
    if Normalise == True:
        y_slice /= find_max(x_temp, np.mean(y_temp[count], axis=0), x_start, x_end)[1]
    plt.plot(x_slice, y_slice, c=Color_list[count], label=ROI)
    count += 1
if Normalise == True:
    plt.ylim(-0.4, 1.2)
    plt.ylabel("Normalised Intensity (offset)")
else:
    plt.ylim(-0.4*y_max, 1.2*y_max)
    plt.ylabel("Relative Intensity (counts)")
if offset > 0:
    plt.yticks([])
plt.legend(loc=2)
plt.minorticks_on()
plt.show()

"""
==================================================
do specialised D&G peak fitting
==================================================
"""

if Fit_DG == True:
    print
    print "fitting D&G peaks..."
    
    # prepare data arrays
    arrtemp = np.zeros(len(unique_ROIs)+1)
    DG_props = {'sample': [], 'R1_parameter': np.copy(arrtemp)}
    if 'D2' in DG_peaks['peak']:
        DG_props["R2_parameter"] = np.copy(arrtemp)
        if 'D3' in DG_peaks['peak'] and 'D4' in DG_peaks['peak']:
            DG_props["RA1_parameter"] = np.copy(arrtemp)
            DG_props["RA2_parameter"] = np.copy(arrtemp)
    for peak in DG_peaks['peak']:
        DG_props["%s_center" % peak] = np.copy(arrtemp)
        DG_props["%s_amplitude" % peak] = np.copy(arrtemp)
        DG_props["%s_integrint" % peak] = np.copy(arrtemp)
        DG_props["%s_fwhm" % peak] = np.copy(arrtemp)
        if DG_fit_function.lower() in ['fd', 'fermidirac', 'fermi-dirac'] and peak != 'O2':
            DG_props['%s_rounding' % peak] = np.copy(arrtemp)
        elif DG_fit_function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt'] and peak != 'O2':
            DG_props['%s_eta' % peak] = np.copy(arrtemp)
    
    x_start, x_end = (900, 1800)
    labels = []
    y_avs = []
    
    x_temp = np.linspace(800, 4000, 4000-800)
    y_temp = []
    for i in range(0, len(scan_av)):
        x_slice = raman_shift_av[i][np.where((x_start <= raman_shift_av[i]) & (raman_shift_av[i] <= x_end))]
        for i2 in range(0, len(y_sub[i])):
            y_slice = y_sub[i][i2][np.where((x_start <= raman_shift_av[i]) & (raman_shift_av[i] <= x_end))]
            y_temp.append(np.interp(x_temp, x_slice, y_slice))
    y_temp = np.asarray(y_temp)
    y_temp_av = np.mean(y_temp, axis=0)
    y_temp_std = np.std(y_temp, axis=0)
    y_avs.append(y_temp_av)
    labels.append("average")
    noise = np.std(y_temp[np.where((2000 <= x_temp) & (x_temp <= 2100))])
    x_slice = x_temp[np.where((x_start <= x_temp) & (x_temp <= x_end))]
    y_slice = y_temp_av[np.where((x_start <= x_temp) & (x_temp <= x_end))]
    std_slice = y_temp_std[np.where((x_start <= x_temp) & (x_temp <= x_end))]
    fit_output, fit_curve, DG_params = DG_fit_script(x_slice, y_slice, DG_peaks, function=DG_fit_function, free_baseline=True, window=DG_window)
    # add data to storage arrays
    DG_props['sample'].append("SAU-008_average")
    DG_props['R1_parameter'][0] = DG_params['R1_parameter']
    if 'D2' in DG_peaks['peak']:
        DG_props['R2_parameter'][0] = DG_params['R2_parameter']
        if 'D3' in DG_peaks['peak'] and 'D4' in DG_peaks['peak']:
            DG_props['RA1_parameter'][0] = DG_params['RA1_parameter']
            DG_props['RA2_parameter'][0] = DG_params['RA2_parameter']
    for peak in DG_peaks['peak']:
        DG_props['%s_center' % peak][0] = DG_params['%s_center' % peak]
        DG_props['%s_amplitude' % peak][0] = DG_params['%s_amplitude' % peak]
        DG_props['%s_fwhm' % peak][0] = DG_params['%s_fwhm' % peak]
        DG_props['%s_integrint' % peak][0] = DG_params['%s_integrint' % peak]
        if DG_fit_function.lower() in ['fd', 'fermidirac', 'fermi-dirac'] and peak != 'O2':
            DG_props['%s_rounding' % peak][0] = DG_params['%s_rounding' % peak]
        elif DG_fit_function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt'] and peak != 'O2':
            DG_props['%s_eta' % peak][0] = DG_params['%s_eta' % peak]
    # create summary figure for fit
    plt.figure(figsize=(10,6))
    # ax1: results of fit
    ax1 = plt.subplot2grid((4,5), (0,0), colspan=4, rowspan=3)
    ax1.set_title("SaU-008 Average Spectrum\nD&G Peak Fitting with %s" % (DG_fit_function))
    ax1.set_ylabel("Average Intensity")
    ax1.minorticks_on()
    # ax2: residuals
    ax2 = plt.subplot2grid((4,5), (3,0), colspan=4, sharex=ax1)
    ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax2.set_ylabel("Residual")
    # histogram of residuals
    ax3 = plt.subplot2grid((4,5), (3,4))
    ax3.set_yticks([])
    # determine y limits for residual, hist plots
    y_min = np.amin(y_slice-fit_curve)
    y_max = np.amax(y_slice-fit_curve)
    res_min = y_min - 0.1*(y_max-y_min)
    res_max = y_max + 0.1*(y_max-y_min)
    ax2.set_ylim(res_min, res_max)
    # report fitted peak properties
    count = 0
    y_max = np.amax(y_slice)-np.amin(y_slice)
    x_temp = np.linspace(x_start, x_end, 10*len(x_slice))
    for peak in DG_peaks['peak']:
        plt.figtext(0.78, 0.9-0.1*count, "%s: %4.1f\nSNR=%0.1f\nFWHM=%0.1f" % (peak, DG_params['%s_center' % peak], DG_params['%s_amplitude' % peak]/noise, DG_params['%s_fwhm' % peak]))
        background = fit_output.params['gradient'].value * DG_params['%s_center' % peak] + fit_output.params['intercept'].value
        ax1.vlines(DG_params['%s_center' % peak], background, DG_params['%s_amplitude' % peak]+background, colors=Color_list[count % len(Color_list)], linestyles=':')
        curve = DG_curve(x_temp, fit_output.params, function=DG_fit_function, peak=peak)
        ax1.plot(x_temp, curve, c=Color_list[count % len(Color_list)], linestyle='--', label="%s fit" % peak)
        ax1.text(DG_params['%s_center' % peak], y_slice[np.argmin(np.abs(x_slice-DG_params['%s_center' % peak]))]+0.1*y_max, peak, ha='center')
        count += 1
    count = 0
    for parameter in ['R1_parameter', 'R2_parameter', 'RA1_parameter', 'RA2_parameter']:
        if parameter in DG_params.keys():
            plt.figtext(0.30, 0.85-0.04*count, "%s: %0.2f" % (parameter.split("_")[0], DG_params[parameter]))
            count += 1
    # plot total fit and baseline curves
    ax1.plot(x_temp, fit_output.params['gradient'].value * x_temp + fit_output.params['intercept'].value, 'k--', label="baseline")
    total_curve = DG_curve(x_temp, fit_output.params, function=DG_fit_function)
    ax1.plot(x_temp, total_curve, 'm', label="total fit")
    # plot input data and residuals
    ax1.plot(x_slice, y_slice, 'k', label="data")
    ax2.plot(x_slice, y_slice-fit_curve, 'k')
    ax3.hist(y_slice-fit_curve, range=(res_min, res_max), bins=20, orientation='horizontal', color='k')
    # finish fitted region figure
    y_min = np.amin(y_slice)
    y_max = np.amax(y_slice)
    ax1.set_xlim(x_start, x_end)
    ax1.set_ylim(y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min))
    ax1.legend(loc='upper left')
    ax2.axhline(0., color='k', linestyle=":")
    ax3.axhline(0., color='w', linestyle=":")
    # save figure to sample figure directory
    plt.savefig('%saverage_DG_fit.png' % (Figure_dir), dpi=300)
    plt.savefig('%saverage_DG_fit.svg' % (Figure_dir), dpi=300)
    plt.show()
    
    # repeat for each unique ROI type
    ROI_count = 0
    for ROI in unique_ROIs:
        x_temp = np.linspace(800, 4000, 4000-800)
        y_temp = []
        count = 0
        for i in range(len(scans)):
            for i2 in range(len(ROI_names[i])):
                if ROI_names[i][i2] == ROI:
                    y_temp.append(np.interp(x_temp, raman_shift[i], ROI_av_sub[i][i2]))
                    count += 1
        if count > 0:
            ROI_count += 1
            y_av = np.mean(y_temp, axis=0)
            std = np.std(y_temp, axis=0)
            y_avs.append(y_av)
            labels.append(ROI)
            noise = np.std(y_av[np.where((2000 <= x_temp) & (x_temp <= 2100))])
            x_slice = x_temp[np.where((x_start <= x_temp) & (x_temp <= x_end))]
            y_slice = y_av[np.where((x_start <= x_temp) & (x_temp <= x_end))]
            std_slice = std[np.where((x_start <= x_temp) & (x_temp <= x_end))]
            fit_output, fit_curve, DG_params = DG_fit_script(x_slice, y_slice, DG_peaks, function=DG_fit_function, free_baseline=True, window=DG_window)
            # add data to storage arrays
            DG_props['sample'].append("SAU-008_%s" % ROI)
            DG_props['R1_parameter'][ROI_count] = DG_params['R1_parameter']
            if 'D2' in DG_peaks['peak']:
                DG_props['R2_parameter'][ROI_count] = DG_params['R2_parameter']
                if 'D3' in DG_peaks['peak'] and 'D4' in DG_peaks['peak']:
                    DG_props['RA1_parameter'][ROI_count] = DG_params['RA1_parameter']
                    DG_props['RA2_parameter'][ROI_count] = DG_params['RA2_parameter']
            for peak in DG_peaks['peak']:
                DG_props['%s_center' % peak][ROI_count] = DG_params['%s_center' % peak]
                DG_props['%s_amplitude' % peak][ROI_count] = DG_params['%s_amplitude' % peak]
                DG_props['%s_fwhm' % peak][ROI_count] = DG_params['%s_fwhm' % peak]
                DG_props['%s_integrint' % peak][ROI_count] = DG_params['%s_integrint' % peak]
                if DG_fit_function.lower() in ['fd', 'fermidirac', 'fermi-dirac'] and peak != 'O2':
                    DG_props['%s_rounding' % peak][ROI_count] = DG_params['%s_rounding' % peak]
                elif DG_fit_function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt'] and peak != 'O2':
                    DG_props['%s_eta' % peak][ROI_count] = DG_params['%s_eta' % peak]
            # create summary figure for fit
            plt.figure(figsize=(10,6))
            # ax1: results of fit
            ax1 = plt.subplot2grid((4,5), (0,0), colspan=4, rowspan=3)
            ax1.set_title("SaU-008 %s ROI mean\nD&G Peak Fitting with %s" % (ROI, DG_fit_function))
            ax1.set_ylabel("Average Intensity")
            # ax2: residuals
            ax2 = plt.subplot2grid((4,5), (3,0), colspan=4, sharex=ax1)
            ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
            ax2.set_ylabel("Residual")
            # histogram of residuals
            ax3 = plt.subplot2grid((4,5), (3,4))
            ax3.set_yticks([])
            # determine y limits for residual, hist plots
            y_min = np.amin(y_slice-fit_curve)
            y_max = np.amax(y_slice-fit_curve)
            res_min = y_min - 0.1*(y_max-y_min)
            res_max = y_max + 0.1*(y_max-y_min)
            ax2.set_ylim(res_min, res_max)
            # report fitted peak properties
            count = 0
            y_max = np.amax(y_slice) - np.amin(y_slice)
            x_temp = np.linspace(x_start, x_end, 10*len(x_slice))
            for peak in DG_peaks['peak']:
                plt.figtext(0.78, 0.9-0.1*count, "%s: %4.1f\nSNR=%0.1f\nFWHM=%0.1f" % (peak, DG_params['%s_center' % peak], DG_params['%s_amplitude' % peak]/noise, DG_params['%s_fwhm' % peak]))
                background = fit_output.params['gradient'].value * DG_params['%s_center' % peak] + fit_output.params['intercept'].value
                ax1.vlines(DG_params['%s_center' % peak], background, DG_params['%s_amplitude' % peak]+background, colors=Color_list[count % len(Color_list)], linestyles=':')
                curve = DG_curve(x_temp, fit_output.params, function=DG_fit_function, peak=peak)
                ax1.plot(x_temp, curve, c=Color_list[count % len(Color_list)], linestyle='--', label="%s fit" % peak)
                ax1.text(DG_params['%s_center' % peak], y_slice[np.argmin(np.abs(x_slice-DG_params['%s_center' % peak]))]+0.1*y_max, peak, ha='center')
                count += 1
            count = 0
            for parameter in ['R1_parameter', 'R2_parameter', 'RA1_parameter', 'RA2_parameter']:
                if parameter in DG_params.keys():
                    plt.figtext(0.30, 0.85-0.04*count, "%s: %0.2f" % (parameter.split("_")[0], DG_params[parameter]))
                    count += 1
            # plot total fit and baseline curves
            ax1.plot(x_temp, fit_output.params['gradient'].value * x_temp + fit_output.params['intercept'].value, 'k--', label="baseline")
            total_curve = DG_curve(x_temp, fit_output.params, function=DG_fit_function)
            ax1.plot(x_temp, total_curve, 'm', label="total fit")
            # plot input data and residuals
            ax1.plot(x_slice, y_slice, 'k', label="data")
            ax2.plot(x_slice, y_slice-fit_curve, 'k')
            ax3.hist(y_slice-fit_curve, range=(res_min, res_max), bins=20, orientation='horizontal', color='k')
            # finish fitted region figure
            y_min = np.amin(y_slice)
            y_max = np.amax(y_slice)
            ax1.set_xlim(x_start, x_end)
            ax1.set_ylim(y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min))
            ax1.legend(loc='upper left')
            ax2.axhline(0., color='k', linestyle=":")
            ax3.axhline(0., color='w', linestyle=":")
            # save figure to sample figure directory
            plt.savefig('%s%s_DG_fit.png' % (Figure_dir, ROI), dpi=300)
            plt.savefig('%s%s_DG_fit.svg' % (Figure_dir, ROI), dpi=300)
            plt.show()
    
    # save data
    print
    print "saving ROI spectra to file"
    
    Save_to_ascii = True

    def prepend_multiple_lines(file_name, list_of_lines):
            dummy_file = file_name + '.bak'
            with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
                for line in list_of_lines:
                    write_obj.write(line + '\n')
                for line in read_obj:
                    write_obj.write(line)
            os.rename(dummy_file, file_name)
    
    print
    print "saving ROI DG fits to file"
    cols = ['R1_parameter', 'R2_parameter', 'RA1_parameter', 'RA2_parameter']
    for peak in DG_peaks['peak']:
        for prop in ['center', 'fwhm', 'amplitude', 'integrint']:
            cols += ["%s_%s" % (peak, prop)]
        if DG_fit_function.lower() in ['fd', 'fermidirac', 'fermi-dirac'] and peak != 'O2':
            cols += ['%s_rounding' % peak]
        elif DG_fit_function.lower() in ['pv', 'pseudovoigt', 'pseudo-voigt'] and peak != 'O2':
            cols += ['%s_eta' % peak]
    save_data = np.vstack([DG_props[key] for key in cols])
    frame = pd.DataFrame(save_data.transpose(), columns=cols, index=DG_props['sample'])
    print
    print "columns:", frame.columns
    print "rows:", frame.index
    frame.to_csv("%sSAU-008/SAU-008/DG_properties_SAU-008.csv" % (Output_dir))

    x_temp = np.linspace(800, 4000, 4000-800)
    header = ["wavelength (nm)", "Raman shift (cm-1)", "Average Baselined Spectrum"] + ["%s Spectrum" % ROI for ROI in unique_ROIs]
    print np.shape(header), np.shape(x_temp), np.shape(y_avs)
    save_data = np.vstack((shift2wavelength(x_temp), x_temp, y_avs))
    
    np.savetxt("%sSAU-008/SAU-008/SAU-008_ROIspectra.txt" % (Output_dir), save_data.transpose(), delimiter='\t', header="\t".join(header))
    np.savetxt("%sSAU-008/SAU-008/SAU-008_ROIspectra.csv" % (Output_dir), save_data.transpose(), delimiter=',', header=",".join(header))
    
if Plot_maps == True:
    print
    print "plotting DG fit maps"
    
    DG_dirs = sorted(glob.glob("%sSAU-008/SAU-008/*/DG_properties_*.csv" % Output_dir))
    print DG_dirs
    
    print
    print "DG map files found:", len(DG_dirs)
    
    DG_maps = {}
    
    maps_to_plot = ['R1_parameter', 'R2_parameter', 'RA1_parameter', 'RA2_parameter', 'G_center', 'G_fwhm', 'D1_center', 'D1_fwhm']
    cmaps = ['viridis', 'inferno', 'viridis', 'inferno', 'plasma', 'cividis']
    
    for DG_dir in DG_dirs:
        print
        print DG_dir
        folder_name = DG_dir.split("/")[-2]
        sample_name = folder_name.split("_")[0]
        scan_name = folder_name.split("_")[1]
        print "importing DG map for %s %s" % (sample_name, scan_name)
        dataframe = pd.read_csv(DG_dir, header=0)
        print "    ", np.shape(dataframe)
        print "        importing point position array..."
        positions_dir = glob.glob("%sSAU-008/%s/working_files/points_array.csv" % (Data_dir, folder_name))
        print "            ", positions_dir
        positions_temp = np.genfromtxt(positions_dir[0], delimiter=',', dtype='int')[1:,1:]
        print "            position array", np.shape(positions_temp)
        grid = np.mgrid[0:np.size(positions_temp, axis=0), 0:np.size(positions_temp, axis=1)]
        print "            grid:", np.shape(grid)
        if len(np.ravel(positions_temp)) != np.size(dataframe.index.values):
                print "            point array size does not match y array size, trimming to match y!"
                max_rows = np.size(dataframe.index.values)/np.size(positions_temp, axis=1)
                print "                complete rows found:", max_rows
                max_spec = max_rows * np.size(positions_temp, axis=1)
                print "                complete scan size: %0.f x %0.f" % (max_rows, np.size(positions_temp, axis=1))
                print "                    %s points in total" % max_spec
                dataframe = dataframe.iloc[:max_spec]
                positions_temp = positions_temp[:max_rows,:]
                grid = np.mgrid[0:np.size(positions_temp, axis=0), 0:np.size(positions_temp, axis=1)]
                print "                trimmed spectra:", np.shape(dataframe.index.values)
                print "                trimmed positions:", np.shape(positions_temp)
        spacing_temp = 100.
        for temp in folder_name.split("_"):
            if len(temp) > 2:
                if temp[-2:] == 'um':
                    spacing_temp = float(temp[:-2])
        coords_temp = (spacing_temp/1000.) * np.roll(grid, 1, axis=0)
        print "    dimensions of scan: %0.1f x %0.1f mm" % (coords_temp[0,0,-1], coords_temp[1,-1,0])
        DG_maps[scan_name] = {
            'sample': sample_name,
            'scan': scan_name,
            'figdir': "%s%s/%s/%s/" % (Figure_dir, sample_name, sample_name, folder_name),
            'map_positions': positions_temp,
            'xy_coords': coords_temp,
            'DG_props': dataframe
        }
        print "    data added to storage array"
    
    vlims = {}
    for key in maps_to_plot:
        temp = []
        for scan in sorted(DG_maps.keys()):
            temp.append(np.ravel(DG_maps[scan]['DG_props'][key]))
        temp = np.sort(np.concatenate(temp))
        print key, np.shape(temp),
        clip = int(0.02*np.size(temp))
        print clip, temp[clip], temp[-clip]
        vlims[key] = (temp[clip], temp[-clip])
    
    aspects = 0
    for scan in sorted(DG_maps.keys()):
        positions = DG_maps[scan]['map_positions']
        DG_props = DG_maps[scan]['DG_props']
        aspect = float(np.size(positions, axis=0)) / float(np.size(positions, axis=1))
        aspects += aspect
        print
        print scan, np.size(positions, axis=0), np.size(positions, axis=1)
        x_coords, y_coords = (DG_maps[scan]['xy_coords'])
        fig, axs = plt.subplots(nrows=len(maps_to_plot)/2, ncols=2, figsize=(8/aspect,4*len(maps_to_plot)/2))
        fig.suptitle("%s %s DG fitting with %s" % (DG_maps[scan]['sample'], DG_maps[scan]['scan'], DG_fit_function))
        print np.shape(axs), axs
        for i, ax in zip(range(len(maps_to_plot)), axs.reshape(-1)):
            print i, ax
            key = maps_to_plot[i]
            cmap = 'viridis'
            print i, key, cmap
            ax.set_title(str(key))
            print np.shape(np.asarray(DG_props[key]))
            print np.shape(positions)
            print np.shape(np.asarray(DG_props[key])[positions])
            band_map = np.asarray(DG_props[key])[positions]
            vmin, vmax = vlims[key]
            im = ax.imshow(band_map, extent=(0., np.amax(x_coords), np.amax(y_coords), 0.), cmap=cmap, vmin=vmin, vmax=vmax)
            plt.imsave('%s%s_%s_DG_fit_%s_map.png' % (DG_maps[scan]['figdir'], DG_maps[scan]['sample'], DG_maps[scan]['scan'], key), np.repeat(np.repeat(band_map, 5, axis=0), 5, axis=1), cmap=cmap, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
        fig.tight_layout()
        fig.savefig('%s%s_%s_DG_fit-map.png' % (DG_maps[scan]['figdir'], DG_maps[scan]['sample'], DG_maps[scan]['scan']), dpi=300)
        fig.savefig('%s%s_%s_DG_fit-map.svg' % (DG_maps[scan]['figdir'], DG_maps[scan]['sample'], DG_maps[scan]['scan']), dpi=300)
        fig.show()
    
print
print
print "DONE"
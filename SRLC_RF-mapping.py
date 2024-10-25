"""
==================================================

This script processes Raman&Fluorescence spectral maps from the SHERLOC instrument. Processing includes:
    1) spectral recalibration and recombination
    2) baseline fitting and subtraction
    3) region and point of interest averaging
    4) automatic peak detection
    5) automatic peak fitting
    
This script then generates figures, including:
    1) intensity maps at user-defined raman shifts
    2) comparison of user-defined ROI average spectra to reference materials
    3) comparison of user-defined POI spectra to reference materials
    4) results from peak detection and fitting
    6) intensity correlation maps

This script requires pre-processed data files generated using Loupe, publicly available software for handling raw SHERLOC data.
- Uckert, K. (2022). nasa/Loupe: LoupeV5.1.5 (Version LoupeV5.1.5a) [Computer software]. https://doi.org/10.5281/ZENODO.7062998

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

# specify a target to import (str, or use '*')
Target = 'Meteorite'

# specify sol number (4 digit string, or '*')
Sol = '*'

# specify scan name (string, or '*')
Scan_name = '*'

# specify pre-processed data type from the following:
# '*' for best available Loupe data (dictated by SRLC_scan_properties.csv)
# 'Loupe' for raw, unprocessed data
# 'Loupe_N' for laser-normalised data
# 'Loupe_C' for laser-normalised data with cosmic rays removed
Data_formatting = '*'

# import Autofocus Context Imager image for scan as well? Required for plotting maps against ACI
Import_ACI = True
ACI_pixel_scale = 0.0101  # pixel scan in mm/px

# define the directories for input/output:
Data_dir =  './data/SRLC/'
Output_dir = './output/SRLC/'
Figure_dir = './figures/SRLC/'
BB_data_dir =  './refs/BB/output/*/'
MOB_R_data_dir = './refs/MOBIUS/output/Raman/'
MOB_F_data_dir = './refs/MOBIUS/output/Fluor/'

# if there are hydration peaks present, set True to do specialised baselining
Hydration = True

# turn on/off each plotting section
Plot_on_ACI = True              # also plot maps as true-to-size spots overlaid on ACI image
Plot_Fmaps = True              # Fluorescence maps
Plot_Rmaps = True              # Raman maps
Compare = True                 # compare to terrestrial standards
Plot_ROIs = True               # plot Raman ROIs based on x,y coordinates
Plot_POIs = True               # plot Raman POIs based on point indices
Compare_Fs = True              # compare ROI/POI fluorescence as well
Intensity_correlation = True   # plot intensity correlations

# do ROI/POI peak fitting?
Fit_ROIs = True
Fit_function = 'Gaussian'       # function to use for fitting, choose from 'Gaussian' or 'Fermi-Dirac'
Manual_fit_peaks = []           # additional peaks to include in fit

# set band centers and widths for RGB 3-channel mapping (or [] for default list)
F_RGB_bands = {
    'center': [275, 305, 340],
    'width': [5, 5, 5]
}

R_RGB_bands = {
    'center': [],
    'width': []
}

# specify any number of fluorescence bands (and their widths) for band-by-band mapping, leave empty for default values
Fluor_bands = {
    'center': [285, 300, 325, 340, 305],
    'width': [5, 5, 5, 5, 5]
}

# specify Raman bands and half-widths for channel mapping, leave empty for default values
Raman_bands = {
    'center': [840, 975, 1010, 1090, 1600, 3300, 3550],
    'width': [25, 25, 25, 25, 50, 75, 75]
}

# specify map normalisation: 'all' = all bands normalised to overall min/max; 'separate': each band normalised on its own; 'None': no normalisation
Band_normalisation = 'all'
# specify lower and upper intensity limits for standardised map normalisation (set to "(None, None)" to auto-scale each map)
Cmin, Cmax = (None, None)
Cmap = 'viridis'

# define Raman shift range for inset figures
inset_limits = (750, 2000)

# plot +/-1 st. dev.? True/False
Plot_std = False

# specify Brassboard standards for comparison ([] to use default list)
BB_samples = []
BB_F_pulses = 25
BB_F_current = 20
BB_R_pulses = 800
BB_R_current = 15

# specify MOBIUS standards for comparison ([] to skip)
MOB_samples = []
MOB_F_pulses = 25
MOB_F_current = 15
MOB_R_pulses = 1200
MOB_R_current = 15

# list of default  standards to compare for each target
Default_comparisons = {
    'Default': ['Apatite_sxtal', 'Gypsum_powder', 'Enstatite_C7229', 'Calcite_powder', 'Quartz_powder']
}

# list of default Fluor bands to map for each target
Default_F_bands = {
    'Default': {'center': [275, 305, 325, 340], 'width': [5, 5, 5, 5]}
}

# list of default Raman bands to map for each target
Default_R_bands = {
    'Default': {'center': [840, 975, 1010, 1090, 1600, 3300, 3550], 'width': [25, 25, 25, 25, 50, 75, 75]}
}

# wavelength calibration parameters
sherloc_calib_params = [246.69, 6.524e-2, -7.85e-6, 247.5, 6.34e-2, -5.66e-6]

# override stored wavelength data and generate fresh values using sherloc_calib_params?
Wavelength_override = True

# color parameters for plotting
Color_list =  ['r', 'b', 'm', 'y', 'g', 'tab:gray', 'c', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:purple']

# ==================================================
# functions

def sherloc_calibration(x, params=sherloc_calib_params):
    # takes SHERLOC pixel channels and converts to wavelength
    new_x = np.zeros_like(x)
    # handle first calibration region
    if np.size(x) == 1:
        # if the input contains a single value
        if x < 500:
            new_x = params[2]*x**2 + params[1]*x + params[0]
        else:
            new_x = params[5]*x**2 + params[4]*x + params[3]
    else:
        # if the input is a list or array
        x = np.asarray(x)
        new_x = params[2]*x**2 + params[1]*x + params[0]
        if np.any(x >= 500):
            i = x >= 500
            new_x[i] = params[5]*x[i]**2 + params[4]*x[i] + params[3]
    return new_x

def sherloc_reverse_calibration(x, params=sherloc_calib_params, debug=False):
    # takes wavelength values and converts to equivalent SHERLOC pixel channels
    if debug == True:
        print "nans in input:", np.count_nonzero(np.isnan(x))
    new_x = np.zeros_like(x)
    # do first calibration region (valid from pix=0 to pix=500)
    new_x = (np.sqrt(4.*params[2]*(x - params[0]) + params[1]**2) - params[1])/(2.*params[2])
    if np.any(4.*params[2]*(x - params[0]) + params[1]**2 < 0):
        # handle any nans caused by negative values inside square root (when x > 401 nm)
        i = 4.*params[2]*(x - params[0]) + params[1]**2 < 0
        new_x[i] = 501
    if debug == True:
        print "nans in 1st calibration:", np.count_nonzero(np.isnan(new_x))
        print new_x
    # do second calibration region (valid from pix=501 to x=426 nm)
    if np.size(x) > 1:
        # if the input is a list or array containing more than 1 value
        if np.any(new_x) >= 500:
            # if any of those values are in the second calibration region
            i = new_x > 500
            if debug == True:
                print "values in 2nd calibration region", np.count_nonzero(i)
            new_x[i] = (np.sqrt(4.*params[5]*(x[i] - params[3]) + params[4]**2) - params[4])/(2.*params[5])
        if debug == True:
            print "input wavelength range: %0.1f to %0.1f nm" % (np.amin(x), np.amax(x))
            print "output pixel range: %0.1f to %0.1f" % (np.amin(new_x), np.amax(new_x))
    else:
        # if the input is a single value
        if new_x > 500:
            # and that value falls in the second calibration region
            new_x = (np.sqrt(4.*params[5]*(x - params[3]) + params[4]**2) - params[4])/(2.*params[5])
        if debug == True:
            print "input: %0.1f nm, output: %0.1f" % (x, new_x)
    return new_x

def spectral_correction(x, params):
    # linear spectral recalibration
    new_x = params[0]*x + params[1]
    return new_x

def quadratic_correction(x, params):
    # quadratic spectral recalibration
    new_x = np.zeros_like(x)
    new_x = params[2]*x**2 + params[1]*x + params[0]
    return new_x

def wavelength2shift(wavelength, excitation=248.5794):
    # convert wavelength (nm) to Raman shift (cm-1)
    shift = ((1./excitation) - (1./wavelength)) * (10**7)
    return shift

def shift2wavelength(shift, excitation=248.5794):
    # convert Raman shift (cm-1) to wavelength (nm)
    wavelength = 1./((1./excitation) - shift/(10**7))
    return wavelength

def smooth_f(y, window_length, polyorder=3):
    # function for smoothing data based on Savitsky-Golay filtering
    if window_length % 2 != 1:
        window_length += 1
    if polyorder < window_length:
        polyorder = window_length-1
    y_smooth = savgol_filter(y, window_length, polyorder)
    return y_smooth

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

def find_maxima(x, y, window_length, threshold):
    # function for finding the maxima of input data. Each maximum will have the largest value within its window.
    index_list = argrelextrema(y, np.greater, order=window_length)  # determines indices of all maxima
    # print "indices: ", index_list
    all_maxima = np.asarray([x[index_list], y[index_list]]) # creates an array of x and y values for all maxima
    # print "x:", all_maxima[0]
    # print "y:", all_maxima[1]
    y_limit = threshold * np.amax(y)                        # set the minimum threshold for defining a 'peak'
    # print "threshold: ", y_limit
    # print all_maxima[1] > y_limit
    x_maxima = all_maxima[0, all_maxima[1] >= y_limit]      # records the x values for all valid maxima
    y_maxima = all_maxima[1, all_maxima[1] >= y_limit]      # records the y values for all valid maxima
    maxima = np.asarray([x_maxima, y_maxima])               # creates an array for all valid maxima
    # print "maxima:"
    # print "x: ", maxima[0]
    # print "y: ", maxima[1]
    return maxima

def average_list(x, y, point_list, window, debug=False):
    # function for taking a set of user-defined points and creating arrays of their average x and y values
    if debug == True:
        print
        print "        ", point_list
    x_averages = np.zeros_like(point_list, dtype=float)
    y_averages = np.zeros_like(point_list, dtype=float)
    point_num = 0
    for i in range(np.size(point_list)):
        point_num += 1
        x_averages[i], y_averages[i] = local_average(x, y, point_list[i], window)
        if debug == True:
            print "        point", str(point_num), ": ", x_averages[i], y_averages[i]
    return x_averages, y_averages

def local_average(x, y, x_0, w):
    # function for finding the average position from a set of points, centered on 'x_0' with 'w' points either side
    center_ind = np.argmin(np.absolute(x - x_0))
    start_ind = center_ind - w
    end_ind = center_ind + w
    x_temp = x[start_ind:end_ind]
    y_temp = y[start_ind:end_ind]
    x_average = (np.average(x_temp))
    y_average = (np.average(y_temp))
    return x_average, y_average

def f_linbase(x, *params):
    # function for generating a linear baseline
    a, b = params
    y = a*x + b
    return y
    
def f_expbase(x, *params):
    # function for generating an exponential baseline
    a, b = params
    y = a * np.exp(x*b)
    return y

def f_sinebase(x, *params):
    # function for generating an exponential baseline
    a, b, c, d = params
    y = a * np.cos(x/b+c) + d
    return y

def f_polybase(x, *params):
    # function for generating an exponential baseline
    y = params[0]
    for i in range(1, len(params)):
        y += params[i] * x**i
    return y

def expbase_fit(x_averages, y_averages, sigma, debug=False):
    # function for fitting selected average data-points using an exponential function
    guess = [1., 0.05]
    if debug == True:
        print "        initial parameters: ", guess
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_expbase, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print "        fitted parameters: ", fit_coeffs
    return fit_coeffs, fit_covar

def sinebase_fit(x_averages, y_averages, sigma, debug=False):
    # function for fitting selected average data-points using a sine function
    guess = [(np.amax(y_averages)-np.amin(y_averages))/2., (np.amax(x_averages)-np.amin(x_averages))/4., 0., np.mean(y_averages)]
    if debug == True:
        print "        initial parameters: ", guess
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_sinbase, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print "        fitted parameters: ", fit_coeffs
    return fit_coeffs, fit_covar

def linbase_fit(x_averages, y_averages, sigma, debug=False):
    # function for fitting selected average data-points using a linear function
    guess = [1., 0.]
    if debug == True:
        print "        initial parameters: ", guess
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_linbase, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print "        fitted parameters: ", fit_coeffs
    return fit_coeffs, fit_covar

def polybase_fit(x_averages, y_averages, sigma, max_order=15, debug=False):
    # function for fitting selected average data-points using a polynominal function
    if len(x_averages) > int(max_order):
        guess = np.zeros((int(max_order)))
    else:
        guess = np.zeros_like(x_averages)
    if debug == True:
        print "        initial parameters: ", guess
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_polybase, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print "        fitted parameters:", fit_coeffs
    return fit_coeffs, fit_covar

def baseline(x, y, i, x_list, base='poly', max_order=15, window=25, hydration=False, find_minima=True, fixed_ends=True, debug=False, plot=False, name=None):
    global data
    # calculate baseline and subtract it
    if name == None:
        name = data['scan_name'][i]
    # smooth data for fitting
    y_s = smooth_f(y, 5, 3)
    # find point positions and fit with curve
    if find_minima == True:
        points = []
        for point in [point for point in x_list if point > np.amin(x)+window and point < np.amax(x)-window]:
            points.append(find_min(x, y_s, point-window, point+window)[0])
    else:
        points = [point for point in x_list if point > np.amin(x)+window and point < np.amax(x)-window]
    if debug == True:
        print "points:", points
    # create arrays of average values for each point +/-5 pixels
    x_averages, y_averages = average_list(x, y_s, points, 5, debug=debug)
    # add fixed first and last points if applicable, and create sigma array for point weighting
    sigma = np.ones_like(y_averages)
    if fixed_ends == True:
        for index in [5, np.argmin(np.absolute(x-4000))]:
            x_0, y_0 = local_average(x, y_s, x[index], 5)
            x_averages = np.insert(x_averages, -1, x_0)
            y_averages = np.insert(y_averages, -1, y_0)
            sigma = np.insert(sigma, -1, 0.1)
    # tidy up and sort point lists
    sort = np.argsort(x_averages)
    x_averages = x_averages[sort]
    y_averages = y_averages[sort]
    sigma = sigma[sort]
    # run curve_fit script with polynomial
    if base in ['lin', 'linear']:
        fit_coeffs, fit_covar = linbase_fit(x_averages, y_averages, sigma, debug=debug)
        y_basefit = f_linbase(x, *fit_coeffs)
    elif base in ['exp', 'exponential']:
        fit_coeffs, fit_covar = expbase_fit(x_averages, y_averages, sigma, debug=debug)
        y_basefit = f_expbase(x, *fit_coeffs)
    elif base in ['sin', 'sine', 'sinewave']:
        fit_coeffs, fit_covar = sinebase_fit(x_averages, y_averages, sigma, debug=debug)
        y_basefit = f_sinebase(x, *fit_coeffs)
    else:
        if max_order > len(y_averages)-1:
            max_order = len(y_averages)-1
        fit_coeffs, fit_covar = polybase_fit(x_averages, y_averages, sigma, max_order=max_order, debug=debug)
        y_basefit = f_polybase(x, *fit_coeffs)
    # do separate hydrate region subtraction
    if hydration == True:
        hyd_start = 3000
        hyd_end = 3800
        if np.amin(x_averages) < hyd_start and np.amax(x_averages) > hyd_end:
            i0 = np.argmin(np.absolute(x - hyd_start))
            i1 = np.argmin(np.absolute(x - hyd_end))
            if debug == True:
                print "hydration start:", i0, x[i0], y_basefit[i0]
                print "hydration end:", i1, x[i1], y_basefit[i1]
            hyd_grad = (y_basefit[i1]-y_basefit[i0])/(x[i1]-x[i0])
            hyd_int = y_basefit[i0]
            if debug == True:
                print "gradient:", hyd_grad
                print "intercept:", hyd_int
                print "region to replace: %0.f-%0.f cm" % (x[i0], x[i1]), np.shape(x[np.where((x[i0] <= x) & (x <= x[i1]))])
            y_basefit[np.where((x[i0] <= x) & (x <= x[i1]))] = hyd_grad * (x[np.where((x[i0] <= x) & (x <= x[i1]))]-x[i0]) + hyd_int
    # subtract polynomial from data
    y_B = y - y_basefit
    if plot == True:
        # create figure
        plt.figure(figsize=(8,8))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        if name != None:
            ax1.set_title("Baseline-Corrected Raman Spectrum\n%s" % (str(name)))
        else:
            ax1.set_title("Baseline-Corrected Raman Spectrum")
        ax1.set_ylabel("Average Intensity")
        ax2.set_ylabel("Average Intensity")
        ax2.set_xlabel("Raman shift (cm$^{-1}$)")
        # plot points and fit in ax1
        ax1.plot(x, y, 'k')
        ax1.plot(x_averages, y_averages, 'or', label='points')
        ax1.plot(x, y_basefit, 'r', label='baseline')
        # plot before and after in ax2
        ax2.plot(x, y, 'k', label='before')
        ax2.plot(x, y_B, 'b', label='after')
        ax2.legend(loc=1)
        y_max = find_max(x_averages, y_averages, np.amin(x_list), np.amax(x_list))[1]
        y_min = find_min(x_averages, y_averages, np.amin(x_list), np.amax(x_list))[1]
        ax1.set_ylim(y_min-0.5*(y_max-y_min), y_min+1.5*(y_max-y_min))
        ax1.set_xlim(np.amin(x_list)-100, np.amax(x_list)+100)
        y_max = find_max(x, y, np.amin(x_list), np.amax(x_list))[1]
        y_min = find_min(x, y, np.amin(x_list), np.amax(x_list))[1]
        ax2.set_ylim(-0.5*(y_max-y_min), y_min+1.5*(y_max-y_min))
        ax2.set_xlim(np.amin(x_list)-100, np.amax(x_list)+100)
        plt.tight_layout()
        plt.savefig("%sSol-%s_%s_av_base.png" % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
        plt.show()
    return y_B

# ==================================================
# functions for fitting overlapping peaks

def G_curve(x, params):
    # generate a gaussian curve
    model = np.zeros_like(x)
    gradient = params['gradient']
    intercept = params['intercept']
    A = params['amplitude']
    mu = params['center']
    sigma = params['sigma']
    model += A * np.exp(-0.5*(x - mu)**2/(sigma**2)) + gradient*x + intercept
    return model

def multiG_curve(x, params, maxima):
    # generate multiple gaussian curves
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiG_fit(params, x, y, maxima):
    # function for fitting data using multiple gaussian curves
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def multiG_fit_script(x, y, maxima, window=30., max_sigma=60., debug=False):
    # script for setting up and running a multi-gaussian fit
    print
    if debug == True:
        print "    input x:", np.shape(x)
        print "    input y:", np.shape(y)
        print "    input maxima:", np.shape(maxima), maxima
        print "    input window:", window
        print "    input max_sigma:", max_sigma
    params = lmfit.Parameters()
    params.add('gradient', value=0.)
    params.add('intercept', value=np.amin(y))
    for i in range(0, len(maxima)):
        y_max = y[np.argmin(np.absolute(x - maxima[i]))]
        params.add('center_%s' % i, value=maxima[i], min=maxima[i]-window, max=maxima[i]+window)
        params.add('amplitude_%s' % i, value=y_max, min=0.)
        params.add('sigma_%s' % i, value=20., min=10., max=2.*max_sigma)
    print "        initial parameters:"
    print params.pretty_print()
    fit_output = lmfit.minimize(multiG_fit, params, args=(x, y, maxima))
    fit_curve = multiG_curve(x, fit_output.params, maxima)
    print "        fit status: ", fit_output.message
    print "        fitted parameters:"
    print fit_output.params.pretty_print()
    return fit_output, fit_curve

# ==================================================
# functions for fitting peaks with symmetric Fermi-Dirac functions

def FD_curve(x, params):
    # generate a fermi-dirac curve
    model = np.zeros_like(x)
    A = params['amplitude']
    mu = params['center']
    W = params['width']
    R = params['round']
    gradient = params['gradient']
    intercept = params['intercept']
    model = A * expit(-((np.absolute(x - mu) - W) / R)) + gradient*x + intercept
    return model

def multiFD_curve(x, params, maxima):
    # generate multiple fermi-dirac curves
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        W = params['width_%s' % i]
        R = params['round_%s' % i]
        model += A * expit(-((np.absolute(x - mu) - W) / R))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def FD_fit(params, x, y, y_std):
    # function for fitting data using a fermi-dirac curve
    model = np.zeros_like(x)
    A = params['amplitude']
    mu = params['center']
    W = params['width']
    R = params['round']
    gradient = params['gradient']
    intercept = params['intercept']
    model = A * expit(-((np.absolute(x - mu) - W) / R)) + gradient*x + intercept
    return (y-model) / y_std

def multiFD_fit(params, x, y, maxima):
    # function for fitting data using multiple fermi-dirac curves
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        W = params['width_%s' % i]
        R = params['round_%s' % i]
        model += A * expit(-((np.absolute(x - mu) - W) / R))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def FD_fit_script(x, y, y_std, maxima):
    # script for setting up and running a fermi-dirac fit
    print
    params = lmfit.Parameters()
    params.add('amplitude', value=maxima[1], min=0)
    params.add('center', value=maxima[0])
    params.add('width', value=10., min=0)
    params.add('round', value=2., min=0)
    params.add('gradient', value=0.)
    params.add('intercept', value=np.amin(y))
    print "initial parameters:"
    print params.pretty_print()
    fit_output = lmfit.minimize(FD_fit, params, args=(x, y, y_std))
    fit_curve = FD_curve(x, fit_output.params)
    print "fit status: ", fit_output.message
    print "fitted parameters:"
    print fit_output.params.pretty_print()
    return fit_output, fit_curve

def multiFD_fit_script(x, y, maxima, window=30.):
    # script for setting up and running a multiple fermi-dirac fit
    print
    params = lmfit.Parameters()
    for i in range(0, len(maxima)):
        y_max = y[np.argmin(np.absolute(x - maxima[i]))]
        params.add('amplitude_%s' % i, value=y_max, min=0, max=1.2*np.amax(y))
        params.add('center_%s' % i, value=maxima[i], min=maxima[i]-window, max=maxima[i]+window)
        params.add('width_%s' % i, value=10., min=0)
        params.add('round_%s' % i, value=2., min=0)
    params.add('gradient', value=0.)
    params.add('intercept', value=np.amin(y))
    print "initial parameters:"
    print params.pretty_print()
    fit_output = lmfit.minimize(multiFD_fit, params, args=(x, y, maxima))
    fit_curve = multiFD_curve(x, fit_output.params, maxima)
    print "fit status: ", fit_output.message
    print "fitted parameters:"
    print fit_output.params.pretty_print()
    return fit_output, fit_curve

# ==================================================
# functions for handling shift/wavelength conversion in figure axes

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
# functions for creating spectral intensity maps

def serpentine(height, width, orientation='horizontal', starting_index=0, debug=False):
    # function for indexing a 1D array into a 2D array according to a serpentine (back and forth) pattern
    indices = np.arange(height*width)
    # determine shape of 2D array
    # create 2D array
    if debug == True:
        print "    generating serpentine index array..."
        print "        input height, width: %d x %d, %s" % (height, width, orientation)
    if 'vertical' in orientation:
        # vertical serpentine
        # begin with horizontal array (using swapped axes)
        index_array = np.reshape(indices, (width, height))
        # reverse every other row
        index_array[1::2,:] = np.flip(index_array, axis=1)[1::2,:]
        if 'reverse' in orientation:
            # flip array horizontally
            index_array = np.flip(index_array, axis=1)
        # rotate array by 90 degrees to make vertical
        index_array = np.swapaxes(index_array, 0, 1)
    else:
        # horizontal serpentine
        index_array = np.reshape(indices, (height, width))
        # reverse every other row
        index_array[1::2,:] = np.flip(index_array, axis=1)[1::2,:]
        if 'reverse' in orientation:
            # flip array horizontally
            index_array = np.flip(index_array, axis=1)
    if debug == True:
        print "    2D index array:", np.shape(index_array)
    return index_array

def map_spectral_band(x_values, spectra, position_array, band_position, band_width, function='sum', vmin=None, vmax=None, clipping=False, clip=0.02, debug=False):
    # function for turning a list of spectra into a 2D array of band values that can be plotted as an image
    global Cmin, Cmax
    print "    generating intensity map for %0.f-%0.f..." % (band_position-band_width, band_position+band_width)
    print "        input:", np.shape(x_values), np.shape(spectra), np.shape(position_array)
    print "        band compression function:", function
    if clipping == True:
        # check if input clipping value makes sense and correct if necessary
        if clip >= 50:
            print "        input clipping value exceeds limit, defaulting to 2%%"
            clip = 0.02
        elif clip >= 0.5:
            print "        input clipping value greater than 0.5, converting to percentage"
            clip = float(clip)/100.
        elif clip < 1./len(spectra):
            print "        input clipping value equates to less than 1 spectrum, defaulting to 0"
            clip = 0.
    else:
        clip = 0.
    # determine which x_values fall within the target band
    band_pass = np.absolute(x_values - band_position) <= band_width
    if np.any(band_pass == True) == False:
        print "    spectral band outside of data range"
    # reindex spectra into a 2D map according to the position array
    spectral_map = spectra[position_array]
    # get the mean band value
    if debug == True:
        print "    spectral map:", np.shape(spectral_map)
        print "    masked spectral map:", np.shape(spectral_map[:, :, band_pass])
    if function == 'sum':
        band_map = np.sum(spectral_map[:, :, band_pass], axis=2)
    elif function == 'std':
        band_map = np.std(spectral_map[:, :, band_pass], axis=2)
    else:
        band_map = np.mean(spectral_map[:, :, band_pass], axis=2)
    if vmin == None or vmax == None:
        # determine min/max values for color mapping
        if clipping == True and clip > 0.:
            indx = int(np.ceil(clip*len(spectra)))
            if debug == True:
                print "        clip percentage: %0.1f%%" % (100.*indx/len(spectra))
                print "        min clipping indx:", indx, "of", len(spectra)
                print "        max clipping indx:", len(spectra)-indx, "of", len(spectra)
            channel_sort = np.sort(band_map, axis=None)
            if vmin == None:
                vmin = channel_sort[indx]
            if vmax == None:
                vmax = channel_sort[-indx-1]
        else:
            if vmin == None:
                vmin = np.amin(band_map)
            if vmax == None:
                vmax = np.amax(band_map)
    print "    output data range: %0.1f to %0.1f" % (np.amin(band_map), np.amax(band_map))
    print "    vmin, vmax range: %0.1f to %0.1f" % (vmin, vmax)
    return band_map, vmin, vmax

def map_spectral_RGB(x_values, spectra, position_array, band_positions, band_widths, function='sum', norm='full', clip=None, floor=0.02, ceiling=0.98):
    # function for turning a list of spectra into a 2D array of RGB channel values (0-1) that can be plotted as an image
    print "input:", np.shape(x_values), np.shape(spectra), np.shape(position_array), np.shape(band_positions), np.shape(band_widths)
    print "band compression function:", function
    if isinstance(band_widths, tuple) == True:
        if len(band_widths) == 3:
            band_widthB, band_widthG, band_widthR = band_widths
    elif isinstance(band_widths, int) == True or isinstance(band_widths, float) == True:
        band_widthB, band_widthG, band_widthR = (float(band_widths), float(band_widths), float(band_widths))
    elif np.amax(x_values) - np.amin(x_values) > 1000:
        print "input band width not recognised, assuming default Raman value of 50 cm-1"
        band_widthB, band_widthG, band_widthR = (50., 50., 50.)
    else:
        print "input band width not recognised, assuming default Fluor value of 5 nm"
        band_widthB, band_widthG, band_widthR = (5., 5., 5.)
    if norm == 'full':
        print "normalising all RGB channels together..."
    elif norm == 'channel':
        print "normalising RGB channels individually..."
    else:
        print "no normalisation, absolute values may not be suitable for plotting image..."
    if clip != None:
        floor = float(clip)
        ceiling = 1. - float(clip)
    # determine which x_values fall within the target band
    R_pass = np.absolute(x_values - band_positions[0]) <= band_widthR
    G_pass = np.absolute(x_values - band_positions[1]) <= band_widthG
    B_pass = np.absolute(x_values - band_positions[2]) <= band_widthB
    print np.shape(R_pass), np.shape(G_pass), np.shape(B_pass)
    if np.any(np.concatenate((R_pass, G_pass, B_pass)) == True) == False:
        print "spectral band outside of data range"
    # reindex spectra into a 2D map according to the position array
    spectral_map = spectra[position_array]
    print np.shape(spectral_map)
    # get the mean band values
    print np.shape(spectral_map[:, :, R_pass])
    if function == 'sum':
        R_map = np.sum(spectral_map[:, :, R_pass], axis=2)
        G_map = np.sum(spectral_map[:, :, G_pass], axis=2)
        B_map = np.sum(spectral_map[:, :, B_pass], axis=2)
    else:
        R_map = np.mean(spectral_map[:, :, R_pass], axis=2)
        G_map = np.mean(spectral_map[:, :, G_pass], axis=2)
        B_map = np.mean(spectral_map[:, :, B_pass], axis=2)
    print "R channel range: %0.1f to %0.1f" % (np.amin(R_map), np.amax(R_map))
    print "G channel range: %0.1f to %0.1f" % (np.amin(G_map), np.amax(G_map))
    print "B channel range: %0.1f to %0.1f" % (np.amin(B_map), np.amax(B_map))
    RGB_hists = np.stack((np.sort(R_map, axis=None), np.sort(G_map, axis=None), np.sort(B_map, axis=None)))
    print np.shape(RGB_hists)
    RGB_limits = np.zeros((3,2))
    if norm == 'full':
        # normalise all three channels to overall min/max
        floor_indx = int(np.ceil(floor*3.*np.size(R_map)))
        print "    floor: %0.1f%%" % (100.*floor_indx/(3.*np.size(R_map)))
        print "    floor indx:", floor_indx, "of", 3.*np.size(R_map)
        ceil_indx = int(np.floor(ceiling*3.*np.size(R_map)))
        print "    ceiling: %0.1f%%" % (100.*ceil_indx/(3.*np.size(R_map)))
        print "    ceiling indx:", ceil_indx, "of", 3.*np.size(R_map)
        channel_sort = np.sort(RGB_hists, axis=None)
        channel_min = channel_sort[floor_indx]
        channel_max = channel_sort[ceil_indx]
        print "min/max: %0.1f to %0.1f" % (channel_min, channel_max)
        R_map = (R_map - channel_min)/(channel_max - channel_min)
        G_map = (G_map - channel_min)/(channel_max - channel_min)
        B_map = (B_map - channel_min)/(channel_max - channel_min)
        RGB_limits = np.tile(np.asarray([channel_min, channel_max]), (3,1))
    elif norm == 'channel':
        # normalise and clip intensity values in each channel individually
        floor_indx = int(np.ceil(floor*np.size(R_map)))
        print "    floor: %0.1f%%" % (100.*floor_indx/(np.size(R_map)))
        print "    floor indx:", floor_indx, "of", np.size(R_map)
        ceil_indx = int(np.floor(ceiling*np.size(R_map)))
        print "    ceiling: %0.1f%%" % (100.*ceil_indx/(np.size(R_map)))
        print "    ceiling indx:", ceil_indx, "of", np.size(R_map)
        # handle R channel
        channel_min = RGB_hists[0][floor_indx]
        channel_max = RGB_hists[0][ceil_indx]
        print "R min/max: %0.1f to %0.1f" % (channel_min, channel_max)
        R_map = (R_map - channel_min)/(channel_max - channel_min)
        RGB_limits[0] = np.asarray([channel_min, channel_max])
        # handle G channel
        channel_min = RGB_hists[1][floor_indx]
        channel_max = RGB_hists[1][ceil_indx]
        print "G min/max: %0.1f to %0.1f" % (channel_min, channel_max)
        G_map = (G_map - channel_min)/(channel_max - channel_min)
        RGB_limits[1] = np.asarray([channel_min, channel_max])
        # handle B channel
        channel_min = RGB_hists[2][floor_indx]
        channel_max = RGB_hists[2][ceil_indx]
        print "B min/max: %0.1f to %0.1f" % (channel_min, channel_max)
        B_map = (B_map - channel_min)/(channel_max - channel_min)
        RGB_limits[2] = np.asarray([channel_min, channel_max])
    else:
        # data is not normalised or clipped
        RGB_limits[0] = np.asarray([np.amin(R_map), np.amax(R_map)])
        RGB_limits[1] = np.asarray([np.amin(G_map), np.amax(G_map)])
        RGB_limits[2] = np.asarray([np.amin(B_map), np.amax(B_map)])
    # create RGB map array
    RGB_map = np.stack((R_map, G_map, B_map), axis=-1)
    print np.shape(RGB_map)
    if norm in ['full', 'channel']:
        # clean up any values outside clipping range
        RGB_map[RGB_map > 1.] = 1.
        RGB_map[RGB_map < 0.] = 0.
    print "output data range: %0.1f to %0.1f" % (np.amin(RGB_map), np.amax(RGB_map))
    print "output:", np.shape(RGB_map), np.shape(RGB_hists), np.shape(RGB_limits)
    return RGB_map, RGB_hists, RGB_limits

def clip_this_map(absolute_map, floor=0.02, ceiling=0.98, norm='full', debug=False):
    # function for clipping and normalising the values of a 2D map (RGB or single-channel) for image generation
    print "    clipping intensity map to %0.2f-%0.2f" % (floor, ceiling)
    if absolute_map.ndim > 2:
        # handling RGB map
        print "        input map is RGB"
        R_map = absolute_map[:,:,0]
        G_map = absolute_map[:,:,1]
        B_map = absolute_map[:,:,2]
        RGB_hists = np.stack((np.sort(R_map, axis=None), np.sort(G_map, axis=None), np.sort(B_map, axis=None)))
        if debug == True:
            print "        R channel range: %0.1f to %0.1f" % (np.amin(R_map), np.amax(R_map))
            print "        G channel range: %0.1f to %0.1f" % (np.amin(G_map), np.amax(G_map))
            print "        B channel range: %0.1f to %0.1f" % (np.amin(B_map), np.amax(B_map))
            print "    sorted values:",np.shape(RGB_hists)
        RGB_limits = np.zeros((3,2))
        if norm == 'full':
            # normalise all three RGB channels to overall min/max
            floor_indx = int(np.ceil(floor*3.*np.size(absolute_map)))
            ceil_indx = int(np.floor(ceiling*3.*np.size(absolute_map)))
            channel_sort = np.sort(RGB_hists, axis=None)
            channel_min = channel_sort[floor_indx]
            channel_max = channel_sort[ceil_indx]
            if debug == True:
                print "        floor: %0.1f%%" % (100.*floor_indx/(3.*np.size(absolute_map)))
                print "        floor indx:", floor_indx, "of", 3.*np.size(absolute_map)
                print "        ceiling: %0.1f%%" % (100.*ceil_indx/(3.*np.size(absolute_map)))
                print "        ceiling indx:", ceil_indx, "of", 3.*np.size(absolute_map)
                print "    min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            R_map = (R_map - channel_min)/(channel_max - channel_min)
            G_map = (G_map - channel_min)/(channel_max - channel_min)
            B_map = (B_map - channel_min)/(channel_max - channel_min)
            RGB_limits = np.tile(np.asarray([channel_min, channel_max]), (3,1))
        else:
            # normalise and clip intensity values in each RGB channel individually
            floor_indx = int(np.ceil(floor*np.size(absolute_map)))
            ceil_indx = int(np.floor(ceiling*3.*np.size(absolute_map)))
            # handle R channel
            channel_sort = RGB_hists[0]
            channel_min = channel_sort[floor_indx]
            channel_max = channel_sort[ceil_indx]
            R_map = (R_map - channel_min)/(channel_max - channel_min)
            RGB_limits[0] = np.asarray([channel_min, channel_max])
            # handle G channel
            channel_sort = RGB_hists[1]
            channel_min = channel_sort[floor_indx]
            channel_max = channel_sort[ceil_indx]
            G_map = (G_map - channel_min)/(channel_max - channel_min)
            RGB_limits[1] = np.asarray([channel_min, channel_max])
            # handle B channel
            channel_sort = RGB_hists[2]
            channel_min = channel_sort[floor_indx]
            channel_max = channel_sort[ceil_indx]
            B_map = (B_map - channel_min)/(channel_max - channel_min)
            RGB_limits[2] = np.asarray([channel_min, channel_max])
            if debug == True:
                print "    R min/max: %0.1f to %0.1f" % (channel_min, channel_max)
                print "    G min/max: %0.1f to %0.1f" % (channel_min, channel_max)
                print "    B min/max: %0.1f to %0.1f" % (channel_min, channel_max)
        # create RGB map array
        RGB_map = np.stack((R_map, G_map, B_map), axis=-1)
        if full_norm == True or channel_norm == True:
            # clean up any values outside clipping range
            RGB_map[RGB_map > 1.] = 1.
            RGB_map[RGB_map < 0.] = 0.
        print "    output data range: %0.1f to %0.1f" % (np.amin(RGB_map), np.amax(RGB_map))
        print "    output:", np.shape(RGB_map), np.shape(RGB_hists), np.shape(RGB_limits)
        return RGB_map, RGB_hists, RGB_limits
    else:
        # handling single-channel map
        print "        input map is single-channel"
        hist = np.sort(absolute_map, axis=None)
        # determine upper, lower limits for clipping
        floor_indx = int(np.ceil(floor*np.size(absolute_map)))
        ceil_indx = int(np.floor(ceiling*np.size(absolute_map)))
        channel_min = hist[floor_indx]
        channel_max = hist[ceil_indx]
        limits = np.asarray([channel_min, channel_max])
        if debug == True:
            print "    sorted values:", np.shape(hist)
            print "        floor: %0.1f%%" % (100.*floor_indx/(3.*np.size(absolute_map)))
            print "        floor indx:", floor_indx, "of", 3.*np.size(absolute_map)
            print "        ceiling: %0.1f%%" % (100.*ceil_indx/(3.*np.size(absolute_map)))
            print "        ceiling indx:", ceil_indx, "of", 3.*np.size(absolute_map)
            print "    clipping min/max: %0.1f to %0.1f" % (channel_min, channel_max)
        # normalise map based on clipping limits
        clipped_map = (absolute_map - channel_min)/(channel_max - channel_min)
        # clean up any values outside clipping range
        clipped_map[clipped_map > 1.] = 1.
        clipped_map[clipped_map < 0.] = 0.
        print "    output data range: %0.1f to %0.1f" % (np.amin(clipped_map), np.amax(clipped_map))
        print "    output:", np.shape(clipped_map), np.shape(hist), np.shape(limits)
        return clipped_map, hist, limits
    
def get_clip_limits(absolute_map, clip=0.02, norm='full', debug=False):
    # function for determining the upper and lower clipping limits for a 2D spectral map (RGB or single-channel)
    print "    getting clip limits of map..."
    # check if input clipping value makes sense and correct if necessary
    if clip >= 50:
        print "    input clipping value exceeds limit, defaulting to 2%%"
        clip = 0.02
    elif clip >= 0.5:
        print "    input clipping value greater than 0.5, converting to fraction"
        clip = float(clip)/100.
    elif clip < 1./np.size(absolute_map):
        print "    input clipping value rounds to 0, defaulting to minimum"
        clip = 1./np.size(absolute_map)
    if absolute_map.ndim > 2:
        # handling RGB map
        print "        input map is RGB"
        RGB_limits = np.zeros((3,2))
        if norm == 'full':
            # normalise all three RGB channels to overall min/max
            indx = int(np.ceil(clip*3.*len(spectra)))
            channel_sort = np.sort(RGB_hists, axis=None)
            channel_min = channel_sort[indx]
            channel_max = channel_sort[-indx-1]
            if debug == True:
                print "        clip percentage: %0.1f%%" % (100.*indx/(3.*np.size(absolute_map)))
                print "        min clipping indx:", indx, "of", 3.*np.size(absolute_map)
                print "        max clipping indx:", len(spectra)-indx, "of", 3.*np.size(absolute_map)
                print "    min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            R_map = (R_map - channel_min)/(channel_max - channel_min)
            G_map = (G_map - channel_min)/(channel_max - channel_min)
            B_map = (B_map - channel_min)/(channel_max - channel_min)
            RGB_limits = np.tile(np.asarray([channel_min, channel_max]), (3,1))
        else:
            indx = int(np.ceil(clip*np.size(absolute_map)))
            # handle R channel
            channel_sort = RGB_hists[0]
            channel_min = channel_sort[indx]
            channel_max = channel_sort[-indx-1]
            RGB_limits[0] = np.asarray([channel_min, channel_max])
            # handle G channel
            channel_sort = RGB_hists[1]
            channel_min = channel_sort[indx]
            channel_max = channel_sort[-indx-1]
            RGB_limits[1] = np.asarray([channel_min, channel_max])
            # handle B channel
            channel_sort = RGB_hists[2]
            channel_min = channel_sort[indx]
            channel_max = channel_sort[-indx-1]
            if debug == True:
                print "        clip percentage: %0.1f%%" % (100.*indx/np.size(absolute_map))
                print "        min clipping indx:", indx, "of", np.size(absolute_map)
                print "        max clipping indx:", np.size(absolute_map)-indx, "of", np.size(absolute_map)
                print "    R min/max: %0.1f to %0.1f" % (channel_min, channel_max)
                print "    G min/max: %0.1f to %0.1f" % (channel_min, channel_max)
                print "    B min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            RGB_limits[2] = np.asarray([channel_min, channel_max])
        print "    output:", np.shape(RGB_limits)
        return RGB_limits
    else:
        # handling single-channel map
        print "        input map is single-channel"
        hist = np.sort(absolute_map, axis=None)
        # determine upper, lower limits for clipping
        indx = int(np.ceil(clip*len(hist)))
        channel_min = hist[indx]
        channel_max = hist[-indx-1]
        if debug == True:
            print "    sorted values:", np.shape(hist)
            print "        clip percentage: %0.1f%%" % (100.*indx/len(hist))
            print "        min clipping indx:", indx, "of", len(hist)
            print "       max clipping indx:", len(hist)-indx, "of", len(hist)
            print "    min/max values: %0.1f/%0.1f" % (channel_min, channel_max)
        limits = np.asarray([channel_min, channel_max])
        print "    output:", np.shape(limits)
        return limits

def area_mean(x_values, spectra, position_array, xy_coords, target, shape='circle', width=5, debug=False):
    # find the mean spectrum for an area around a targeted point expressed in x,y space
    print "    getting mean spetrum of %s centered at (%0.2f, %0.2f)..." % (shape, target[0], taret[1])
    print "        input:", np.shape(x_values), np.shape(spectra), np.shape(position_array)
    # check values
    width = float(width)
    # check coordinate arrays
    x_coords, y_coords = (xy_coords)
    # check target location (tranposed coords)
    y_loc, x_loc = target
    if x_loc < np.amin(x_coords):
        print "        target x coordinate outside map area!"
        x_loc = np.amin(x_coords)
    elif x_loc > np.amax(x_coords):
        print "        target x coordinate outside map area!"
        x_loc = np.amax(x_coords)
    if y_loc < np.amin(y_coords):
        print "        target y coordinate outside map area!"
        y_loc = np.amin(y_coords)
    elif y_loc > np.amax(y_coords):
        print "        target y coordinate outside map area!"
        y_loc = np.amax(y_coords)
    # mask the map to exclude points outside the targeted area
    if shape == 'square':
        mask = np.where((np.absolute(x_coords - x_loc) <= width/2) & (np.absolute(y_coords - y_loc) <= width/2))
        print "        ", np.shape(mask)
    else:
        mask = np.where(np.absolute(xy_coords[0] - target[1])**2 + np.absolute(xy_coords[1] - target[0])**2 <= width**2)
        print np.shape(mask)
    target_spectra = spectra[position_array][mask]
    print "    spectra within ROI:", np.size(target_spectra, axis=0)
    mean_spectrum = np.mean(target_spectra, axis=0)
    print "    output:", np.shape(mean_spectrum)
    return mean_spectrum

def convert_map_to_spot_image(band_map, position_array, img_coords, pass_array=[], img=np.empty((1200,1648)), pixel_scale=None, spot_diameter=0.1, spot_scale_factor=1., vmin=None, vmax=None, cmap='viridis', cmap_norm=False, debug=False):
    # function for converting a band map into an RGBA image with true-to-size spots
    global ACI_pixel_scale
    # check inputs
    if pixel_scale == None:
        pixel_scale = ACI_pixel_scale
    if np.size(pass_array) != np.size(img_coords[0]):
        # assume all points past muster
        pass_array = np.full_like(np.ravel(img_coords[0]), True)
    # reshape img_coords, pass_array to match band_map array
    map_x, map_y = img_coords[:,position_array]
    pass_array = pass_array[position_array]
    if debug == True:
        print "input map:", np.shape(band_map)
        print "    reshaped coords:", np.shape(map_x), np.shape(map_y)
        print map_x
    if img.ndim in [2,3]:
        # use input image as template for pixel coordinate grid
        if debug == True:
            print "input image:", np.shape(img)
        img_height = np.size(img, axis=0)
        img_width = np.size(img, axis=1)
    else:
        # default to 1200x1648 image
        if debug == True:
            print "no input image, assuming 1200x1648 pixels"
        img_height = 1200
        img_width = 1648
    # create empty RGBA map and x,y coordinate arrays with same dimensions as image
    img_yx_coords = np.mgrid[-img_height/2:img_height/2, -img_width/2:img_width/2] * pixel_scale
    extent = (np.amin(img_yx_coords[1]), np.amax(img_yx_coords[1]), np.amin(img_yx_coords[0]), np.amax(img_yx_coords[0]))
    output_map = np.zeros((img_height, img_width, 4))
    if debug == True:
        print "pixel coordinate arrays:", np.shape(img_yx_coords)
        print "x range: %0.2f - %0.2f" % (extent[0], extent[1])
        print "y range: %0.2f - %0.2f" % (extent[2], extent[3])
        print "output map:", np.shape(output_map)
    # convert input map to RGBA
    if band_map.ndim == 3:
        # input map is RGB, with 3 channels
        map_flat = np.reshape(band_map, (-1,3))
        print np.shape(map_flat), np.shape(np.ones_like(map_flat[:,0])[:,np.newaxis])
        map_flat = np.concatenate((map_flat, np.ones_like(map_flat[:,0])[:,np.newaxis]), axis=1)
        norm = None
    else:
        # input map is single-channel
        if vmin == None:
            vmin = np.amin(vmin)
        if vmax == None:
            vmax = np.amax(vmax)
        # convert to RGBA based on vmin, vmax, and Cmap
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colormap.set_array([])
        map_flat = colormap.to_rgba(np.ravel(band_map), alpha=1.)
    if debug == True:
        print "flattened input map:", np.shape(map_flat)
        print "flattened coords:", np.shape(np.ravel(map_x)), np.shape(np.ravel(map_y))
    # iterate over points in map and fill in any pixels that are within range
    for i, color, point_x, point_y, check in zip(range(np.size(map_flat, axis=0)), map_flat, np.ravel(map_x), np.ravel(map_y), np.ravel(pass_array)):
        if check == True:
            dists = np.sqrt((img_yx_coords[1] - point_x)**2 + (img_yx_coords[0] - point_y)**2)
            dists = np.flip(dists, axis=0)
            mask = dists <= float(spot_scale_factor) * float(spot_diameter)/2
            output_map[mask] = color
    return output_map, extent, norm

"""
==================================================
create data arrays
==================================================
"""

data = {
    'scan_name': [],
    'sol': [],
    'scan': [],
    'target': [],
    'target_group': [],
    'laser_pulses': [],
    'laser_current': [],
    'photodiode_av': [],
    'table_size': [],
    'orientation': [],
    'step_size': [],
    'grid_index': [],
    'xy_coords': [],
    'img_coords': [],
    'ccd_temp': [],
    'wavelength': [],
    'raman_shift': [],
    'y': [],
    'y_r': [],
    'y_av': [],
    'y_r_av': [],
    'y_std': [],
    'y_r_std': [],
    'figdir': [],
    'outdir': [],
    'img': []
}

print
print "importing SRLC scan info spreadsheet"

SRLC_scan_info = pd.read_csv("%sSRLC_scan_properties.csv" % Data_dir, index_col=0, header=0, dtype={'sol': str, 'scan': str, 'target': str, 'group': str, 'plot color': str, 'table size': int, 'laser pulses': int, 'laser current': int, 'photodiode average': float, 'CRR-safe': bool})

print SRLC_scan_info.describe()

"""
==================================================
import SHERLOC FM data
==================================================
"""

print
print "input target: %s"  % Target
print "input Sol: %s" % Sol
print "input scan: %s" % Scan_name

if Target != '*':
    folders = []
    print
    print "importing by target name"
    if Sol == '*':
        sols = np.unique(SRLC_scan_info['sol'][SRLC_scan_info['target'] == Target].values)
        print "target measured on sols %s" % ", ".join(sols)
    else:
        sols = [Sol]
    print "sol list:", sols
    folders = []
    for sol in sols:
        temp = []
        if Scan_name != '*':
            temp = glob.glob('%ssol_%s/%s/' % (Data_dir, sol, Scan_name))
        else:
            for scan in np.unique(SRLC_scan_info['scan'][np.logical_and(SRLC_scan_info['target'] == Target, SRLC_scan_info['sol'] == sol)].values):
                temp += glob.glob('%ssol_%s/%s/' % (Data_dir, sol, scan))
        folders += temp
    if 'algan' not in Scan_name.lower():
        # trim AlGaN
        folders = [folder for folder in folders if 'AlGaN' not in folder]
else:
    folders = glob.glob('%ssol_*%s/%s/' % (Data_dir, Sol, Scan_name))
folders = sorted(folders)

print
print "folders found:", len(folders)

for folder in folders:
    print "    ", folder

print
if Data_formatting in ['*', 'Loupe', 'Loupe_N', 'Loupe_C']:
    # Loupe file formatting, with an active-dark file
    for folder in folders:
        sol = folder.split("/")[-3][4:]
        scan = folder.split("/")[-2]
        print
        print "attempting to import data for scan %s on Sol-%s" % (scan, sol)
        index = "Sol-%s %s" % (sol, scan)
        check = index in SRLC_scan_info.index.values
        print "    spreadsheet index check:", np.any(check)
        success = False
        while True:
            try:
                # attempt to find active-dark spectra file
                preprocessing = ''
                if Data_formatting == '*':
                    if check == True:
                        if SRLC_scan_info['CRR-safe'].loc[index] == True:
                            # import NC data
                                preprocessing = 'C'
                        else:
                            # import N data
                            print "    scan recorded as unsuitable for CRR, using N instead"
                            preprocessing = 'N'
                    else:
                        # import NC data by default
                        preprocessing = 'C'
                elif Data_formatting == 'Loupe_C':
                    # import NC data
                    preprocessing = 'C'
                elif Data_formatting == 'Loupe_N':
                    # import N data
                    preprocessing = 'N'
                else:
                    # import raw data
                    preprocessing = ''
                if preprocessing == '':
                    print "    looking for unprocessed spec file..."
                else:
                    print "    looking for Loupe-%s spec file..." % preprocessing
                spec_dir = glob.glob('%s*_Loupe_working/darkSubSpectra%s.csv' % (folder, preprocessing))
                if len(spec_dir) > 0:
                    spec_dir = spec_dir[0]
                print "        spec dir:", spec_dir
                
                # look for scan in info spreadsheet
                print "    getting ref data from spreadsheet:"
                if check == True:
                    target = SRLC_scan_info['target'].loc[index]
                    print "        target:        %s" % target
                    group = SRLC_scan_info['group'].loc[index]
                    print "        target group:  %s" % group
                    color = SRLC_scan_info['plot color'].loc[index]
                    print "        plot color:    %s" % color
                    table_size = SRLC_scan_info['table size'].loc[index]
                    print "        table size:    %d" % table_size
                    orientation = SRLC_scan_info['table orientation'].loc[index]
                    print "        table orientation:    %s" % orientation
                    step_size = SRLC_scan_info['step size (mm)'].loc[index]
                    print "        step size:    %0.3f mm" % step_size
                    table_dims = SRLC_scan_info['table dimensions'].loc[index]
                    print "        table dimensions:    %s" % table_dims
                    laser_x = float(SRLC_scan_info['laser x (px)'].loc[index])
                    laser_y = float(SRLC_scan_info['laser y (px)'].loc[index])
                    print "        laser x,y (px):    %0.f, %0.f" % (laser_x, laser_y)
                    laser_x_mm = -(laser_x - 824) * ACI_pixel_scale
                    laser_y_mm = (laser_y - 600) * ACI_pixel_scale
                    print "                 in mm:    %0.3f, %0.3f" % (laser_x_mm, laser_y_mm)
                    height, width = [int(dim) for dim in table_dims.split("x")]
                    # calculate position index array (for reshaping 1D array into 2D maps) and xy coordinates (in mm)
                    grid_index = serpentine(height, width, orientation, debug=False)
                    xy_coords = step_size * np.roll(np.mgrid[0:height, 0:width], 1, axis=0)
                    print "            grid index array:", np.shape(grid_index)
                    print "            xy_coords array: ", np.shape(xy_coords)
                    laser_pulses = SRLC_scan_info['laser pulses'].loc[index]
                    print "        laser pulses:  %d ppp" % laser_pulses
                    laser_current = SRLC_scan_info['laser current (A)'].loc[index]
                    print "        laser current: %d A" % laser_current
                    photodiode_av = SRLC_scan_info['photodiode average (counts)'].loc[index]
                    print "        photodiode av: %0.1f counts" % photodiode_av
                    ccd_temp = SRLC_scan_info['CCD temperature (C)'].loc[index]
                    print "        CCD temperature : %0.1f deg C" % ccd_temp
                else:
                    print "    could not find scan info in spreadsheet! Import failed!"
                    break
                    
                if Import_ACI == True:
                    # attempt to get ACI image
                    img_dir = glob.glob('%s*_Loupe_working/img/*.PNG' % folder)
                    print "    ACI image files found:", len(img_dir)
                    if len(img_dir) > 0:
                        img = plt.imread(img_dir[0])
                        print "        image array:", np.shape(img)
                        img_height_mm = np.size(img, axis=0) * ACI_pixel_scale
                        img_width_mm = np.size(img, axis=1) * ACI_pixel_scale
                        print "            height: %0.2f mm" % img_height_mm
                        print "             width: %0.2f mm" % img_width_mm
                        laser_x_mm = -(laser_x - float(np.size(img, axis=1))/2) * ACI_pixel_scale
                        laser_y_mm = (laser_y - float(np.size(img, axis=0))/2) * ACI_pixel_scale
                
                # attempt to get spatial data
                spatial_dir = glob.glob('%s*_Loupe_working/spatial.csv' % folder)
                print "    spatial data files found:", len(spatial_dir)
                # attempt to import spatial_dir csv file
                if len(spatial_dir) > 0:
                    spatial_data = np.genfromtxt(spatial_dir[0], delimiter=',', skip_header=1)
                    print "        spatial data array:", np.shape(spatial_data)
                    x = -spatial_data[table_size+1:table_size*2+1,0]    # assume coordinates in mm
                    y = spatial_data[table_size+1:table_size*2+1,1]
                    # point coordinates in mm (relative to image), adjusted for laser center
                    img_coords = np.asarray([x - laser_x_mm, y - laser_y_mm])
                    print img_coords
                        
                # prepare wavelength values based on pre-defined calibration
                wavelength = sherloc_calibration(np.arange(0., 2148.), sherloc_calib_params)[50:-50]
                print "    assuming default wavelength calibration"
                print "        wavelength range:   %3.1f - %3.1f nm" % (wavelength[0], wavelength[-1])
                print "          in raman shift:   %4.1f - %4.1f cm-1" % (wavelength2shift(wavelength[0]), wavelength2shift(wavelength[-1]))
                
                # import spec file
                spec = np.genfromtxt(spec_dir, delimiter=',', skip_header=1)
                print "    spec array:", np.shape(spec)
                # split imported array into the separate CCD read regions
                reg1 = spec[0:table_size, 50:-50]
                reg2 = spec[table_size+1:table_size*2+1, 50:-50]
                reg3 = spec[table_size*2+2:, 50:-50]
                print "        region 1:     ", np.shape(reg1), "nan check:", np.any(np.isnan(reg1))
                print "        region 2:     ", np.shape(reg2), "nan check:", np.any(np.isnan(reg2))
                print "        region 3:     ", np.shape(reg3), "nan check:", np.any(np.isnan(reg3))
                
                print "    combining regions"
                reg123 = reg1 + reg2 + reg3
                print "        full spectrum:", np.shape(reg123), "nan check:", np.any(np.isnan(reg123))
                
                success = True
                break
            except Exception as e:
                print "something went wrong! Exception:", e
                break
        if success == True:
            print "adding data to arrays"
            # add target/scan info
            data['scan_name'].append("Sol-%s %s (%s)" % (sol, scan, target))
            data['sol'].append(sol)
            data['scan'].append(scan)
            data['target'].append(target)
            data['target_group'].append(group)
            data['laser_pulses'].append(laser_pulses)
            data['laser_current'].append(laser_current)
            data['photodiode_av'].append(photodiode_av)
            data['ccd_temp'].append(ccd_temp)
            
            # add position data
            data['table_size'].append(table_size)
            data['orientation'].append(orientation)
            data['step_size'].append(step_size)
            data['grid_index'].append(grid_index)
            data['xy_coords'].append(xy_coords)
            data['img_coords'].append(img_coords)
            
            # add x values
            data['wavelength'].append(wavelength)
            data['raman_shift'].append(wavelength2shift(wavelength))
            
            # add spectra
            data['y'].append(reg123)
            data['y_r'].append(reg1)
            data['y_av'].append(np.mean(reg123, axis=0))
            data['y_r_av'].append(np.mean(reg1, axis=0))
            data['y_std'].append(np.std(reg123, axis=0))
            data['y_r_std'].append(np.std(reg1, axis=0))
            
            if Import_ACI == True:
                # add image
                data['img'].append(img)
            
            # create output/figure directories if required
            figdir = './figures by target/%s/' % (target)
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            figdir = '%sSol-%s/%s/' % (Figure_dir, sol, scan)
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            data['figdir'].append(figdir)
            outdir = '%sSol-%s/%s/' % (Output_dir, sol, scan)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            data['outdir'].append(outdir)
            
            # plot CCD region figure
            plt.figure(figsize=(8,4))
            plt.title("Spectral Region Averages:\nSol-%s %s" % (sol, scan))
            plt.plot(wavelength, np.mean(reg1, axis=0), 'b', label='region 1')
            plt.plot(wavelength, np.mean(reg2, axis=0), 'g', label='region 2')
            plt.plot(wavelength, np.mean(reg3, axis=0), 'r', label='region 3')
            ### plt.plot(wavelength, np.mean(reg123, axis=0), 'k', label='recombined')
            plt.figtext(0.75, 0.8, "%s point scan" % table_size, ha='right')
            plt.figtext(0.75, 0.75, "%s pulses per point" % laser_pulses, ha='right')
            plt.legend(loc=0)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity (counts)")
            plt.minorticks_on()
            plt.tight_layout()
            plt.savefig('%sSol-%s_%s_spectral-regions.png' % (figdir, sol, scan), dpi=300)
            plt.show()
            
            # plot x,y coordinates against ACI image
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            band_map, vmin, vmax = map_spectral_band(wavelength, reg123, grid_index, 310, 45, clipping=True, clip=0.04)
            ax1.set_title("Fluorescence Intensity")
            im = ax1.imshow(band_map, aspect='equal', vmin=vmin, vmax=vmax, cmap=Cmap)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05) 
            fig.colorbar(im, cax=cax)
            ax2.set_title("ACI image")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            im = ax2.imshow(img, extent=(-img_width_mm/2, img_width_mm/2, -img_height_mm/2, img_height_mm/2), cmap='gray')
            map_overlay, extent, norm = convert_map_to_spot_image(band_map, grid_index, img_coords, img=img, vmin=vmin, vmax=vmax, cmap=Cmap)
            ax2.imshow(map_overlay, extent=extent)
            fig.tight_layout(rect=[0, 0., 1, 0.93])
            fig.savefig('%sSol-%s_%s_scan-area.png' % (figdir, sol, scan), dpi=300)
            fig.show()
            
        else:
            print "could not find Loupe data files, skipping import"
            
"""
==================================================
begin FM data processing
==================================================
"""

# ==================================================
# normalise spectrum

print
print "normalising SHERLOC spectra..."

data['y_av_norm'] = []
data['y_std_norm'] = []
data['y_r_av_norm'] = []
data['y_r_std_norm'] = []

for i in range(0, len(data['scan'])):
    print
    print "%s: Sol-%s %s" % (i, data['sol'][i], data['scan'][i])
    y_max = find_max(data['wavelength'][i], data['y_av'][i], 255, 355)
    print "    average maximum found at %0.f nm: %0.f counts" % (y_max[0], y_max[1])
    data['y_av_norm'].append(data['y_av'][i]/y_max[1])
    data['y_std_norm'].append(data['y_std'][i]/y_max[1])
    y_max = find_max(data['raman_shift'][i], data['y_r_av'][i], 800, 4000)
    print "    R region average maximum found at %0.f cm-1: %0.f counts" % (y_max[0], y_max[1])
    data['y_r_av_norm'].append(data['y_r_av'][i]/y_max[1])
    data['y_r_std_norm'].append(data['y_r_std'][i]/y_max[1])
    
"""
# ==================================================
# create fluorescence maps
# ==================================================
"""

x_start, x_end = (250, 355)

if Plot_Fmaps == True:
    for i in range(0, len(data['scan'])):
        print
        print "%s: %s" % (i, data['scan_name'][i])
        x_coords, y_coords = (data['xy_coords'][i])

        # generate a fluorescence intensity map for entire range
        band_map, vmin, vmax = map_spectral_band(data['wavelength'][i], data['y'][i], data['grid_index'][i], 310, 45, clipping=True, clip=0.04)
        plt.imsave("%sSol-%s_%s_Fmap_%0.f-%0.fnm.png" % (data['figdir'][i], data['sol'][i], data['scan'][i], 265, 355), np.repeat(np.repeat(band_map, 5, axis=0), 5, axis=1), vmin=vmin, vmax=vmax)
        
        # generate a spectrum figure showing the RGB band positions
        plt.figure(figsize=(8,4))
        plt.suptitle("Fluorescence RGB Band Positions:\n%s" % (data['scan_name'][i]))
        # ax1: fluorescence spectra
        ax1 = plt.subplot(111)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Average Intensity")
        ax1.set_xlim(250, 355)
        # add wavenumber ticks to top of ax1
        ax1_top = ax1.twiny()
        x_range = ax1.get_xlim()
        major_labels, major_locations, minor_locations = get_shift_ticks(x_range)
        ax1_top.set_xticks(major_locations)
        ax1_top.set_xticklabels(major_labels)
        ax1_top.xaxis.set_minor_locator(plt.FixedLocator(minor_locations))
        ax1_top.set_xlabel("Raman Shift (cm$^{-1}$)")
        print major_labels
        print major_locations
        for color, center, width in zip(['r', 'g', 'b'], F_RGB_bands['center'], F_RGB_bands['width']):
            ax1.axvspan(center-width, center+width, color=color, linewidth=0., alpha=0.1)
        if Plot_std == True:
            ax1.fill_between(data['wavelength'][i], np.mean(data['y'][i], axis=0)-np.std(data['y'][i], axis=0), np.mean(data['y'][i], axis=0)+np.std(data['y'][i], axis=0), color='k', alpha=0.1, linewidth=0.)
        ax1.plot(data['wavelength'][i], np.mean(data['y'][i], axis=0), 'k')
        y_max = find_max(data['wavelength'][i], np.mean(data['y'][i], axis=0), 250, 355)[1]
        ax1.set_ylim(-0.2*y_max, 1.5*y_max)
        ax1.minorticks_on()
        plt.tight_layout(rect=[0, 0., 1, 0.93])
        plt.savefig('%sSol-%s_%s_bands.png' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
        plt.savefig('%sSol-%s_%s_bands.svg' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
        plt.show()
        
        
        # generate a map for each Fluor band and show strongest spectra
        band_maps = []
        colors = []
        hist_colors = ['r', 'g', 'b', 'k', 'c', 'tab:orange', 'tab:purple', 'tab:gray', 'tab:brown', 'tab:pink']
        
        # select bands for plotting
        if len(Fluor_bands['center']) > 0 and len(Fluor_bands['center']) == len(Fluor_bands['width']):
            # use specified band list
            sort = np.argsort(Fluor_bands['center'])
            band_centers = np.asarray(Fluor_bands['center'])[sort]
            band_widths = np.asarray(Fluor_bands['width'])[sort]
        else:
            # use default list
            if data['target'][i] in Default_F_bands.keys():
                sort = np.argsort(Default_F_bands[data['target'][i]]['center'])
                band_centers = np.asarray(Default_F_bands[data['target'][i]]['center'])[sort]
                band_widths = np.asarray(Default_F_bands[data['target'][i]]['width'])[sort]
            else:
                sort = np.argsort(Default_F_bands[data['Default']]['center'])
                band_centers = np.asarray(Default_F_bands['Default']['center'])[sort]
                band_widths = np.asarray(Default_F_bands['Default']['width'])[sort]
        print "bands for plotting:", band_centers
        
        aspect = float(np.size(data['grid_index'][i], axis=0)) / float(np.size(data['grid_index'][i], axis=1))
        print "map x/y aspect ratio:", (1./aspect)
        plt.figure(figsize=(4.*len(band_centers), 4.*aspect+2.))
        plt.suptitle("Fluorescence Band Intensities\n%s" % data['scan_name'][i])
        colorbar_factor = 8
        
        clipping = False
        clip = 0.04
        vmins = []
        vmaxs = []
        if Band_normalisation == 'fixed' and Cmin != None and Cmax != None:
            print "using pre-assigned upper/lower intensity limits for map normalisation"
            norm = 'fixed'
            vmins = np.full((len(band_centers)), Cmin)
            vmaxs = np.full((len(band_centers)), Cmax)
        elif Band_normalisation in ['all', 'all_bands', 'full', 'norm']:
            print "normalising maps using %0.2f and %0.2f percentiles of all band values" % (clip, 1.-clip)
            norm = 'fixed'
            band_maps = []
            for center, width in zip(band_centers, band_widths):
                temp, vmin, vmax = map_spectral_band(data['wavelength'][i], data['y'][i], data['grid_index'][i], center, width)
                band_maps.append(temp)
            temp = np.sort(np.ravel(band_maps))
            print np.shape(temp)
            indx = int(np.ceil(clip*len(temp)))
            print "    clip percentage: %0.1f%%" % (100.*indx/len(temp))
            print "    min clipping indx:", indx, "of", len(temp), "value:", temp[indx]
            print "    max clipping indx:", len(temp)-indx, "of", len(temp), "value:", temp[-indx]
            vmins = np.full((len(band_centers)), temp[indx])
            vmaxs = np.full((len(band_centers)), temp[-indx])
        else:
            print "defaulting to individual band normalisation, using %0.2f - %0.2f clipping" % (clip, 1.-clip)
            norm = 'full'
            vmins = np.full((len(band_centers)), None)
            vmaxs = np.full((len(band_centers)), None)
            clipping=True
        print
        print "band normalisation:", norm
        print "vmins:, vmaxes:", vmins, vmaxs
        print "clipping:", clipping
        print "clip percentage:", clip
        print
        
        # adjust size of histogram bins based on data volume
        if np.size(np.ravel(band_maps[0]))>1000:
            bins = 20
        else:
            bins = 10
            
        subplot_width = colorbar_factor*len(band_centers)+1
        i2 = 0
        for band_map, center, width, color, vmin, vmax in zip(band_maps, band_centers, band_widths, hist_colors, vmins, vmaxs):
            print "    band %0.f, from %0.f to %0.f" % (center, center-width, center+width)
            if i2 == 0:
                ax1 = plt.subplot2grid((3, subplot_width), (0, colorbar_factor*i2), colspan=colorbar_factor)
                ax1.set_title("%0.f-%0.f nm" % (center-width, center+width))
                ax1.hist(np.ravel(band_map), bins=bins, range=(vmin, vmax), color=color)
            else:
                plt.subplot2grid((3, subplot_width), (0, colorbar_factor*i2), colspan=colorbar_factor, sharey=ax1)
                plt.title("%0.f-%0.f nm" % (center-width, center+width))
                plt.hist(np.ravel(band_map), bins=bins, range=(vmin, vmax), color=color)
                plt.yticks([])
            plt.subplot2grid((3, subplot_width), (1, colorbar_factor*i2), colspan=colorbar_factor, rowspan=2)
            plt.imshow(band_map, aspect='equal', vmin=vmin, vmax=vmax, cmap=Cmap)
            if i2 != 0:
                plt.yticks([])
            i2 += 1
        ax2 = plt.subplot2grid((3, subplot_width), (1, subplot_width-1), rowspan=2)
        plt.colorbar(cax=ax2, shrink=0.9, label="intensity (counts)")
        plt.savefig("%sSol-%s_%s_fluor_maps.png" % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
        plt.show()
        
        # plot brightest spectra for each band
        brightest_indices = []
        for band_map, center, width, color, vmin, vmax in zip(band_maps, band_centers, band_widths, hist_colors, vmins, vmaxs):
            plt.figure(figsize=(10,4))
            plt.suptitle("%s\n%0.f-%0.f nm Intensity Map" % (data['scan_name'][i], center-width, center+width))
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
            ax2.set_xlim(x_start, x_end)
            vmin = 0.
            if Plot_on_ACI == True:
                map_overlay, extent, norm = convert_map_to_spot_image(band_map, data['grid_index'][i], data['img_coords'][i], vmin=vmin, vmax=vmax, cmap=Cmap)
                plt.imsave('%sSol-%s_%s_Fmap_%0.fnm_img.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], center), map_overlay)
            im = ax1.imshow(band_map, aspect='equal', vmin=vmin, vmax=vmax)
            plt.imsave('%sSol-%s_%s_Fmap_%0.fnm.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], center), np.repeat(np.repeat(band_map, 5, axis=0), 5, axis=1), vmin=vmin, vmax=vmax, cmap=Cmap)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            y_max = find_max(data['wavelength'][i], np.mean(data['y'][i], axis=0), x_start, x_end)[1]
            ax2.axvspan(center-width, center+width, color='k', alpha=0.1)
            
            width_px = len(np.ravel(np.where((wavelength >= center-width) & (wavelength <= center+width))))

            # find hottest points and add to list for plotting
            print
            print "    hottest point:",
            point = np.argmax(np.ravel(band_map))
            serp_indx = np.unravel_index(point, np.shape(band_map))
            ### print "        serpentine index:", serp_indx
            map_indx = data['grid_index'][i][serp_indx]
            print "    point", map_indx
            ### print "        map index:", map_indx
            print "        integrated intensity: %0.f counts" % (band_map.flatten()[point])
            print "           average intensity: %0.f counts" % (band_map.flatten()[point] / width_px)
            brightest_indices.append(map_indx)
            # find 3 hottest points
            hottest_points = np.flip(np.argsort(np.ravel(band_map))[-6:])
            print hottest_points
            y_max = 0.
            for point in hottest_points:
                serp_indx = np.unravel_index(point, np.shape(band_map))
                ### print "        serpentine index:", serp_indx
                map_indx = data['grid_index'][i][serp_indx]
                print "    point", map_indx
                ### print "        map index:", map_indx
                map_indx_2d = np.unravel_index(map_indx, np.shape(band_map))
                ### print "       2D:", map_indx_2d
                point_x = x_coords[map_indx_2d]
                point_y = y_coords[map_indx_2d]
                ### print "        x,y coords:", point_x, point_y
                print "        integrated intensity: %0.f counts" % (np.ravel(band_map)[point])
                print "           average intensity: %0.f counts" % (np.ravel(band_map)[point] / width_px)
                ax1.text(serp_indx[1], serp_indx[0], "%0.f" % (map_indx), color='w')
                ax2.plot(data['wavelength'][i], data['y'][i][map_indx], label='point %s' % (map_indx))
                ### if find_max(data['wavelength'][i], data['y'][i][map_indx], x_start, x_end)[1] > y_max:
                    ### y_max = find_max(data['wavelength'][i], data['y'][i][map_indx], x_start, x_end)[1]
                if np.sort(data['y'][i][map_indx])[-5] > y_max:
                    y_max = np.sort(data['y'][i][map_indx])[-5]
            ax2.plot(data['wavelength'][i], np.mean(data['y'][i], axis=0), 'k', label='mean')
            ax2.legend()
            ax2.set_ylim(-0.2*y_max, 1.2*y_max)
            ax2.minorticks_on()
            plt.tight_layout(rect=[0, 0., 1, 0.9])
            plt.savefig('%sSol-%s_%s_Fmap_%0.fnm_spec.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], center), dpi=300)
            plt.show()
            
        # produce figure showing best spectra for each band
        x_range = (250., 355.)
        plt.figure(figsize=(8,4))
        plt.suptitle("%s\nBrightest Fluorescence Spectra" % (data['scan_name'][i]))
        ax1 = plt.subplot(111)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Intensity (counts)")
        ax1.set_xlim(x_range)
        # add wavenumber ticks to top of ax1
        ax1_top = ax1.twiny()
        x_range = ax1.get_xlim()
        major_labels, major_locations, minor_locations = get_shift_ticks(x_range)
        ax1_top.set_xticks(major_locations)
        ax1_top.set_xticklabels(major_labels)
        ax1_top.xaxis.set_minor_locator(plt.FixedLocator(minor_locations))
        ax1_top.set_xlabel("Raman Shift (cm$^{-1}$)")
        F_max = find_max(data['wavelength'][i], data['y_av'][i], x_range[0], x_range[1])[1]
        for band_map, center, width, color, vmin, vmax, indx in zip(band_maps, band_centers, band_widths, hist_colors, vmins, vmaxs, brightest_indices):
            ax1.plot(data['wavelength'][i], data['y'][i][indx], label="%0.f nm: point %0.f" % (center, indx))
            y_max = np.mean(data['y'][i][indx][np.where((center-width <= data['wavelength'][i]) & (data['wavelength'][i] <= center+width))])
            noise = np.std(data['y'][i][indx][np.where((265-width <= data['wavelength'][i]) & (data['wavelength'][i] <= 265+width))])
            print
            print "Brightest %0.f +/- %0.f nm: point %s - %0.1f counts (noise: %0.1f)" % (center, width, indx, y_max, noise)
            y_max = find_max(data['wavelength'][i], data['y'][i][indx], x_range[0], x_range[1])[1]
            if y_max > F_max:
                F_max = y_max
        ax1.plot(data['wavelength'][i], data['y_av'][i], 'k', label="Average")
        ax1.legend()
        ax1.set_ylim(-0.2*F_max, 1.2*F_max)
        ax1.minorticks_on()
        plt.tight_layout(rect=[0, 0., 1, 0.93])
        plt.savefig('%sSol-%s_%s_best-F.png' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
        plt.show()
        
        # define RGB channels
        if len(F_RGB_bands['center']) == 3 and len(F_RGB_bands['width']) == 3:
            sort = np.argsort(F_RGB_bands['center'])[::-1]
            F_centers = np.asarray(F_RGB_bands['center'])[sort]
            F_widths = np.asarray(F_RGB_bands['width'])[sort]
        else:
            F_centers = np.asarray([275, 305, 340])
            F_widths = np.asarray([5, 5, 5])
        
        # plot RGB map of fluorescence, full normalisation
        RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(data['wavelength'][i], data['y'][i], data['grid_index'][i], F_centers, F_widths, norm='full', clip=0.02)
        print np.shape(RGB_map)
        plt.imsave('%sSol-%s_%s_Fmap_R=%0.f_G=%0.f_B=%0.f.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], F_centers[0], F_centers[1], F_centers[2]), np.repeat(np.repeat(RGB_map, 5, axis=0), 5, axis=1))
        if Plot_on_ACI == True:
            # save map as a grid of spots in ACI pixel space
            map_overlay, extent, norm = convert_map_to_spot_image(RGB_map, data['grid_index'][i], data['img_coords'][i], vmin=vmin, vmax=vmax, cmap=Cmap)
            plt.imsave('%sSol-%s_%s_Fmap_R=%0.f_G=%0.f_B=%0.f_img.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], F_centers[0], F_centers[1], F_centers[2]), map_overlay)
            
        # plot RGB map of fluorescence, channel normalisation
        RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(data['wavelength'][i], data['y'][i], data['grid_index'][i], F_centers, F_widths, norm='channel', floor=0.5, ceiling=0.98)
        print np.shape(RGB_map)
        plt.imsave('%sSol-%s_%s_Fmap_R=%0.f_G=%0.f_B=%0.f_norm.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], F_centers[0], F_centers[1], F_centers[2]), np.repeat(np.repeat(RGB_map, 5, axis=0), 5, axis=1))
        if Plot_on_ACI == True:
            # save map as a grid of spots in ACI pixel space
            map_overlay, extent, norm = convert_map_to_spot_image(RGB_map, data['grid_index'][i], data['img_coords'][i], vmin=vmin, vmax=vmax, cmap=Cmap)
            plt.imsave('%sSol-%s_%s_Fmap_R=%0.f_G=%0.f_B=%0.f_norm_img.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], F_centers[0], F_centers[1], F_centers[2]), map_overlay)
            
        # plot 340/300 nm fluorescence ratio
        F_centers = np.asarray([305, 340])
        F_widths = np.asarray([5, 5])
        band_maps = []
        for center, width in zip(F_centers, F_widths):
            temp, vmin, vmax = map_spectral_band(data['wavelength'][i], data['y'][i], data['grid_index'][i], center, width)
            band_maps.append(temp)
        ratio_map = band_maps[0] / band_maps[1]
        limits = get_clip_limits(ratio_map, 0.02)
        vmin = np.amax([0, limits[0]])
        vmax = limits[1]
        plt.imshow(ratio_map, aspect='equal', vmin=vmin, vmax=vmax, cmap='plasma')
        plt.imsave('%sSol-%s_%s_F-ratio-map_%0.f-v-%0.f.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], F_centers[0], F_centers[1]), np.repeat(np.repeat(ratio_map, 5, axis=0), 5, axis=1), vmin=vmin, vmax=vmax, cmap='plasma')
        if Plot_on_ACI == True:
            # save map as a grid of spots in ACI pixel space
            map_overlay, extent, norm = convert_map_to_spot_image(ratio_map, data['grid_index'][i], data['img_coords'][i], vmin=vmin, vmax=vmax, cmap='plasma')
            plt.imsave('%sSol-%s_%s_F-ratio-map_%0.f-v-%0.f_img.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], F_centers[0], F_centers[1]), map_overlay)
        plt.colorbar(label='300/340 Ratio')
        plt.show()
        
"""
# ==================================================
# baseline subtraction
# ==================================================
"""

print
print "baselining Raman spectrum in region 1"

data['y_r_av_sub'] = []
data['y_r_av_sub_norm'] = []
data['y_r_std_sub'] = []
data['y_r_std_sub_norm'] = []

data['y_r_sub'] = []
data['y_r_sub_norm'] = []

for i in range(0, len(data['scan'])):
    print
    print "%s: Sol-%s %s" % (i, data['sol'][i], data['scan'][i])
    sol = data['sol'][i]
    scan = data['scan'][i].lower()
    
    y_temp = data['y_r_av'][i]
    text = 'Average'
    
    # default settings
    base = 'poly'
    order = 11
    window = 25
    
    # select list of x positions to fit with baseline
    if 'HOPG' in scan:
        x_list = [300,400,900,1000,1800,1900,2000,2100,2500,3500,4000,4200,4500]
    elif 'Calcite' in scan:
        x_list = [300,400,600,800,1300,1900,2000,2500,2700,3000,3100,3300,3500,3800,4000,4100]
        order = 11
    elif 'teflon' in scan or 'ngimat' in scan or 'vectran' in scan or 'fabric' in scan or 'geocache' in scan or 'polycarbonate' in scan:
        x_list = [300,400,800,900,1000,1100,1800,2000,2300,2500,2700,2900,3300,3500,3800,4000,4100]
    elif 'diffusil' in scan:
        x_list = [300,400,600,700,950,1300,1400,1500,1600,1800,2000,2300,2500,2700,3800,4000,4100]
        order = 11
        x_list = [300,400,800,900,1000,1100,1800,2000,2300,2500,2700,2900,3200,3500,3800,4000,4100]
    elif 'algan' in scan:
        x_list = [300,400,950,1300,1400,1500,1600,1800,2000,2300,2500,2700,3800,4000,4100]
        order = 11
    elif 'maze' in scan:
        x_list = [300,400,950,1300,1400,1500,1800,2000,2300,2500,2700,3000,3500,3800,4000,4100]
        order = 11
    elif 'sau008' in scan or 'meteorite' in scan:
        x_list = [300,400,600,900,1250,1800,2000,2300,2500,2700,3000,3500,3800,4000,4100]
        order = 11
    elif sol == '0141':
        x_list = [300,400,550,600,900,1200,1300,1400,1600,1800,2000,2300,2500,2700,3000,3800,4000,4100]
        order = 11
    elif sol in ['0207', '0208', '0847', '0851', '0852']:
        x_list = [300,400,450,575,730,920,1250,1300,1400,1600,1800,2000,2300,2500,2700,3000,3400,3800,4000,4100]
        order = 11
        window = 1
    elif sol in ['0257', '0269']:
        x_list = [300,400,550,600,900,1200,1300,1400,1600,1800,2000,2100,2200,2300,2400,2800,3000,3800,4000,4100]
        order = 11
    elif sol == '0463':
        x_list = [300,400,550,600,900,1200,1300,1400,1500,1700,1800,1900,2000,2100,2300,2400,2700,3000,3400,3800,4000,4100]
        order = 11
    elif sol in ['0489']:
        x_list = [300,400,550,900,950,1200,1300,1450,1650,1700,1800,2000,2100,2300,2400,2600,2800,2900,3000,3100,3300,3500,3700,3900,4000,4100]
        order = 15
        if scan == 'survey1':
            x_list.remove(1650)
            x_list.remove(2800)
    elif sol in ['0614', '0617', '0618', '0620', '0697']:
        x_list = [300,400,580,900,1200,1300,1400,1450,1500,1650,1700,1800,2000,2100,2300,2400,2600,2800,2900,3000,3800,3900,4000,4100]
        order = 15
    elif sol in ['0789']:
        x_list = [300,400,580,875,925,1200,1300,1400,1450,1500,1650,1700,1800,2000,2100,2300,2400,2600,2800,2900,3000,3800,3900,4000,4100]
        order = 15
    elif sol in ['0938']:
        x_list = [300,400,450,575,875,890,1250,1300,1500,1700,1800,2000,2100,2300,2500,2700,3000,3400,3800,4000,4100]
    elif Hydration == True:
        x_list = [300,400,550,900,1200,1300,1400,1500,1700,1800,2000,2100,2300,2400,2800,2900,3000,3800,4000,4100]
        order = 11
    else:
        # optimised for amorphous silicate spectra on Sol 141
        x_list = [300,400,450,575,750,875,890,1250,1300,1500,1700,1800,2000,2100,2300,2500,2700,3000,3400,3800,4000,4100]
    print "    x positions for baselining:", x_list
    
    # create empty arrays for baselined point spectra
    data['y_r_sub'].append(np.zeros_like(data['y_r'][i]))
    data['y_r_sub_norm'].append(np.zeros_like(data['y_r'][i]))
    
    # do baseline subtraction on each point spectrum
    failures = []
    print "baseling %s point spectra for %s..." % (data['table_size'][i], data['scan_name'][i])
    for i2 in range(0, data['table_size'][i]):
        # attempt subtraction
        while True:
            try:
                basefit = baseline(data['raman_shift'][i], data['y_r'][i][i2], i, x_list, base=base, max_order=order, hydration=Hydration, plot=False, debug=False)
                data['y_r_sub'][i][i2] = basefit
                break
            except Exception as e:
                print "    spectrum %s: something went wrong! exception:" % i2, e
                data['y_r_sub'][i][i2] = data['y_r'][i][i2]
                failures.append(i2)
                break
        # normalise baselined point spectrum
        y_max = find_max(data['raman_shift'][i], data['y_r_sub'][i][i2], 800, 4000)[1]
        data['y_r_sub_norm'][i][i2] = data['y_r_sub'][i][i2]/y_max
    print "    baseline failures: %s of %s" % (len(failures), data['table_size'][i])
    print "         failed points:", failures
    
    print "baselining %s average spectrum..." % data['scan_name'][i]
    while True:
        try:
            basefit = baseline(data['raman_shift'][i], data['y_r_av'][i], i, x_list, base=base, max_order=order, hydration=Hydration, plot=True, debug=False)
            print "    successful fit"
            data['y_r_av_sub'].append(basefit)
            data['y_r_std_sub'].append(data['y_r_std'][i])
            break
        except Exception as e:
            print "    something went wrong! exception:", e
            data['y_r_av_sub'].append(y_temp)
            data['y_r_std_sub'].append(data['y_r_std'][i])
            break
    # normalise baselined average spectrum
    y_max = find_max(data['raman_shift'][i], data['y_r_av_sub'][i], 800, 4000)[1]
    data['y_r_av_sub_norm'].append(data['y_r_av_sub'][i]/y_max)
    
"""
# ==================================================
# do noise level analysis
# ==================================================
"""

data['F_noise'] = []
data['R_noise'] = []
for i in range(len(data['scan'])):
    print data['sol'][i], data['scan'][i]
    plt.figure(figsize=(12,4))
    plt.suptitle("Sol-%s %s" % (data['sol'][i], data['scan'][i]))
    
    # fluorescence signal/noise
    ax1 = plt.subplot(221)  # F signal
    ax2 = plt.subplot(223)  # F noise
    band_map, vmin, vmax = map_spectral_band(data['wavelength'][i], data['y'][i], data['grid_index'][i], 310, 45, function='mean', clipping=True, clip=0.04)
    img = ax1.imshow(band_map)
    ax1.set_title("F Signal")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    print np.shape(img)
    plt.colorbar(img, cax=cax)
    noise_map, vmin, vmax = map_spectral_band(data['wavelength'][i], data['y'][i], data['grid_index'][i], 265, 5, function='std', clipping=True, clip=0.04)
    img = ax2.imshow(noise_map)
    ax2.set_title("F Noise")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    data['F_noise'].append(np.ravel(noise_map))
    
    # raman
    ax3 = plt.subplot(222)
    ax4 = plt.subplot(224)
    band_map, vmin, vmax = map_spectral_band(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], 1300, 500, function='mean', clipping=True, clip=0.04)
    img = ax3.imshow(band_map)
    ax3.set_title("R Signal")
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    noise_map, vmin, vmax = map_spectral_band(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], 2050, 50, function='std', clipping=True, clip=0.04)
    img = ax4.imshow(noise_map)
    ax4.set_title("R Noise")
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    data['R_noise'].append(np.ravel(noise_map))
    
    plt.tight_layout()
    plt.show()
    
data['F_noise'] = np.asarray(data['F_noise'])
data['R_noise'] = np.asarray(data['R_noise'])

norm = mpl.colors.LogNorm(vmin=np.amin(data['laser_pulses']), vmax=np.amax(data['laser_pulses']))
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=Cmap)
cmap.set_array([])

print
plt.figure(figsize=(8,4))
plt.title("Fluorescence Noise vs Temperature")
for pulses in np.unique(data['laser_pulses']):
    temp = []
    F_noise = []
    R_noise = []
    for i in range(len(data['scan'])):
        if data['laser_pulses'][i] == pulses:
            temp.append(data['ccd_temp'][i])
            F_noise.append(list(data['F_noise'][i]))
            R_noise.append(list(data['R_noise'][i]))
    print "%3d ppp: %s spectra" % (pulses, len(F_noise))
    print "    mean F noise of +/- %4.1f counts" % np.mean(np.ravel(F_noise))
    print "    mean R noise of +/- %4.1f counts" % np.mean(np.ravel(R_noise))
    color = cmap.to_rgba(pulses)
    plt.errorbar(temp, np.mean(F_noise, axis=1), yerr=np.std(F_noise, axis=1), capsize=3., marker='o', ecolor=color, mec=color, mfc=color, elinewidth=1., linewidth=0., label="%d ppp" % pulses)
plt.xlabel("CCD Temperature (C)")
plt.ylabel("Mean Noise Level (counts)")
plt.show()
    
"""
# ==================================================
# create Raman maps
# ==================================================
"""

print
print "generating Raman band maps..."

x_start, x_end = (400, 4000)

if Plot_Rmaps == True:
    for i in range(0, len(data['scan'])):
        print
        print "%s: %s" % (i, data['scan_name'][i])
        
        x_coords, y_coords = data['xy_coords'][i]
        
        # select bands for plotting
        if len(Raman_bands['center']) > 0 and len(Raman_bands['center']) == len(Raman_bands['width']):
            # use specified band list
            sort = np.argsort(Raman_bands['center'])
            band_centers = np.asarray(Raman_bands['center'])[sort]
            band_widths = np.asarray(Raman_bands['width'])[sort]
        else:
            # use default list
            if data['target'][i] in Default_R_bands.keys():
                sort = np.argsort(Default_R_bands[data['target'][i]]['center'])
                band_centers = np.asarray(Default_R_bands[data['target'][i]]['center'])[sort]
                band_widths = np.asarray(Default_R_bands[data['target'][i]]['width'])[sort]
            else:
                sort = np.argsort(Default_R_bands[data['Default']]['center'])
                band_centers = np.asarray(Default_R_bands['Default']['center'])[sort]
                band_widths = np.asarray(Default_R_bands['Default']['width'])[sort]
        print "Raman bands for plotting:", band_centers
        
        aspect = float(np.size(data['grid_index'][i], axis=0)) / float(np.size(data['grid_index'][i], axis=1))
        print "map x/y aspect ratio:", (1./aspect)
        plt.figure(figsize=(4.*len(band_centers), 4.*aspect+2.))
        plt.suptitle("Raman Band Intensities\n%s" % data['scan_name'][i])
        colorbar_factor = 8
        
        clipping = False
        clip = 0.04
        vmins = []
        vmaxs = []
        if Band_normalisation == 'fixed' and Cmin != None and Cmax != None:
            print "using pre-assigned upper/lower intensity limits for map normalisation"
            norm = 'fixed'
            vmins = np.full((len(band_centers)), Cmin)
            vmaxs = np.full((len(band_centers)), Cmax)
        elif Band_normalisation in ['all', 'all_bands', 'full', 'norm']:
            print "normalising maps using %0.2f and %0.2f percentiles of all band values" % (clip, 1.-clip)
            norm = 'fixed'
            band_maps = []
            for center, width in zip(band_centers, band_widths):
                temp, vmin, vmax = map_spectral_band(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], center, width)
                band_maps.append(temp)
            temp = np.sort(np.ravel(band_maps))
            print np.shape(temp)
            indx = int(np.ceil(clip*len(temp)))
            print "    clip percentage: %0.1f%%" % (100.*indx/len(temp))
            print "    min clipping indx:", indx, "of", len(temp), "value:", temp[indx]
            print "    max clipping indx:", len(temp)-indx, "of", len(temp), "value:", temp[-indx]
            vmins = np.full((len(band_centers)), temp[indx])
            vmaxs = np.full((len(band_centers)), temp[-indx])
        else:
            print "defaulting to individual band normalisation, using %0.2f - %0.2f clipping" % (clip, 1.-clip)
            norm = 'full'
            vmins = np.full((len(band_centers)), None)
            vmaxs = np.full((len(band_centers)), None)
            clipping=True
        print
        print "band normalisation:", norm
        print "vmins:, vmaxes:", vmins, vmaxs
        print "clipping:", clipping
        print "clip percentage:", clip
        print
        
        # adjust size of histogram bins based on data volume
        if np.size(np.ravel(band_maps[0]))>1000:
            bins = 20
        else:
            bins = 10
            
        subplot_width = colorbar_factor*len(band_centers)+1
        i2 = 0
        for band_map, center, width, color, vmin, vmax in zip(band_maps, band_centers, band_widths, hist_colors, vmins, vmaxs):
            print "    band %0.f, from %0.f to %0.f cm-1" % (center, center-width, center+width)
            if i2 == 0:
                ax1 = plt.subplot2grid((3, subplot_width), (0, colorbar_factor*i2), colspan=colorbar_factor)
                ax1.set_title("%0.f-%0.f cm$^{-1}$" % (center-width, center+width))
                ax1.hist(np.ravel(band_map), bins=bins, range=(vmin, vmax), color=color)
            else:
                plt.subplot2grid((3, subplot_width), (0, colorbar_factor*i2), colspan=colorbar_factor, sharey=ax1)
                plt.title("%0.f-%0.f cm$^{-1}$" % (center-width, center+width))
                plt.hist(np.ravel(band_map), bins=bins, range=(vmin, vmax), color=color)
                plt.yticks([])
            plt.subplot2grid((3, subplot_width), (1, colorbar_factor*i2), colspan=colorbar_factor, rowspan=2)
            plt.imshow(band_map, aspect='equal', vmin=vmin, vmax=vmax, cmap=Cmap)
            if i2 != 0:
                plt.yticks([])
            i2 += 1
        ax2 = plt.subplot2grid((3, subplot_width), (1, subplot_width-1), rowspan=2)
        plt.colorbar(cax=ax2, shrink=0.9, label="intensity (counts)")
        plt.savefig("%sSol-%s_%s_Raman_maps.png" % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
        plt.show()
        
        # plot brightest spectra for each band
        brightest_indices = []
        for band_map, center, width, color, vmin, vmax in zip(band_maps, band_centers, band_widths, hist_colors, vmins, vmaxs):
            plt.figure(figsize=(10,4))
            plt.suptitle("%s\n%0.f-%0.f cm$^{-1}$ Intensity Map" % (data['scan_name'][i], center-width, center+width))
            ax1 = plt.subplot2grid((1,2), (0,0))
            ax2 = plt.subplot2grid((1,2), (0,1), colspan=1)
            ax2.set_xlim(x_start, x_end)
            vmin = 0.
            if Plot_on_ACI == True:
                map_overlay, extent, norm = convert_map_to_spot_image(band_map, data['grid_index'][i], data['img_coords'][i], vmin=vmin, vmax=vmax, cmap=Cmap)
                plt.imsave('%sSol-%s_%s_Rmap_%0.fcm_img.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], center), map_overlay)
            im = ax1.imshow(band_map, aspect='equal', vmin=vmin, vmax=vmax)
            plt.imsave('%sSol-%s_%s_Rmap_%0.fcm.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], center), np.repeat(np.repeat(band_map, 5, axis=0), 5, axis=1), vmin=vmin, vmax=vmax, cmap=Cmap)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            y_max = find_max(data['raman_shift'][i], np.mean(data['y_r_sub'][i], axis=0), x_start, x_end)[1]
            ax2.axvspan(center-width, center+width, color='k', alpha=0.1)

            # find hottest point and add to list for plotting separately
            print
            print "    hottest point:",
            point = np.argmax(np.ravel(band_map))
            print point
            serp_indx = np.unravel_index(point, np.shape(band_map))
            ### print "        serpentine index:", serp_indx
            map_indx = data['grid_index'][i][serp_indx]
            ### print "        map index:", map_indx
            print "        integrated intensity: %0.f counts" % (band_map.flatten()[point])
            brightest_indices.append(map_indx)
            # find 3 hottest points
            hottest_points = np.flip(np.argsort(np.ravel(band_map))[-6:])
            print hottest_points
            y_max = 0.
            for point in hottest_points:
                print "    point", point
                serp_indx = np.unravel_index(point, np.shape(band_map))
                ### print "        serpentine index:", serp_indx
                map_indx = data['grid_index'][i][serp_indx]
                ### print "        map index:", map_indx
                map_indx_2d = np.unravel_index(map_indx, np.shape(band_map))
                ### print "       2D:", map_indx_2d
                point_x = x_coords[map_indx_2d]
                point_y = y_coords[map_indx_2d]
                ### print "        x,y coords:", point_x, point_y
                print "        integrated intensity: %0.f counts" % (np.ravel(band_map)[point])
                ax1.text(serp_indx[1], serp_indx[0], "%0.f" % (map_indx), color='w')
                ax2.plot(data['raman_shift'][i], data['y_r_sub'][i][map_indx], label='point %s' % (map_indx))
                if find_max(data['raman_shift'][i], data['y_r_sub'][i][map_indx], x_start, x_end)[1] > y_max:
                    y_max = find_max(data['raman_shift'][i], data['y_r_sub'][i][map_indx], x_start, x_end)[1]
            ax2.plot(data['raman_shift'][i], np.mean(data['y_r_sub'][i], axis=0), 'k', label='mean')
            ax2.legend()
            ax2.set_ylim(-0.2*y_max, 1.2*y_max)
            ax2.minorticks_on()
            plt.tight_layout(rect=[0, 0., 1, 0.9])
            plt.savefig('%sSol-%s_%s_Rmap_%0.fcm_spec.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], center), dpi=300)
            plt.show()
            
        # produce figure showing best spectra for each band
        plt.figure(figsize=(8,4))
        plt.suptitle("%s\nBrightest Raman Spectra" % (data['scan_name'][i]))
        ax1 = plt.subplot(111)
        ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax1.set_ylabel("Intensity (counts)")
        ax1.set_xlim(x_start, x_end)
        # add wavenumber ticks to top of ax1
        ax1_top = ax1.twiny()
        x_range = ax1.get_xlim()
        major_labels, major_locations, minor_locations = get_wavelength_ticks(x_range)
        ax1_top.set_xticks(major_locations)
        ax1_top.set_xticklabels(major_labels)
        ax1_top.xaxis.set_minor_locator(plt.FixedLocator(minor_locations))
        ax1_top.set_xlabel("Wavelength (nm)")
        R_max = find_max(data['raman_shift'][i], data['y_r_av_sub'][i], x_range[0], x_range[1])[1]
        for band_map, center, width, color, vmin, vmax, indx in zip(band_maps, band_centers, band_widths, hist_colors, vmins, vmaxs, brightest_indices):
            ax1.plot(data['raman_shift'][i], data['y_r_sub'][i][indx], label="%0.f cm" % (center))
            y_max = find_max(data['raman_shift'][i], data['y_r_sub'][i][indx], x_range[0], x_range[1])[1]
            if y_max > R_max:
                R_max = R_max
        ax1.plot(data['raman_shift'][i], data['y_r_av_sub'][i], 'k', label="Average")
        ax1.legend()
        ax1.set_ylim(-0.2*R_max, 1.2*R_max)
        ax1.minorticks_on()
        plt.tight_layout(rect=[0, 0., 1, 0.93])
        plt.savefig('%sSol-%s_%s_best-R.png' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
        plt.show()
        
        # define RGB channels
        if len(R_RGB_bands['center']) == 3 and len(R_RGB_bands['width']) == 3:
            sort = np.argsort(R_RGB_bands['center'])
            R_centers = np.asarray(R_RGB_bands['center'])[sort]
            R_widths = np.asarray(R_RGB_bands['width'])[sort]
        else:
            R_centers = np.asarray([275, 305, 340])
            R_widths = np.asarray([5, 5, 5])
        
        # plot RGB map of Raman intensities
        RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], R_centers, R_widths, norm='full', floor=0.5, ceiling=0.98)
        print np.shape(RGB_map)
        plt.imsave('%sSol-%s_%s_Rmap_R=%0.f_G=%0.f_B=%0.f.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], R_centers[0], R_centers[1], R_centers[2]), np.repeat(np.repeat(RGB_map, 5, axis=0), 5, axis=1))
        if Plot_on_ACI == True:
            # save map as a grid of spots in ACI pixel space
            map_overlay, extent, norm = convert_map_to_spot_image(RGB_map, data['grid_index'][i], data['img_coords'][i], vmin=vmin, vmax=vmax, cmap=Cmap)
            plt.imsave('%sSol-%s_%s_Rmap_R=%0.f_G=%0.f_B=%0.f_img.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], R_centers[0], R_centers[1], R_centers[2]), map_overlay)

    
"""
==================================================
import terrestrial standards
==================================================
"""

# import brassboard standard spectra

BB_F_data = {
    'standard': [],
    'wavelength': [],
    'y_av': [],
    'y_std': [],
    'y_av_norm': [],
    'y_std_norm': [],
    'laser_pulses': [],
    'laser_current': [],
    'laser_dose': []
}
BB_R_data = {
    'standard': [],
    'raman_shift': [],
    'y_av_sub': [],
    'y_std_sub': [],
    'y_av_sub_norm': [],
    'y_std_sub_norm': [],
    'laser_pulses': [],
    'laser_current': [],
    'laser_dose': [],
    'fitted_peaks': []
}

BB = False
if Compare == True:
    print
    if len(BB_samples) == 0:
        if len(np.unique(data['target'])) == 1 and data['target'][0] in Default_comparisons.keys():
            BB_samples = Default_comparisons[data['target'][0]]
        else:
            BB_samples = Default_comparisons['Default']
    if len(BB_samples) > 0:
        print "importing Brassboard standard spectra..."
    else:
        print "skipping Brassboard standards"
    # attempt to import BB standards
    for standard in BB_samples:
        if standard not in BB_F_data['standard']:
            print "    attempting to import %s Fluor" % standard
            BB_F = False
            while True:
                try:
                    # attempt to import Fluorescence spectrum
                    spec_dirs = glob.glob('%s%s/*_*ppp_*A_*_Fluor*spectra.txt' % (BB_data_dir, standard))
                    print "        files found:", len(spec_dirs)
                    # find spectrum with closest matching laser settings
                    pulse_temp = []
                    current_temp = []
                    dose_temp = []
                    for spec in spec_dirs:
                        # extract laser info from each filename
                        spec_split = spec.split("/")[-1].split("_")
                        pulse_temp.append(float(spec_split[2][:-3]))
                        current_temp.append(float(spec_split[3][:-1]))
                        dose_temp.append(float(spec_split[4][:-2]))
                    print "        pulses:", pulse_temp
                    print "        currents:", current_temp
                    print "        doses:", dose_temp
                    
                    # find best match for Fluor settings
                    both_test = np.logical_and(np.asarray(current_temp) == BB_F_current, np.asarray(pulse_temp) == BB_F_pulses)
                    print "        both_test:", both_test
                    if np.any(both_test) == True:
                        # then at least one spectrum has matching settings
                        indx = np.arange(0, len(spec_dirs))[both_test][-1]
                        print "        match found: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    elif np.any(pulse_temp == BB_R_pulses) == True:
                        # then at least one spectrum has a matching pulse count
                        indx = np.arange(0, len(spec_dirs))[pulse_temp == BB_R_pulses][-1]
                        print "        nearest match found: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    else:
                        # default to highest dose
                        indx = np.argmax(dose_temp)
                    spec_dir = spec_dirs[indx]
                    print "        selected:", spec_dir
                    # extract info from filename
                    spec_split = spec_dir.split("/")[-1].split("_")
                    print "             file info:", spec_split
                    # import data from txt file
                    spec = np.genfromtxt(spec_dir, delimiter="\t").transpose()
                    print "        imported array:", np.shape(spec)
                    # find laser info
                    pulse_temp = float(spec_split[2][:-3])
                    current_temp = float(spec_split[3][:-1])
                    dose_temp = float(spec_split[4][:-2])
                    print "        BB laser values: %0.0f ppp, %0.0f A, %0.2f uJ" % (pulse_temp, current_temp, dose_temp)
                    # check saturation
                    test = spec[2] > 60000.
                    saturation_temp = float(np.sum(test)) / float(np.size(spec[2]))
                    print "        spectrum saturation: %0.2f" % saturation_temp
                    if 'Fluor' in spec_split:
                        BB_F = True
                        # store Fluor data in brassboard dict
                        BB_F_data['standard'].append(standard)
                        BB_F_data['wavelength'].append(spec[0])
                        BB_F_data['y_av'].append(spec[2])
                        BB_F_data['y_std'].append(spec[3])
                        BB_F_data['y_av_norm'].append(spec[4])
                        BB_F_data['y_std_norm'].append(spec[5])
                        BB_R_data['laser_pulses'].append(pulse_temp)
                        BB_R_data['laser_current'].append(current_temp)
                        BB_R_data['laser_dose'].append(dose_temp)
                        print "    successfully imported BB Fluor spectrum for %s!" % standard
                    else:
                        print "    attempted to import a non-Fluor file!"
                    break
                except Exception as e:
                    print "    something went wrong! Exception:", e
                    break
        if standard not in BB_R_data['standard']:
            print "    attempting to import %s Raman" % standard
            BB_R = False
            while True:
                try:
                    # attempt to import Raman spectrum
                    spec_dirs = glob.glob('%s%s/*_*ppp_*A_*_Raman*spectra.txt' % (BB_data_dir, standard))
                    print "        files found:", len(spec_dirs)
                    # find spectrum with closest matching laser settings
                    pulse_temp = []
                    current_temp = []
                    dose_temp = []
                    for spec in spec_dirs:
                        # extract laser info from each filename
                        spec_split = spec.split("/")[-1].split("_")
                        pulse_temp.append(float(spec_split[2][:-3]))
                        current_temp.append(float(spec_split[3][:-1]))
                        dose_temp.append(float(spec_split[4][:-2]))
                    print "        pulses:", pulse_temp
                    print "        currents:", current_temp
                    print "        doses:", dose_temp
                    # find best match for Raman settings
                    both_test = np.logical_and(np.asarray(current_temp) == BB_R_current, np.asarray(pulse_temp) == BB_R_pulses)
                    print "        both_test:", both_test
                    if np.any(both_test) == True:
                        # then at least one spectrum has matching settings
                        indx = np.arange(0, len(spec_dirs))[both_test][-1]
                        print "        match found: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    elif np.any(pulse_temp == BB_R_pulses) == True:
                        # then at least one spectrum has a matching pulse count
                        indx = np.arange(0, len(spec_dirs))[pulse_temp == BB_R_pulses][-1]
                        print "        nearest match found: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    else:
                        # default to highest dose
                        indx = np.argmax(dose_temp)
                    spec_dir = spec_dirs[indx]
                    print "        selected:", spec_dir
                    # extract info from filename
                    spec_split = spec_dir.split("/")[-1].split("_")
                    print "             file info:", spec_split
                    # import data from txt file
                    spec = np.genfromtxt(spec_dir, delimiter="\t").transpose()
                    print "        imported array:", np.shape(spec)
                    # find laser info
                    pulse_temp = float(spec_split[2][:-3])
                    current_temp = float(spec_split[3][:-1])
                    dose_temp = float(spec_split[4][:-2])
                    print "        BB laser values: %0.0f ppp, %0.0f A, %0.2f uJ" % (pulse_temp, current_temp, dose_temp)
                    # check saturation
                    test = spec[2] > 60000.
                    saturation_temp = float(np.sum(test)) / float(np.size(spec[2]))
                    print "        spectrum saturation: %0.2f" % saturation_temp
                    if 'Raman' in spec_split:
                        BB_R = True
                        # store Raman data in brassboard dict
                        BB_R_data['standard'].append(standard)
                        BB_R_data['raman_shift'].append(spec[1])
                        BB_R_data['y_av_sub'].append(spec[6])
                        BB_R_data['y_std_sub'].append(spec[7])
                        BB_R_data['y_av_sub_norm'].append(spec[8])
                        BB_R_data['y_std_sub_norm'].append(spec[9])
                        BB_R_data['laser_pulses'].append(pulse_temp)
                        BB_R_data['laser_current'].append(current_temp)
                        BB_R_data['laser_dose'].append(dose_temp)
                    
                        print "    successfully imported BB Raman spectrum for %s!" % standard
                        peak_dirs = glob.glob('%s%s/*_fitted_peaks.txt' % (BB_data_dir, standard))
                        print "        peak fit files found:", len(peak_dirs)
                        if len(peak_dirs) > 0:
                            fitted_peaks_temp = np.genfromtxt(peak_dirs[0], delimiter="\t").transpose()
                            print "        peak fit array:", np.shape(fitted_peaks_temp)
                            BB_R_data['fitted_peaks'].append(fitted_peaks_temp)
                            print "        successfully imported peak fits for %s!" % standard
                        else:
                            BB_R_data['fitted_peaks'].append([])
                    else:
                        print "    attempted to import a non-Raman file!"
                    break
                except Exception as e:
                    print "    something went wrong! Exception:", e
                    break
    if len(BB_R_data['standard']) > 0 or len(BB_F_data['standard']) > 0:
        BB = True
    

# import MOBIUS standards

MOB_F_data = {
    'standard': [],
    'wavelength': [],
    'y_av': [],
    'y_std': [],
    'y_av_norm': [],
    'y_std_norm': [],
    'laser_pulses': [],
    'laser_current': [],
    'laser_dose': []
}

MOB_R_data = {
    'standard': [],
    'raman_shift': [],
    'y_av_sub': [],
    'y_std_sub': [],
    'y_av_sub_norm': [],
    'y_std_sub_norm': [],
    'laser_pulses': [],
    'laser_current': [],
    'laser_dose': [],
    'fitted_peaks': []
}

MOB = False
if Compare == True:
    print
    if len(MOB_samples) > 0:
        print "importing MOBIUS standard spectra..."
    else:
        print "skipping MOBIUS standards"

    # attempt to import MOBIUS standards
    for standard in MOB_samples:
        if standard not in MOB_F_data['standard']:
            print "    attempting to import MOBIUS Fluor spectra for %s" % standard
            MOB_F = False
            while True:
                try:
                    # import Fluorescence
                    spec_dirs = glob.glob('%s%s/*_*ppp_*A_*_Fluor_spectra.txt' % (MOB_F_data_dir, standard))
                    print "        Fluor files found:", len(spec_dirs)
                    # find spectrum with closest matching laser settings
                    pulse_temp = []
                    current_temp = []
                    dose_temp = []
                    for spec in spec_dirs:
                        # extract laser info from each filename
                        spec_split = spec.split("/")[-1].split("_")
                        pulse_temp.append(float(spec_split[2][:-3]))
                        current_temp.append(float(spec_split[3][:-1]))
                        dose_temp.append(float(spec_split[4][:-2]))
                    print "        pulses:", pulse_temp
                    print "        currents:", current_temp
                    print "        doses:", dose_temp
                    both_test = np.logical_and(np.asarray(current_temp) == MOB_F_current, np.asarray(pulse_temp) == MOB_F_pulses)
                    print "        both_test:", both_test
                    if np.any(both_test) == True:
                        # then at least one spectrum has matching settings
                        indx = np.arange(0, len(spec_dirs))[both_test][-1]
                        print "        match found: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    elif np.any(pulse_temp == MOB_F_pulses) == True:
                        # then at least one spectrum has a matching pulse count
                        indx = np.arange(0, len(spec_dirs))[pulse_temp == MOB_F_pulses][-1]
                        print "        nearest match found: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    else:
                        # no matches, default to highest-dosage exposure
                        indx = np.argmax(dose_temp)
                        print "        no matches, using highest energy dosage: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    spec_dir = spec_dirs[indx]
                    print "        selected:", spec_dir
                    # extract info from filename
                    spec_split = spec_dir.split("/")[-1].split("_")
                    print "             file info:", spec_split
                    # import data from txt file
                    spec = np.genfromtxt(spec_dir, delimiter="\t").transpose()
                    # find laser info
                    pulse_temp = float(spec_split[2][:-3])
                    current_temp = float(spec_split[3][:-1])
                    dose_temp = float(spec_split[4][:-2])
                    print "        MOBIUS laser values: %0.0f ppp, %0.0f A, %0.2f uJ" % (pulse_temp, current_temp, dose_temp)
                    # check saturation
                    test = spec[2] > 60000.
                    saturation_temp = float(np.sum(test)) / float(np.size(spec[2]))
                    print "        spectrum saturation: %0.2f" % saturation_temp
                    MOB_F = True
                    MOB_F_data['standard'].append(standard)
                    MOB_F_data['wavelength'].append(spec[0])
                    MOB_F_data['y_av'].append(spec[2])
                    MOB_F_data['y_std'].append(spec[3])
                    MOB_F_data['y_av_norm'].append(spec[4])
                    MOB_F_data['y_std_norm'].append(spec[5])
                    MOB_F_data['laser_pulses'].append(pulse_temp)
                    MOB_F_data['laser_current'].append(current_temp)
                    MOB_F_data['laser_dose'].append(dose_temp)
                    print "    successfully imported MOBIUS Fluor spectrum for %s!" % standard
                    break
                except Exception as e:
                    print "    something went wrong! Exception:", e
                    break
        if standard not in MOB_R_data['standard']:
            print "    attempting to import MOBIUS Raman spectra for %s" % standard
            MOB_R = False
            while True:
                try:
                    # import Raman
                    spec_dirs = glob.glob('%s%s/*_*ppp_*A_*_Raman_spectra.txt' % (MOB_R_data_dir, standard))
                    print "        Raman files found:", len(spec_dirs)
                    # find spectrum with closest matching laser settings
                    pulse_temp = []
                    current_temp = []
                    dose_temp = []
                    for spec in spec_dirs:
                        # extract laser info from each filename
                        spec_split = spec.split("/")[-1].split("_")
                        pulse_temp.append(float(spec_split[2][:-3]))
                        current_temp.append(float(spec_split[3][:-1]))
                        dose_temp.append(float(spec_split[4][:-2]))
                    print "        pulses:", pulse_temp
                    print "        currents:", current_temp
                    print "        doses:", dose_temp
                    both_test = np.logical_and(np.asarray(current_temp) == MOB_R_current, np.asarray(pulse_temp) == MOB_R_pulses)
                    print "        both_test:", both_test
                    if np.any(both_test) == True:
                        # then at least one spectrum has matching settings
                        indx = np.arange(0, len(spec_dirs))[both_test][-1]
                        print "        match found: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    elif np.any(pulse_temp == MOB_R_pulses) == True:
                        # then at least one spectrum has a matching pulse count
                        indx = np.arange(0, len(spec_dirs))[pulse_temp == MOB_R_pulses][-1]
                        print "        nearest match found: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    else:
                        # no matches, default to highest-dosage exposure
                        indx = np.argmax(dose_temp)
                        print "        no matches, using highest energy dosage: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                    spec_dir = spec_dirs[indx]
                    print "        selected:", spec_dir
                    # extract info from filename
                    spec_split = spec_dir.split("/")[-1].split("_")
                    print "             file info:", spec_split
                    # import data from txt file
                    spec = np.genfromtxt(spec_dir, delimiter="\t").transpose()
                    print "        imported array:", np.shape(spec)
                    # find laser info
                    pulse_temp = float(spec_split[2][:-3])
                    current_temp = float(spec_split[3][:-1])
                    dose_temp = float(spec_split[4][:-2])
                    print "        MOBIUS laser values: %0.0f ppp, %0.0f A, %0.2f uJ" % (pulse_temp, current_temp, dose_temp)
                    # check saturation
                    test = spec[2] > 60000.
                    saturation_temp = float(np.sum(test)) / float(np.size(spec[2]))
                    print "        spectrum saturation: %0.2f" % saturation_temp
                    # store data in mobius dict
                    MOB_R = True
                    MOB_R_data['standard'].append(standard)
                    MOB_R_data['raman_shift'].append(spec[1])
                    MOB_R_data['y_av_sub'].append(spec[6])
                    MOB_R_data['y_std_sub'].append(spec[7])
                    MOB_R_data['y_av_sub_norm'].append(spec[8])
                    MOB_R_data['y_std_sub_norm'].append(spec[9])
                    MOB_R_data['laser_pulses'].append(pulse_temp)
                    MOB_R_data['laser_current'].append(current_temp)
                    MOB_R_data['laser_dose'].append(dose_temp)
                    print "    successfully imported MOBIUS Raman spectrum for %s!" % standard
                    peak_dirs = glob.glob('%s%s/*_fitted_peaks.txt' % (MOB_R_data_dir, standard))
                    print "        peak fit files found:", len(peak_dirs)
                    if len(peak_dirs) > 0:
                        fitted_peaks_temp = np.genfromtxt(peak_dirs[0], delimiter="\t").transpose()
                        print "        peak fit array:", np.shape(fitted_peaks_temp)
                        MOB_R_data['fitted_peaks'].append(fitted_peaks_temp)
                    print "        successfully imported peak fits for %s!" % standard
                    break
                except Exception as e:
                    print "    something went wrong! Exception:", e
                    break
    if len(MOB_F_data['standard']) > 0 or len(MOB_R_data['standard']) > 0:
        MOB = True
        
if Compare == True:
    if BB == True:
        print
        print "BB standards imported:", np.unique(BB_R_data['standard']+BB_F_data['standard'])
        print "BB spectra imported: %d F, %d R" % (len(BB_F_data['standard']), len(BB_R_data['standard']))
    if MOB == True:
        print
        print "MOBIUS standards imported:", np.unique(MOB_R_data['standard']+MOB_F_data['standard'])
        print "MOBIUS spectra imported: %d F, %d R" % (len(MOB_F_data['standard']), len(MOB_R_data['standard']))

"""
# ==================================================
# plot ROIs (defined by x,y coordinates)
# ==================================================
"""

# ROIs are indexed like so: ROIs[sol (4 digit string)][sequence][radius (in pixels), point1, point2, etc.]
ROIs = {}

if Plot_ROIs == True:
    print
    print "plotting manually-specified ROIs"
    for i in range(0, len(data['scan'])):
        print
        print "%s: %s" % (i, data['scan_name'][i])
        sol = data['sol'][i]
        scan = data['scan'][i]
        x_coords, y_coords = data['xy_coords'][i]
        
        if sol in ROIs.keys():
            if scan in ROIs[sol].keys():

                # define RGB channels
                if len(F_RGB_bands['center']) == 3 and len(F_RGB_bands['width']) == 3:
                    sort = np.argsort(F_RGB_bands['center'])
                    F_centers = np.asarray(F_RGB_bands['center'])[sort]
                    F_widths = np.asarray(F_RGB_bands['width'])[sort]
                else:
                    F_centers = np.asarray([275, 305, 340])
                    F_widths = np.asarray([5, 5, 5])

                if len(R_RGB_bands['center']) == 3 and len(R_RGB_bands['width']) == 3:
                    sort = np.argsort(R_RGB_bands['center'])
                    R_centers = np.asarray(R_RGB_bands['center'])[sort]
                    R_widths = np.asarray(R_RGB_bands['width'])[sort]
                else:
                    R_centers = np.asarray([840, 1020, 1090])
                    R_widths = np.asarray([25, 25, 25])

                colors = []
                plt.figure(figsize=(12,6))
                plt.suptitle("%s ROIs" % (data['save_name'][i]))
                ax0 = plt.subplot2grid((2,3), (0,0), rowspan=2)
                ax1 = plt.subplot2grid((2,3), (0,1), rowspan=2)
                ax2 = plt.subplot2grid((2,3), (0,2))
                ax3 = plt.subplot2grid((2,3), (1,2))
                ax0.set_title("Fluorescence\nR=%0.f, G=%0.f, B=%0.f nm" % (F_centers[0], F_centers[1], F_centers[2]))
                ax0.set_xlim(-0.5, np.size(x_coords, axis=1)-0.5)
                ax0.set_ylim(np.size(x_coords, axis=0)-0.5, -0.5)
                ax1.set_title("Raman\nR=%0.f, G=%0.f, B=%0.f cm-1" % (R_centers[0], R_centers[1], R_centers[2]))
                ax1.set_xlim(-0.5, np.size(x_coords, axis=1)-0.5)
                ax1.set_ylim(np.size(x_coords, axis=0)-0.5, -0.5)
                ax2.set_xlim(250, 355)
                ax2.set_xlabel("Wavelength (nm)")
                x_start, x_end = (np.amin(R_centers-R_widths-100), np.amax(R_centers+R_widths+100))
                ax3.set_xlim(x_start, x_end)
                ax3.set_xlabel("Raman Shift (cm$^{-1}$)")
                print "map position array:", np.shape(data['grid_index'][i])
                RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(data['wavelength'][i], data['y'][i], data['grid_index'][i], F_centers, F_widths, norm='full', floor=0.5, ceiling=0.98)
                print np.shape(RGB_map)
                for center, width, color in zip(F_centers, F_widths, ['r','g','b']):
                    ax2.axvspan(center-width, center+width, color=color, alpha=0.1)
                plt.imsave('%sSol-%s_%s_Fmap_R=%0.f_G=%0.f_B=%0.f.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], F_centers[0], F_centers[1], F_centers[2]), np.repeat(np.repeat(RGB_map, 5, axis=0), 5, axis=1))
                if Plot_on_ACI == True:
                    # save map as a grid of spots in ACI pixel space
                    map_overlay, extent = convert_map_to_spot_image(RGB_map, data['grid_index'][i], data['img_coords'][i], vmin=vmin, vmax=vmax, cmap=Cmap)
                    plt.imsave('%sSol-%s_%s_Fmap_R=%0.f_G=%0.f_B=%0.f_img.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], F_centers[0], F_centers[1], F_centers[2]), map_overlay)
                # plot RGB map
                im = ax0.imshow(RGB_map, aspect='equal')
                RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], R_centers, R_widths, norm='full', clip=0.02)
                print np.shape(RGB_map)
                for center, width, color in zip(R_centers, R_widths, ['r','g','b']):
                    ax2.axvspan(center-width, center+width, color=color, alpha=0.1)
                plt.imsave('%sSol-%s_%s_Rmap_R=%0.f_G=%0.f_B=%0.f.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], R_centers[0], R_centers[1], R_centers[2]), np.repeat(np.repeat(RGB_map, 5, axis=0), 5, axis=1))
                if Plot_on_ACI == True:
                    # save map as a grid of spots in ACI pixel space
                    map_overlay, extent = convert_map_to_spot_image(RGB_map, data['grid_index'][i], data['img_coords'][i], vmin=vmin, vmax=vmax, cmap=Cmap)
                    plt.imsave('%sSol-%s_%s_Rmap_R=%0.f_G=%0.f_B=%0.f_img.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], R_centers[0], R_centers[1], R_centers[2]), map_overlay)
                # plot RGB map
                im = ax1.imshow(RGB_map, aspect='equal')
                # find mean F/R maxima for settings y limits
                F_max = find_max(data['wavelength'][i], np.mean(data['y'][i], axis=0), 255, 355)[1]
                R_max = find_max(data['raman_shift'][i], np.mean(data['y_r_sub'][i], axis=0), x_start, x_end)[1]
                # plot mean spectra for whole scan
                ### ax2.plot(data['wavelength'][i], np.mean(data['y'][i], axis=0), 'k', label='mean')
                ### ax3.plot(data['raman_shift'][i], np.mean(data['y_r_sub'][i], axis=0), 'k', label='mean')
                num = 0
                mask = np.full_like(x_coords, True)
                ROI_F = []
                ROI_R = []
                # plot each ROI
                for roi in ROIs[sol][scan][1:]:
                    num += 1
                    print "        ROI %s: (x,y) =" % num, roi
                    # plot ROI area on ax1 image
                    circle_azi = np.linspace(-np.pi, np.pi, 100)
                    circle_x = roi[0] + width * np.cos(circle_azi)
                    circle_y = roi[1] + width * np.sin(circle_azi)
                    ax0.plot(circle_x, circle_y, 'w')
                    ax0.text(roi[0], roi[1], num, color='w')
                    ax1.plot(circle_x, circle_y, 'w')
                    ax1.text(roi[0], roi[1], num, color='w')
                    roi_mean = area_mean(data['wavelength'][i], data['y'][i], data['grid_index'][i], [x_coords, y_coords], roi, shape='circle', width=width)
                    roi_smooth = smooth_f(roi_mean, 10, 3)
                    ROI_F.append(roi_mean)
                    ax2.plot(data['wavelength'][i], roi_mean, label="ROI #%s" % num)
                    if find_max(data['wavelength'][i], roi_mean, 255, 355)[1] > F_max:
                        F_max = find_max(data['wavelength'][i], roi_mean, 255, 355)[1]
                    roi_mean = area_mean(data['raman_shift'][i], data['y_r_sub'][i], ['grid_index'][i], [x_coords, y_coords], roi, shape='circle', width=width)
                    ROI_R.append(roi_mean)
                    ax3.plot(data['raman_shift'][i], roi_mean, label="ROI #%s" % num)
                    if find_max(data['raman_shift'][i], roi_mean, x_start, x_end)[1] > R_max:
                        R_max = find_max(data['raman_shift'][i], roi_mean, x_start, x_end)[1]
                    ### print np.shape(mask)
                    ### print np.shape(np.absolute(xy_coords[0] - roi[1])**2 + np.absolute(xy_coords[1] - roi[0])**2 > width**2)
                    mask = np.logical_and(mask, np.absolute(x_coords - roi[1])**2 + np.absolute(y_coords - roi[0])**2 > width**2)
                if np.all(mask) == True:
                    # plot mean spectra for all points
                    ax2.plot(data['wavelength'][i], np.mean(data['y'][i], axis=0), 'k', label='non-ROI')
                    ax3.plot(data['raman_shift'][i], np.mean(data['y_r_sub'][i], axis=0), 'k', label='non-ROI')
                else:
                    # plot mean spectra for all points outside ROIs
                    print "    spectra outside ROIs:", np.shape(data['y_r_sub'][i][data['grid_index'][i]][mask])
                    ax2.plot(data['wavelength'][i], np.mean(data['y'][i][data['grid_index'][i]][mask], axis=0), 'k', label='non-ROI')
                    ax3.plot(data['raman_shift'][i], np.mean(data['y_r_sub'][i][data['grid_index'][i]][mask], axis=0), 'k', label='non-ROI')
                ax2.legend()
                ax2.set_ylim(-0.2*F_max, 1.2*F_max)
                ax3.set_ylim(-0.2*R_max, 1.2*R_max)
                ax2.minorticks_on()
                ax3.minorticks_on()
                plt.tight_layout()
                plt.savefig('%sSol-%s_%s_ROIs.png' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
                plt.savefig('%sSol-%s_%s_ROIs.svg' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
                plt.show()
                
                # plot ROI R/F spectra separately
                if len(ROI_F) > 0:
                    x_range = (250, 355)
                    print x_range
                    plt.figure(figsize=(8,4))
                    plt.suptitle("%s ROI Fluor. Spectra" % (data['save_name'][i]))
                    ax1 = plt.subplot(111)
                    ax1.set_xlabel("Wavelength (nm)")
                    ax1.set_ylabel("Intensity (counts)")
                    ax1.set_xlim(x_range)
                    # add wavenumber ticks to top of ax1
                    ax1_top = ax1.twiny()
                    x_range = ax1.get_xlim()
                    major_labels, major_locations, minor_locations = get_shift_ticks(x_range)
                    ax1_top.set_xticks(major_locations)
                    ax1_top.set_xticklabels(major_labels)
                    ### ax1_top.xaxis.set_minor_locator(plt.FixedLocator(minor_locations))
                    ax1_top.set_xlabel("Raman Shift (cm$^{-1}$)")
                    F_max = find_max(data['wavelength'][i], data['y_av'][i], x_range[0], x_range[1])[1]
                    for i2 in range(0, len(ROI_F)):
                        ax1.plot(data['wavelength'][i], ROI_F[i2], label="ROI %s" % (i2+1))
                        y_max = find_max(data['wavelength'][i], ROI_F[i2], x_range[0], x_range[1])[1]
                        if y_max > F_max:
                            F_max = y_max
                    ax1.plot(data['wavelength'][i], data['y_av'][i], 'k', label='Average')
                    ax1.legend(loc=2)
                    ax1.set_ylim(-0.2*F_max, 1.2*F_max)
                    ax1.minorticks_on()
                    plt.tight_layout(rect=[0, 0., 1, 0.93])
                    plt.savefig('%sSol-%s_%s_ROIs-F.png' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
                    plt.savefig('%sSol-%s_%s_ROIs-F.svg' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
                    plt.show()

                if len(ROI_R) > 0:
                    x_range = (750, 1800)
                    print x_range
                    plt.figure(figsize=(8,4))
                    plt.suptitle("%s ROI Raman Spectra" % (data['save_name'][i]))
                    ax1 = plt.subplot(111)
                    ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
                    ax1.set_ylabel("Intensity (counts)")
                    ax1.set_xlim(x_range)
                    # add wavenumber ticks to top of ax1
                    ax1_top = ax1.twiny()
                    x_range = ax1.get_xlim()
                    major_labels, major_locations, minor_locations = get_wavelength_ticks(x_range)
                    ax1_top.set_xticks(major_locations)
                    ax1_top.set_xticklabels(major_labels)
                    ### ax1_top.xaxis.set_minor_locator(plt.FixedLocator(minor_locations))
                    ax1_top.set_xlabel("Wavelength (nm)")
                    R_max = find_max(data['raman_shift'][i], data['y_r_av_sub'][i], x_range[0], x_range[1])[1]
                    for i2 in range(0, len(ROI_R)):
                        ax1.plot(data['raman_shift'][i], ROI_R[i2], label="ROI %s" % (i2+1))
                        y_max = find_max(data['raman_shift'][i], ROI_R[i2], x_range[0], x_range[1])[1]
                        if y_max > R_max:
                            R_max = y_max
                    ax1.plot(data['raman_shift'][i], data['y_r_av_sub'][i], 'k', label='Average')
                    ax1.legend(loc=1)
                    ax1.set_ylim(-0.2*R_max, 1.2*R_max)
                    ax1.minorticks_on()
                    plt.tight_layout(rect=[0, 0., 1, 0.93])
                    plt.savefig('%sSol-%s_%s_ROIs-R.png' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
                    plt.savefig('%sSol-%s_%s_ROIs-R.svg' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
                    plt.show()

"""
# ==================================================
# plot POIs (defined by point index)
# ==================================================
"""

# POIs are indexed like so: ROIs[sol (to 4 digits)][sequence][[point1, point2, etc.], [standard1, standard2, etc.]]
# point numbers are 0-indexed, if a set of points are put in a nested list then the average will be taken
POIs = {
    '0059': {
        'meteorite': [[[86, 128, 556, 664, 973]], ['Apatite_sxtal', 'Enstatite_C7758', 'Lignite_PSOC-1533']]
    },
    '0161': {
        'HDR_300': [[46, 10, 52, 13], ['Olivine_OLV-SC', 'Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']]
    },
    '0162': {
        'HDR_250_1': [[5, 18, 28, 29, 33, 39, 66, 14], ['Sodium Perchlorate Hydrate_powder', 'Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']], # [18, 37, 39, 1, 21, 23, 66, 2, 28, 29, 30, 41, 89, 98, 0, 6, 7, 14], ['Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']],
        'HDR_250_2': [[5, 18, 33, 37, 39, 1, 14, 23, 40, 66, 73, 17, 28, 29, 30, 41, 98, 84], ['Sodium Perchlorate Hydrate_powder', 'Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']], # [[5, 98], ['Sodium Perchlorate Hydrate_powder', 'Apatite_sxtal']], # [[5, 28, 40, 98, 73, 84], ['Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']],
        'survey_1296': [[215, 362, 356, 427], ['Sodium Perchlorate Hydrate_powder', 'Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']], # [[215, 362, 356, 427], ['Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']]
    },
    '0181': {
        'meteorite_1': [[[53, 73]], ['Apatite_sxtal', 'Enstatite_C7758', 'Lignite_PSOC-1533']],
        ### 'meteorite_2': [[735, 1095], ['Apatite_sxtal', 'Enstatite_C7758', 'Lignite_PSOC-1533']]
        'meteorite_2': [[[946, 947, 948, 949, 997, 1016, 1017, 1072], [52, 673, 735, 827, 921, 993, 1025, 1044, 1045, 1095, 1198], np.concatenate((np.arange(1080, 1088), np.arange(1144, 1160), np.arange(1216, 1231), np.arange(1288, 1295)))], ['Apatite_sxtal', 'Enstatite_C7758', 'Lignite_PSOC-1533']]
    },
    '0186': {
        'HDR_250_1': [[63, 24, 16, 27], ['Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']], # [[56, 63, 94, 24, 38, 48], ['Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']]
        'HDR_250_2': [[5, 82, 88, 16, 27, 34], ['Apatite_sxtal', 'Kieserite_powder', 'Sodium Perchlorate Hydrate_powder']], # [[34, 88, 91, 92, 6, 8, 20, 22, 23, 48, 56, 63, 69, 90, 94, 5, 38, 97], ['Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']],
        'survey_1296': [[[6, 7, 8, 63, 64, 65, 66, 67, 74, 75, 76, 77, 78, 79, 80, 135, 136, 137, 138, 139, 140, 141], [846, 847, 848, 880, 881, 917, 918, 954, 955, 956, 957, 958, 959, 983, 984, 985, 986, 987]], ['Anhydrite_new', 'Gypsum_powder']] # [[48, 67, 68, 76, 77, 591], ['Apatite_sxtal', 'Anhydrite_new', 'Magnesite_sxtal']]
    },
    '0207': {
        'HDR_500_1': [[53, 50, 0, 72], ['Olivine_OLV-SC', 'Calcite_powder', 'Magnesite_sxtal']], # [[0, 1, 4, 5, 8, 15, 16, 17, 21, 23, 24, 28, 31, 39, 40, 41, 47, 48, 50, 51, 53, 64, 67, 69, 72, 73, 77, 78, 88, 91, 97], ['Olivine_OLV-SC', 'Apatite_sxtal', 'Calcite_powder', 'Magnesite_sxtal']],
        'HDR_500_2': [[53, 55, 76, 90, 97], ['Olivine_OLV-SC', 'Apatite_sxtal', 'Calcite_powder', 'Magnesite_sxtal']], # [[0, 1, 4, 5, 6, 9, 16, 17, 19, 28, 34, 43, 47, 50, 51, 53, 54, 55, 56, 58, 59, 62, 76, 77, 83, 90, 95, 97], ['Olivine_OLV-SC', 'Apatite_sxtal', 'Calcite_powder', 'Magnesite_sxtal']]
    },
    '0208': {
        'detailed_center_1': [[18, 17, 22, 23, 44, 53, 54, 65], ['Naphthalene_powder']], # [[[17, 18, 21, 22, 23, 35, 36, 37, 42, 43, 44, 45, 53, 54, 55, 56, 65, 66, 67, 68, 71, 72, 73], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 27, 28, 29, 30, 31, 32, 59, 60, 61, 77, 78, 79, 80, 81, 82, 83, 95, 96, 97, 98, 99]], ['Apatite_Durango-yellow']], # ['Olivine_OLV-SC', 'Apatite_sxtal', 'Calcite_powder', 'Magnesite_sxtal']], # [[34, 51, 52, 65], ['Olivine_OLV-SC', 'Apatite_sxtal', 'Calcite_powder', 'Magnesite_sxtal']],
        'detailed_corner_1': [[3, 56, 45, 81], ['Olivine_OLV-SC', 'Apatite_sxHDR_500_1tal', 'Calcite_powder', 'Magnesite_sxtal']],
        'detailed_corner_2': [[27, 45, 4, 58], ['Olivine_OLV-SC', 'Apatite_sxtal', 'Calcite_powder', 'Magnesite_sxtal']]
    },
    '0257': {
        'HDR_500_1': [[[55, 62, 64, 96], [0, 1, 2]], ['Thenardite_powder', 'Augite_C20339', 'Diopside_C7229']] # [[[1, 22, 23, 29, 49], [10, 12, 16, 18, 19, 20, 21, 26, 27, 30, 33, 34, 35, 37, 38, 40, 41, 44, 47, 53, 56, 59, 60, 61, 67, 69, 70, 71, 74, 76, 89, 93, 94, 98]], ['Olivine_OLV-SC', 'Calcite_powder', 'Magnesite_powder']], # [[20, 53, 1, 2, 22, 23, 29, 41, 52, 55, 62], ['Amorphous Silicate_Synthetic Mg', 'Apatite_sxtal', 'Calcite_powder', 'Gypsum_powder']] # [[[1, 22, 23, 29, 49], [10, 12, 16, 18, 19, 20, 21, 26, 27, 30, 33, 34, 35, 37, 38, 40, 41, 44, 47, 53, 56, 59, 60, 61, 67, 69, 70, 71, 74, 76, 89, 93, 94, 98]], ['Calcite_powder', 'Olivine_OLV-SC']] # [[20, 53, 1, 2, 22, 23, 29, 41, 52, 55, 62], ['Amorphous Silicate_Synthetic Mg', 'Apatite_sxtal', 'Calcite_powder', 'Gypsum_powder']]
    },
    '0269': {
        'detail_1': [[12, 13, 69], ['Gypsum_powder', 'Melanterite_powder', 'Enstatite_C7758']],
        'detail_2': [[39, 94, 9, 45, 70], ['Olivine_OLV-SC', 'Apatite_sxtal', 'Calcite_powder', 'Obsidian_sxtal']]
    },
    '0293': {
        'survey_1296': [[[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 243, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 408, 409, 410, 411, 412, 413, 414, 415, 416, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 436, 437, 438, 439, 440, 441, 442, 447, 448, 449, 450, 451, 452, 453, 481, 482, 483, 484, 485, 486, 487, 495, 496, 497, 521, 522, 523, 524, 525, 526, 552, 553, 554, 555, 556, 557, 595, 596, 597, 598, 599, 624, 625, 626, 627, 628, 668, 669, 670], range(700, 1296)], ['Olivine_OLV-SC', 'Gypsum_powder', 'Melanterite_powder', 'Calcite_powder']],
        'HDR_1': [[3, 8, 15, 24, 25, 27, 34, 36, 45, 47, 52, 55, 57, 58, 67, 74], ['Olivine_OLV-SC', 'Gypsum_powder', 'Melanterite_powder', 'Calcite_powder']]
    },
    '0304': {
        'detail_1': [[3, 9, 10, 12, 18, 21, 27, 31, 32, 34, 39, 44, 46, 47, 67, 73, 76, 83], ['Olivine_OLV-SC', 'Gypsum_powder', 'Melanterite_powder', 'Calcite_powder']],
        'detail_2': [[15, 21, 34, 43, 62, 82, 95, 97], ['Olivine_OLV-SC', 'Gypsum_powder', 'Melanterite_powder', 'Calcite_powder']],
        'detail_3': [[0, 16, 18, 33, 38, 65, 88], ['Olivine_OLV-SC', 'Gypsum_powder', 'Melanterite_powder', 'Calcite_powder']],
        'detail_4': [[4, 5, 6, 11, 31, 48, 49, 63, 73, 76, 83, 96], ['Olivine_OLV-SC', 'Gypsum_powder', 'Melanterite_powder', 'Calcite_powder']]
    },
    '0349': {
        'HDR_1': [[20, 27, 40, 55, 58, 64, 75, 85, 93], ['Obsidian_sxtal', 'Enstatite_C7758', 'Calcite_powder', 'Apatite_sxtal', 'Sodium Perchlorate Hydrate_powder']],
        'survey_1296': [[], ['Obsidian_sxtal', 'Enstatite_C7758', 'Calcite_powder', 'Apatite_sxtal', 'Sodium Perchlorate Hydrate_powder']]
    },
    '0368': {
        'meteorite_1': [[], ['Augite_C20339', 'Enstatite_C7758', 'HOPG_sxtal']]
    },
    '0370': {
        'HDR_1': [[[30, 31, 32, 33, 34, 45, 46, 47, 48, 49], [60, 61, 62, 63, 64, 75, 76, 77, 78, 79], [70, 71, 72, 73, 86, 87, 88, 89, 90, 91, 92, 93], [0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23, 36, 37, 38, 39]], ['Obsidian_sxtal', 'Amorphous Silicate_Synthetic Mg', 'Bytownite_pellet']],
        'survey_1296': [[], ['Obsidian_sxtal', 'Enstatite_C7758', 'Calcite_powder', 'Apatite_sxtal', 'Sodium Perchlorate Hydrate_powder']]
    },
    '0489': {
        'HDR_1': [[[28, 29, 31, 32, 43, 48, 49, 50, 51, 56, 58, 61, 68, 69], [0, 1, 18, 19, 20, 39, 44, 52, 55, 67, 70, 71, 72, 73, 86, 87, 88, 89, 90, 91, 92, 93, 94]], ['Calcite_powder', 'Magnesite_powder', 'Obsidian_sxtal', 'Quartz_powder']], # [[7, 51, 61, 92], ['Calcite_powder', 'Obsidian_sxtal', 'Quartz_powder']],
        'HDR_2': [[[42, 57, 62, 73, 80, 98, 99], [2, 3, 4, 5, 14, 15, 20, 21, 30, 31, 32, 38, 39, 40, 41, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 65, 68, 69, 85, 86, 87, 88, 89, 90, 91, 92, 93,96, 97]], ['Calcite_powder', 'Magnesite_powder', 'Obsidian_sxtal', 'Quartz_powder']], # [[23, 29, 73], ['Calcite_powder', 'Obsidian_sxtal', 'Quartz_powder']],
        'detail_1': [[[54, 55, 56, 64, 65, 66, 71, 72, 73, 74, 86, 87, 88], [0, 1, 4, 5, 6, 13, 14, 15, 18, 19, 20, 21, 24, 25, 26, 37, 38, 39], [48, 49, 50, 51, 68, 69]], ['Calcite_powder', 'Magnesite_powder', 'Quartz_powder']], # [[1, 5, 8, 9, 13, 14, 19, 25, 34, 45, 71, 74, 99], ['Calcite_powder', 'Obsidian_sxtal', 'Quartz_powder', 'Magnesium Sulfate_pellet']],
        'detail_2': [[0, 1, 5, 13, 18, 30, 41, 71, 80, 87, 88, 99], ['Calcite_powder', 'Obsidian_sxtal', 'Quartz_powder']]
    },
    '0505': {
        'HDR_1': [[[0, 5, 14, 17, 18, 19, 20, 23, 24, 25, 26, 27, 29, 34, 35, 44, 45, 58, 63, 69, 71, 82, 83, 84, 86, 88, 89, 94], [36], [96]], ['Sodium Perchlorate Hydrate_powder', 'Gypsum_powder', 'Kieserite_powder', 'Magnesium Sulfate_pellet', 'Melanterite_powder']], #[14, 20, 24, 25, 36, 69, 83]
        'detail_1': [[[1, 7, 12, 15, 21, 22, 23, 24, 28, 32, 33, 34, 36, 37, 43, 44, 55, 60, 67, 71, 72, 87, 90, 91, 92], [81]], ['Sodium Perchlorate Hydrate_powder', 'Gypsum_powder', 'Kieserite_powder', 'Magnesium Sulfate_pellet', 'Melanterite_powder']], #[1, 7, 12, 21, 24, 33]
        'survey_1296': [[], ['Sodium Perchlorate Hydrate_powder', 'Gypsum_powder', 'Kieserite_powder', 'Magnesium Sulfate_pellet', 'Melanterite_powder']]
    },
    '0513': {
        'survey_1296': [[[29, 30, 41, 42, 53, 54, 90, 175, 177, 189, 203, 234, 244, 245, 246, 252, 257, 270, 292, 319, 337, 358, 364, 365, 366, 284, 428, 429, 430, 433, 436, 437, 438, 439, 440, 496, 501, 506, 511, 512, 567, 577, 582, 587, 606, 623, 798, 799, 880, 931, 932, 936, 939, 985, 1004, 1031, 1032, 1071, 1089, 1090, 1157, 1176, 1177, 1238, 1274], [444, 445, 446, 489, 490, 491, 492, 493, 518, 519, 520, 556, 557, 558, 559, 560, 592, 593, 594, 595, 626, 629, 630, 669, 694, 695, 696, 697, 698, 742, 743, 744, 745, 767, 768, 769, 816, 817, 837, 838, 839, 889, 890, 891, 907, 908, 977, 978, 1038, 1039, 1047, 1048, 1049, 1112, 1113, 1114, 1118, 1186, 1189], [626, 627, 628, 1115, 1116, 1117, 666, 667, 668]], ['Calcite_powder', 'Gypsum_powder', 'Kieserite_powder', 'Magnesium Sulfate_pellet', 'Melanterite_powder']],
        'detail_offset_1': [[[28, 29, 30, 31, 39, 40, 59, 81, 82, 83, 97, 98], [49, 50, 51, 52, 67, 68, 69, 70, 71, 72, 73, 84, 85, 86, 87, 88, 89, 92, 93, 94, 95, 96]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet', 'Kieserite_powder']],
        'detail_offset_2': [[[4, 5, 6, 7, 12, 15, 16, 21, 22, 23, 35, 36, 38, 41, 57, 58, 59, 60, 61, 62, 72, 99], [83, 84, 85, 86, 93, 94, 95, 96, 97]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet', 'Kieserite_powder']],
        'detail_offset_3': [[[22, 23, 24, 26, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 49, 54, 56, 57, 58, 59]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet', 'Kieserite_powder']],
        'detail_offset_4': [[[5, 6, 16, 17, 42, 43, 44, 56, 57, 66, 67, 73, 78, 80, 82, 84, 96]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet', 'Kieserite_powder']]
    },
    '0600': {
        'HDR': [[[3, 7, 15, 16, 21, 23, 32, 33, 35, 39, 45, 54, 57, 66, 71], [56, 57]], ['Olivine_OLV-SC', 'Amorphous Silicate_Synthetic Mg', 'Bytownite_pellet']]
    },
    '0614': {
        'survey_1296': [[[477, 527, 528, 529, 530, 531, 532, 546, 547, 548, 549, 550, 551, 552, 598, 599, 600, 601, 602, 603, 604, 605, 617, 618, 619, 620, 621, 622, 623, 624, 671, 672, 673, 674, 675, 675, 676, 677, 678, 679, 680, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 813, 814, 815, 916, 817, 818, 819, 820, 821, 822, 823, 824, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1161, 1162, 1167, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1184, 1185, 1195, 1196, 1197, 1198, 1202, 1203, 1204, 1209, 1210, 1214, 1215, 1231, 1232, 1233, 1236, 1237, 1238, 1239, 1244, 1245, 1248, 1249, 1250, 1251, 1252, 1253, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287], [406, 531, 532, 533, 534, 546, 547, 548, 605, 606, 617, 618, 619, ]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet']],
        'HDR': [[[46, 47, 50, 52, 53, 65, 66, 67, 68, 69, 70, 71, 73, 75, 84, 85, 86, 87, 88, 93, 94, 95]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet']],
        'detail_1': [[[8, 9, 30, 31, 44, 46, 52, 69, 73, 88, 89, 90, 91]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet']],
        'detail_2': [[[1, 18, 21, 30, 33, 40, 46, 48, 57, 62, 65, 71, 83, 89, 91, 94, 95], [8, 14, 28, 61]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet']]
    },
    '0617': {
        'survey_1296': [[[446, 447, 487, 488, 489, 490, 491, 492, 514, 515, 516, 517, 518, 519, 520, 521, 556, 557, 558, 559, 560, 561, 562, 563, 563, 564, 565, 566, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 667, 668, 669, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 800, 801, 802, 803, 804, 805, 806, 807, 808, 850, 851, 852, 853, 854], [17, 18, 49, 50, 51, 52, 53, 90, 91, 92, 93, 94, 95, 119, 120, 121, 122, 123, 124, 125], [1050, 1108, 1109, 1110, 1122, 1123, 1124, 1125, 1179, 1180, 1181, 1195, 1196, 1197, 1250, 1251, 1267, 1268, 1269]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet']],
        'HDR': [[[35, 43, 44, 45, 54, 55, 56], [12, 14, 15, 16, 22, 23, 24, 25, 26, 28, 29, 36, 37, 38, 40, 48, 52, 61, 62, 64, 66, 67, 68, 72, 73, 77, 78, 81, 82, 86, 90, 91, 92, 96, 97, 98, 99], [17, 19, 31, 63]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet']],
        'detail_1': [[[1, 2, 3, 16, 17, 18, 22, 38], [4, 8, 9, 10]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet']]
    },
    '0618': {
        'survey_1296': [[[235, 236, 266, 267, 268, 269, 270, 304, 305, 306, 307, 308, 309, 310, 311, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 373, 374, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 675, 676, 677, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 816, 817, 818, 819, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 1023, 1024, 1025, 1026, 1027, 1028, 1029], [463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 535, 536, 537, 538, 539, 540, 541, 542, 543, 608, 609, 610, 611, 612, 613, 614, 615, 680, 681, 682, 683, 684, 685, 686, 687, 688, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 822, 823, 824, 825, 826, 830, 831, 832, 833, 834, 892, 893, 894, 895, 896, 904, 905, 906, 907, 908, 962, 963, 963, 964, 965, 966, 967], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 364, 365, 366, 367, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 566, 567, 568, 569, 570, 571, 572, 573, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 726, 727, 728, 729, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 857, 858, 859, 860, 869]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']],
        'detail_2': [[[40, 41, 42, 43, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, 48, 49, 50, 51, 68, 69]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']]
    },
    '0620': {
        'detail_1': [[[50, 68, 69, 70, 71, 87, 88, 89, 90, 91, 92], [29, 30, 31, 47, 48, 49, 51, 52, 53, 66, 67, 68, 72, 73, 85, 86, 93, 94, 95]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_pellet']],
        'detail_2': [[np.concatenate((range(2,17), range(23,37), range(43,52))), range(60,100)], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']]
    },
    '0712': {
        'meteorite_survey': [[908, 918], ['Apatite_sxtal', 'Enstatite_C7758', 'Lignite_PSOC-1533']]
    },
    '0860': {
        'detail_1': [[[60, 78, 79, 80, 81, 82, 83, 96, 97, 98, 99], [80, 82]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']],
        'detail_2': [[[68, 69, 70, 71, 72, 73, 86, 87, 88, 89, 90, 91, 92, 93, 94]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']]
    },
    '0861': {
        'detail_1': [[[4, 14, 15, 16, 17, 18, 21, 22, 23, 24, 36, 37, 38, 41, 42], [26, 27, 28, 29, 30, 31, 32, 33, 47, 48, 49, 50, 51, 52]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']],
        'detail_2': [[[79, 80, 81, 82, 83, 96, 97, 98, 99], [70, 71, 87, 88, 89, 90, 91, 92, 93, 94], [61, 96]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']],
        'survey_1296': [[range(399,840)], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']]
    },
    '0874': {
        'detail': [[[7, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 49], [4, 5, 6, 8, 9, 10, 11, 15, 16, 22, 23, 35, 36, 44, 45, 46, 47, 48, 50, 51]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']]
    },
    '0938': {
        'detail_1': [[[60, 61, 78, 79, 80, 81, 82, 97, 98, 99], [0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 21, 22, 24, 35, 37, 38, 39, 40, 41, 42, 44, 55, 56, 57, 63]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']],
        'detail_2': [[[0, 2, 16, 17, 18, 19, 20, 21], [20, 37, 38, 39, 40, 41, 42, 59]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']]
    },
    '0939':{
        'detail_1': [[[0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 36, 37, 38, 39], [26, 27, 28, 29, 30, 31, 32, 33, 34, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], [55, 56, 57, 58, 59, 62, 63, 64, 75, 77], [85, 86, 87, 88, 89]], ['Anhydrite_powder', 'Gypsum_powder', 'Magnesium Sulfate_VWR']]
    }
}

data['ROI_names'] = [[] for i in range(len(data['scan']))]
data['ROI_indices'] = [[] for i in range(len(data['scan']))]
data['ROI_Rspectra'] = [[] for i in range(len(data['scan']))]
data['ROI_Fspectra'] = [[] for i in range(len(data['scan']))]
data['standards'] = [[] for i in range(len(data['scan']))]
print np.shape(data['ROI_names']), np.shape(data['ROI_indices'])
x_start, x_end = (400., 1700.)

print
print "collating specified ROIs and POIs"
for i in range(0, len(data['scan'])):
    print
    print "%s: %s" % (i, data['scan_name'][i])
    sol = data['sol'][i]
    scan = data['scan'][i]
    x_coords, y_coords = data['xy_coords'][i]
    if sol in POIs.keys():
        if scan in POIs[sol].keys():
            points = POIs[sol][scan][0]
            data['standards'][i] = POIs[sol][scan][1]
            print "    specified ROIs:", len(points)
            if len(points) > 0:
                ROI_count = 0
                for i2 in range(0, len(points)):
                    if np.size(points[i2]) > 1:
                        ROI_count += 1
                        print "        ROI %s: %s spectra" % (ROI_count, len(points[i2])), np.shape(np.mean(data['y_r_sub'][i][points[i2]], axis=0))
                        data['ROI_names'][i].append("ROI %s" % ROI_count)
                        data['ROI_indices'][i].append(np.unique(points[i2]))
                        data['ROI_Rspectra'][i].append(np.mean(data['y_r_sub'][i][points[i2]], axis=0))
                        data['ROI_Fspectra'][i].append(np.mean(data['y'][i][points[i2]], axis=0))
                    else:
                        print "        single point %s:" % points[i2], np.shape(data['y_r_sub'][i][points[i2]])
                        data['ROI_names'][i].append("POI %s" % points[i2])
                        data['ROI_indices'][i].append([points[i2]])
                        data['ROI_Rspectra'][i].append(data['y_r_sub'][i][points[i2]])
                        data['ROI_Fspectra'][i].append(data['y'][i][points[i2]])
                if ROI_count > 0:
                    nonROI = [i3 for i3 in range(data['table_size'][i]) if i3 not in np.concatenate(points)]
                    print "        %s out of %s points are not assigned to an ROI/POI" % (len(nonROI), len(data['y_r_sub'][i]))
                    data['ROI_names'][i].insert(0, "non-ROI")
                    data['ROI_indices'][i].insert(0, nonROI)
                    data['ROI_Rspectra'][i].insert(0, np.mean(data['y_r_sub'][i][nonROI], axis=0))
                    data['ROI_Fspectra'][i].insert(0, np.mean(data['y'][i][nonROI], axis=0))

if Plot_POIs == True:
    print
    print "plotting ROI/POI spectra and comparing to spectral standards"
    
    for i in range(0, len(data['scan'])):
        print
        print "%s: %s" % (i, data['scan_name'][i])
        sol = data['sol'][i]
        scan = data['scan'][i]
        x_coords, y_coords = data['xy_coords'][i]
        
        # define RGB channels
        if len(F_RGB_bands['center']) == 3 and len(F_RGB_bands['width']) == 3:
            sort = np.argsort(F_RGB_bands['center'])
            F_centers = np.asarray(F_RGB_bands['center'])[sort]
            F_widths = np.asarray(F_RGB_bands['width'])[sort]
        else:
            F_centers = np.asarray([275, 305, 340])
            F_widths = np.asarray([5, 5, 5])
        if len(R_RGB_bands['center']) == 3 and len(R_RGB_bands['width']) == 3:
            sort = np.argsort(R_RGB_bands['center'])
            R_centers = np.asarray(R_RGB_bands['center'])[sort]
            R_widths = np.asarray(R_RGB_bands['width'])[sort]
        else:
            R_centers = np.asarray([840, 1020, 1090])
            R_widths = np.asarray([25, 25, 25])
        
        standards = data['standards'][i]
        points = data['ROI_indices'][i]
        point_names = data['ROI_names'][i]
        point_Rspectra = np.asarray(data['ROI_Rspectra'][i])
        point_Fspectra = np.asarray(data['ROI_Fspectra'][i])
        print point_names
        print np.shape(point_Rspectra), np.shape(point_Fspectra)
        ROIs = [name for name in point_names if name[:3] == 'ROI']
        ROI_count = len(ROIs)
        
        if ROI_count > 0:
            # generate figure showing ROI locations
            plt.figure(figsize=(12,6))
            plt.suptitle("%s ROIs" % (data['scan_name'][i]))
            ax0 = plt.subplot2grid((1,3), (0,0))
            ax1 = plt.subplot2grid((1,3), (0,1))
            ax2 = plt.subplot2grid((1,3), (0,2))
            ax0.set_title("Fluorescence\nR=%0.f, G=%0.f, B=%0.f nm" % (F_centers[0], F_centers[1], F_centers[2]))
            ax0.set_xlim(-0.5, np.size(x_coords, axis=1)-0.5)
            ax0.set_ylim(np.size(x_coords, axis=0)-0.5, -0.5)
            ax1.set_title("Raman\nR=%0.f, G=%0.f, B=%0.f cm-1" % (R_centers[0], R_centers[1], R_centers[2]))
            ax1.set_xlim(-0.5, np.size(x_coords, axis=1)-0.5)
            ax1.set_ylim(np.size(x_coords, axis=0)-0.5, -0.5)
            ax2.set_title("ROIs")
            ax2.set_xlim(-0.5, np.size(x_coords, axis=1)-0.5)
            ax2.set_ylim(np.size(x_coords, axis=0)-0.5, -0.5)
            print "xy index array:", np.shape(x_coords), np.shape(y_coords)
            print "map position array:", np.shape(data['grid_index'][i])
            RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(data['wavelength'][i], data['y'][i], data['grid_index'][i], F_centers, F_widths, norm='full', clip=0.02)
            print np.shape(RGB_map)
            im = ax0.imshow(RGB_map, aspect='equal')
            RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], R_centers, R_widths, norm='full', clip=0.02)
            print np.shape(RGB_map)
            im = ax1.imshow(RGB_map, aspect='equal')
            ROI_map = np.zeros_like(RGB_map)
            colors = [(0., 0., 1.), (1., 0., 0.), (0., 1., 0.), (1., 0., 1.), (1., 1., 0.), (0., 1., 1.)]
            num = 0
            handles = []
            text = 'single points'
            print "POIs to plot", point_names
            for i2 in range(0, len(point_names)):
                color = colors[num]
                if "ROI" in point_names[i2]:
                    text = 'ROI averages'
                    if point_names[i2] == 'non-ROI':
                        color = (0., 0., 0.)
                    else:
                        color = colors[num]
                        num += 1
                    for point in points[i2]:
                        serp_indx = np.unravel_index(point, np.shape(RGB_map[:,:,0]))
                        map_indx = data['grid_index'][i][serp_indx]
                        map_indx_2d = np.unravel_index(map_indx, np.shape(RGB_map[:,:,0]))
                        ### print map_indx, map_indx_2d
                        ROI_map[map_indx_2d] = color
                    handles.append(mpl.patches.Patch(color=color, label=point_names[i2]))
            if Plot_on_ACI == True:
                map_overlay, extent, norm = convert_map_to_spot_image(ROI_map, data['grid_index'][i], data['img_coords'][i])
                plt.imsave('%sSol-%s_%s_ROImap_img.png' % (data['figdir'][i], data['sol'][i], data['scan'][i]), map_overlay)
            im = ax2.imshow(ROI_map, aspect='equal')
            plt.imsave('%sSol-%s_%s_ROImap.png' % (data['figdir'][i], data['sol'][i], data['scan'][i]), np.repeat(np.repeat(ROI_map, 5, axis=0), 5, axis=1))
            ax2.legend(handles=handles)
            plt.savefig('%sSol-%s_%s_ROImaps.png' % (data['figdir'][i], data['sol'][i], data['scan'][i]), dpi=300)
            plt.show()
            
            for i2 in range(0, len(point_names)):
                if point_names[i2][:3] == 'ROI':
                    plt.figure(figsize=(8,4))
                    # ax2: raman region
                    ax2 = plt.subplot(111)
                    ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
                    ax2.set_ylabel("Intensity (counts)")
                    ax2.set_xlim(400, 4000)
                    ### ax2.axvline(wavelength2shift(252.93), c='k', linestyle=':')
                    # add wavelength ticks to top of ax2
                    ax2_top = ax2.twiny()
                    x_range = ax2.get_xlim()
                    major_labels, major_locations, minor_locations = get_wavelength_ticks(x_range)
                    ax2_top.set_xticks(major_locations)
                    ax2_top.set_xticklabels(major_labels)
                    ax2_top.xaxis.set_minor_locator(plt.FixedLocator(minor_locations))
                    ax2_top.set_xlabel("Wavelength (nm)")
                    ax2.axvspan(960, 1060, color='k', linewidth=0., alpha=0.1)
                    # find maxima for Raman and Fluor regions
                    R_min = find_min(data['raman_shift'][i], point_Rspectra[i2], inset_limits[0], inset_limits[1])[1]
                    R_max = find_max(data['raman_shift'][i], point_Rspectra[i2], inset_limits[0], inset_limits[1])[1]
                            
                    # plot mean Raman spectrum
                    x_start, x_end = ax2.get_xlim()
                    sliced = np.where((x_start <= data['raman_shift'][i]) & (data['raman_shift'][i] < x_end))
                    x_slice = data['raman_shift'][i][sliced]
                    y_slice = np.mean(data['y_r_sub'][i], axis=0)[sliced]
                    ax2.plot(x_slice, y_slice, 'k', label='scan mean')
                    text = 'scan mean: %s points' % (data['table_size'][i])
                    # plot ROI mean spectrum
                    y_slice = point_Rspectra[i2][sliced]
                    ax2.plot(x_slice, y_slice, c='b', label="%s mean" % point_names[i2])
                    text += '\n%s mean: %s points' % (point_names[i2], len(points[i2]))
                            
                    # adjust y limits
                    temp_min = find_min(data['raman_shift'][i], point_Rspectra[i2], inset_limits[0], inset_limits[1])[1]
                    temp_max = find_max(data['raman_shift'][i], point_Rspectra[i2], inset_limits[0], inset_limits[1])[1]
                    if temp_min < R_min:
                        R_min = temp_min
                    if temp_max > R_max:
                        R_max = temp_max
                    ax2.set_ylim(R_min-0.2*(R_max-R_min), R_max+0.2*(R_max-R_min))
                    ax2.legend(loc=2)
                    ax2.minorticks_on()
                    plt.tight_layout(rect=[0, 0., 1, 0.95])
                    plt.suptitle("%s ROI-%s" % (data['scan_name'][i], i2))
                    plt.figtext(0.93, 0.7, text, ha='right', va='top')
                    # save figure
                    plt.savefig('%sSol-%s_%s_%s_R.png' % (data['figdir'][i], sol, scan, point_names[i2]), dpi=300)
                    plt.savefig('%sSol-%s_%s_%s_R.svg' % (data['figdir'][i], sol, scan, point_names[i2]), dpi=300)
                    plt.show()
                
            print
            print "   generating POI figure..."
                
            x_start, x_end = (750, 2000)
            length = len(points) + len([standard for standard in standards if standard in BB_R_data['standard'] or standard in MOB_R_data['standard']])
            if Hydration == True:
                width = 9.
            else:
                width = 5.
            print
            print "    spectra to plot: %s SHERLOC %s and %s standards" % (len(points), text, length-len(points))
            plt.figure(figsize=(width, np.amax([6., 0.5+0.5*length])))
            if Hydration == True:
                plt.suptitle(data['scan_name'][i])
                ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
                ax2 = plt.subplot2grid((1, 3), (0, 2))
                ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
                ax1.set_ylabel("Normalised Intensity")
                ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
                ax1.set_xlim(x_start, x_end)
                ax1.set_yticks([])
                ax2.set_xlim(3000, 4000)
                ax2.set_yticks([])
            else:
                plt.suptitle(data['scan_name'][i])
                ax1 = plt.subplot(111)
                ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
                ax1.set_ylabel("Normalised Intensity")
                ax1.set_xlim(x_start, x_end)
                ax1.set_yticks([])
            cumulative_offset = 0.
            count = 0
            # plot SRLC POI spectra
            print "        SRLC:"
            for i2 in range(0, len(point_names)):
                if np.size(point_Rspectra[i2], axis=0) == 1:
                    y_temp = np.ravel(point_rspectra[i2])
                else:
                    y_temp = point_Rspectra[i2]
                print "            ", i2, np.shape(data['raman_shift'][i]), np.shape(y_temp), x_start, x_end
                y_max = find_max(data['raman_shift'][i], y_temp, x_start, x_end)[1]
                ax1.plot(data['raman_shift'][i], y_temp/y_max+cumulative_offset, 'k', label=point_names[i2])
                if Hydration == True:
                    H_max = find_max(data['raman_shift'][i], y_temp, 3000, 4000)[1]
                    ax2.plot(data['raman_shift'][i], y_temp/H_max+cumulative_offset, 'k', label=point_names[i2])
                ax1.text(x_start+0.98*(x_end-x_start), cumulative_offset+0.3, point_names[i2], ha='right')
                cumulative_offset += 1.2
            # plot terrestrial standards
            if BB == True:
                print "        BB standards:"
                for i2 in range(0, len(BB_R_data['standard'])):
                    print "            %s" % BB_R_data['standard'][i2]
                    # add Brassboard spectra
                    sample = BB_R_data['standard'][i2]
                    standard = sample.split("_")[0]
                    if include_instrument_names == True:
                        standard += " BB"
                    color = Color_list[count % len(Color_list)]
                    x_temp = BB_R_data['raman_shift'][i2]
                    y_temp = BB_R_data['y_av_sub_norm'][i2]
                    std_temp = BB_R_data['y_std_sub_norm'][i2]
                    y_max = find_max(x_temp, y_temp, 800, 4000)[1]
                    # plot average Raman spectrum
                    ax1.plot(x_temp, y_temp/y_max+cumulative_offset, color, alpha=0.5, label=standard)
                    if Hydration == True:
                        H_max = find_max(x_temp, y_temp, 3000, 4000)[1]
                        ax2.plot(x_temp, y_temp/H_max+cumulative_offset, color, label=standard.split("_")[0])
                    ax1.text(x_start+0.98*(x_end-x_start), cumulative_offset+0.3, standard.split("_")[0], ha='right')
                    # plot vertical line at standard's strongest peak position
                    if len(BB_R_data['fitted_peaks'][i2]) > 0:
                        peaks = BB_R_data['fitted_peaks'][i2].transpose()
                        peak_mask = peaks[np.where((x_start < peaks[:,0]) & (peaks[:,0] < x_end) & (peaks[:,2] > 10))]
                        if np.size(peak_mask) > 0:
                            max_peak = peak_mask[np.argmax(peak_mask[:,1])]
                            y_max = max_peak[0]
                            print "                  max peak %s: center = %0.1f cm-1, amplitude = %0.0f cts, sigma/width = %0.1f cm-1, round = %0.1f cm-1" % (i2, max_peak[0], max_peak[1], max_peak[2], max_peak[3])
                    else:
                        y_max = find_max(BB_R_data['raman_shift'][i2], BB_R_data['y_av_sub_norm'][i2], x_start, x_end)[0]
                    ax1.axvline(y_max, c=color, ls=':', alpha=0.5)
                    cumulative_offset += 1.2
                    count += 1
            if MOB == True:
                # add MOBIUS spectra
                print "        MOBIUS standards:"
                for i2 in range(0, len(MOB_R_data['standard'])):
                    print "            %s" % MOB_R_data['standard'][i2]
                    sample = MOB_R_data['standard'][i2]
                    standard = sample.split("_")[0]
                    if include_instrument_names == True:
                        standard += " MOB"
                    color = Color_list[count % len(Color_list)]
                    x_temp = MOB_R_data['raman_shift'][i2]
                    y_temp = MOB_R_data['y_av_sub_norm'][i2]
                    std_temp = MOB_R_data['y_std_sub_norm'][i2]
                    y_max = find_max(x_temp, y_temp, 800, 4000)[1]
                    # plot average Raman spectrum
                    ax1.plot(x_temp, y_temp/y_max+cumulative_offset, color, alpha=0.5, label=standard)
                    if Hydration == True:
                        H_max = find_max(x_temp, y_temp, 3000, 4000)[1]
                        ax2.plot(x_temp, y_temp/H_max+cumulative_offset, color, label=standard.split("_")[0])
                    ax1.text(x_start+0.98*(x_end-x_start), cumulative_offset+0.3, standard.split("_")[0], ha='right')
                    # plot vertical line at standard's strongest peak position
                    if len(MOB_R_data['fitted_peaks'][i2]) > 0:
                        peaks = MOB_R_data['fitted_peaks'][i2].transpose()
                        peak_mask = peaks[np.where((x_start < peaks[:,0]) & (peaks[:,0] < x_end) & (peaks[:,2] > 10))]
                        if np.size(peak_mask) > 0:
                            max_peak = peak_mask[np.argmax(peak_mask[:,1])]
                            y_max = max_peak[0]
                            print "              max peak %s: center = %0.1f cm-1, amplitude = %0.0f cts, sigma/width = %0.1f cm-1, round = %0.1f cm-1" % (i2, max_peak[0], max_peak[1], max_peak[2], max_peak[3])
                    else:
                        y_max = find_max(MOB_R_data['raman_shift'][i2], MOB_R_data['y_av_sub_norm'][i2], x_start, x_end)[0]
                    ax1.axvline(y_max, c=color, ls=':', alpha=0.5)
                    cumulative_offset += 1.2
                    count += 1
            # add reference lines
            if x_start < 680 and 680 < x_end:
                ax1.axvline(695, c='k', ls=':', alpha=0.5)
                ax1.text(695+20, cumulative_offset+0.1, 'laser')
            if x_start < 1550 and 1550 < x_end:
                ax1.axvline(1550, c='k', ls=':', alpha=0.5)
                ax1.text(1550+20, cumulative_offset+0.1, 'O$_2$')
            if x_start < 2330 and 2330 < x_end:
                ax1.axvline(2330, c='k', ls=':', alpha=0.5)
                ax1.text(2330+20, cumulative_offset+0.1, 'N$_2$')
            # set limits
            ax1.set_ylim(-0.5, cumulative_offset+0.3)
            ax1.minorticks_on()
            if Hydration == True:
                ax2.set_ylim(-0.5, cumulative_offset+0.3)
                ax2.minorticks_on()
            # finish figure
            plt.tight_layout(rect=[0, 0., 1, 0.93])
            plt.savefig('%sSol-%s/%s_POI-comparison.png' % (Figure_dir, sol, scan), dpi=300)
            plt.savefig('%sSol-%s/%s_POI-comparison.svg' % (Figure_dir, sol, scan), dpi=300)
            plt.show()
                
            if Fit_ROIs == True and len(points) > 0:
                    print
                    print "    attempting to fit %s POIs" % (data['scan_name'][i])
                    # fit each ROI spectrum separately
                    x_start, x_end = (800, 2000)
                    
                    # manually adjust the maximum sigma value for peak fitting, as necessary
                    max_sigma = 60.
                    
                    # define thresholds for peak detection
                    rel_threshold = 0.1
                    noise_threshold = 5.
                    
                    for i2 in range(0, len(points)):
                        # trim to fitting region
                        x_slice = data['raman_shift'][i][np.where((x_start <= data['raman_shift'][i]) & (data['raman_shift'][i] <= x_end))]
                        y_slice = point_Rspectra[i2][np.where((x_start <= data['raman_shift'][i]) & (data['raman_shift'][i] <= x_end))]
                        noise = np.std(point_Rspectra[i2][np.where((2000 <= data['raman_shift'][i]) & (data['raman_shift'][i] <= 2100))])
                        
                        # find peaks above relative intensity threshold
                        maxima = find_maxima(x_slice, y_slice, 5, rel_threshold)
                        maxima_pass = [[],[]]
                        if len(maxima[0]) > 0:
                            # check if peaks are also above noise threshold
                            SNR = maxima[1] / noise
                            for i3 in range(0, len(maxima[0])):
                                if SNR[i3] > noise_threshold:
                                    # if SNR also above threshold, add to pass list
                                    maxima_pass[0].append(np.round(maxima[0][i3]))
                                    maxima_pass[1].append(maxima[1][i3])
                        maxima_pass = np.asarray(maxima_pass)
                        print "    peaks detected:", ["%0.f" % peak for peak in maxima_pass[0]]
                        # add any manually defined peaks
                        if len(Manual_fit_peaks) > 0:
                            print "    additional user-specified peaks:", len(Manual_fit_peaks)
                            for i2 in range(0, len(Manual_fit_peaks)):
                                loc = Manual_fit_peaks[i2]
                                if len(maxima_pass[0]) == 0 and x_start < loc and loc < x_end:
                                    y_max = y_temp[np.argmin(np.absolute(x_temp - loc))]
                                    maxima_pass = np.asarray([[loc], [y_max]])
                                else:
                                    if np.amin(np.absolute(maxima_pass[0]-loc)) > 20  and x_start < loc and loc < x_end:
                                        indx = np.argmax(maxima_pass[0] > loc)
                                        y_max = y_temp[np.argmin(np.absolute(x_temp - loc))]
                                        maxima_pass = np.insert(maxima_pass, indx, np.asarray([loc, y_max]), axis=1)
                        print np.shape(maxima_pass)
                        # tidy up detected peak array
                        sort_i = np.argsort(maxima_pass[0])
                        maxima_pass = maxima_pass[:,sort_i]
                        print
                        print "    %s: %s detected peaks" % (point_names[i2], len(maxima_pass[0])), maxima_pass[0]
                        
                        # attempt fit
                        peaks = maxima_pass[0]
                        if Fit_function in ['FD', 'Fermi-Dirac']:
                            # then use Fermi-Dirac fitting
                            function = 'FD'
                            text = 'Fermi-Dirac'
                            print "            Fermi-Dirac fitting"
                            fit_output, fit_curve = multiFD_fit_script(x_slice, y_slice, peaks, window=30.)
                        else:
                            # then default to Gaussian fitting
                            function = 'G'
                            text = 'Gaussian'
                            print "            Gaussian fitting"
                            fit_output, fit_curve = multiG_fit_script(x_slice, y_slice, peaks, window=30., max_sigma=max_sigma)
                            
                        # plot fit results
                        plt.figure(figsize=(8,6))
                        # ax1: fitted spectrum
                        ax1 = plt.subplot2grid((4,5), (0,0), colspan=4, rowspan=3)
                        ax1.set_title("%s %s\n%0.f-%0.f cm$^{-1}$ Peak Fitting" % (data['scan_name'][i], point_names[i2], x_start, x_end))
                        ax1.set_ylabel("Average Intensity (counts)")
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
                        # plot input data and residuals
                        ax1.plot(x_slice, y_slice, 'k')
                        ax2.plot(x_slice, y_slice-fit_curve, 'k')
                        ax3.hist(y_slice-fit_curve, range=(res_min, res_max), bins=20, orientation='horizontal', color='k')
                        x_curve = np.linspace(x_start, x_end, 5*len(x_slice))
                        for i3 in range(0, len(peaks)):
                            # plot and report peak positions
                            plt.figtext(0.78, 0.93-0.08*i3, "%s: %.1f (SNR=%0.1f)" % (i3+1, fit_output.params["center_%s" % i3].value, fit_output.params["amplitude_%s" % i3].value))
                            ax1.axvline(fit_output.params["center_%s" % i3], color='k', linestyle=':')
                            pos = fit_output.params["center_%s" % i3]
                            error = fit_output.params["center_%s" % i3].stderr
                            print "        peak %s:" % i3
                            print "            position: %0.1f cm / %0.2f nm / %0.2f pixels" % (pos, shift2wavelength(pos), sherloc_reverse_calibration(shift2wavelength(pos)))
                            if error == None:
                                print "            error could not be evaluated"
                            elif error >= 100:
                                print "            error exceeds 100 cm-1! Please check fit is accurate"
                            else:
                                print "            error: %0.2f cm / %0.3f nm / %0.3f pixels" % (error, shift2wavelength(pos+error)-shift2wavelength(pos), sherloc_reverse_calibration(shift2wavelength(pos+error))-sherloc_reverse_calibration(shift2wavelength(pos)))
                            # generate plot for peak function
                            params = lmfit.Parameters()
                            params.add('amplitude', value=fit_output.params["amplitude_%s" % i3])
                            params.add('center', value=fit_output.params["center_%s" % i3])
                            params.add('gradient', value=fit_output.params["gradient"])
                            params.add('intercept', value=fit_output.params["intercept"])
                            if function == 'FD':
                                plt.figtext(0.78, 0.9-0.08*i3, " FWHM %s: %.1f" % (i3+1, 2.*fit_output.params["width_%s" % i3].value))
                                params.add('width', value=fit_output.params["width_%s" % i3])
                                params.add('round', value=fit_output.params["round_%s" % i3])
                                peak_curve = FD_curve(x_curve, params)
                            else:
                                plt.figtext(0.78, 0.9-0.08*i3, " FWHM %s: %.1f" % (i3+1, 2.355*fit_output.params["sigma_%s" % i3].value))
                                params.add('sigma', value=fit_output.params["sigma_%s" % i3])
                                peak_curve = G_curve(x_curve, params)
                            ax1.plot(x_curve, peak_curve, 'b:')
                        if len(peaks) > 1:
                            # plot total fitted curve (if appropriate)
                            if function == 'FD':
                                total_curve = multiFD_curve(x_curve, fit_output.params, peaks)
                            else:
                                total_curve = multiG_curve(x_curve, fit_output.params, peaks)
                            ax1.plot(x_curve, total_curve, 'b--')
                        else:
                            total_curve = peak_curve
                        # finish fitted region figure
                        y_max = np.amax(y_slice)
                        y_max = np.amax(y_slice)
                        ax1.set_xlim(x_start, x_end)
                        ax1.set_ylim(np.amin([-0.2*y_max, np.amin(y_slice), np.amin(total_curve)]), 1.2*y_max)
                        # save figure to sample figure directory
                        plt.savefig('%sSol-%s_%s_%s_fit.png' % (data['figdir'][i], sol, scan, point_names[i2]), dpi=300)
                        plt.savefig('%sSol-%s_%s_%s_fit.svg' % (data['figdir'][i], sol, scan, point_names[i2]), dpi=300)
                        plt.show()

                
"""
# ==================================================
# plot Raman vs Fluorescence intensity
# ==================================================
"""

# define bands for intensity comparison in each scan, by sol number
Default_intensity_correlations = {
    'Default': [['F', 305, 5], ['F', 325, 5], ['F', 340, 5]],
    '0059': {
        'meteorite': [['F', 340, 5], ['R', 975, 25], ['R', 1080, 25], ['R', 1610, 25]]
    },
    '0181': {
        'meteorite_1': [['F', 350, 5], ['F', 340, 5], ['F', 300, 5], ['R', 975, 25], ['R', 1080, 25], ['R', 1610, 25]],
        'meteorite_2': [['F', 350, 5], ['F', 340, 5], ['F', 300, 5], ['R', 975, 25], ['R', 1080, 25], ['R', 1610, 25]]
    },
    '0368': {
        'meteorite_1': [['F', 350, 5], ['F', 340, 5], ['R', 975, 25], ['R', 1080, 25], ['R', 1610, 25]]
    },
    '0545': {
        'MarsMeteorite_1': [['F', 350, 5], ['F', 340, 5], ['R', 975, 25], ['R', 1080, 25], ['R', 1610, 25]]
    },
    '0704': {
        'meteorite': [['F', 350, 5], ['F', 340, 5], ['R', 975, 25], ['R', 1080, 25], ['R', 1610, 25]]
    },
    '0712': {
        'meteorite_detail_1': [['F', 350, 5], ['F', 340, 5], ['R', 975, 25], ['R', 1080, 25], ['R', 1610, 25]],
        'meteorite_survey': [['F', 350, 5], ['F', 340, 5], ['R', 975, 25], ['R', 1080, 25], ['R', 1610, 25]]
    },
    '0935': {
        'MarsMeteorite': [['F', 350, 5], ['F', 340, 5], ['R', 975, 25], ['R', 1080, 25], ['R', 1610, 25]]
    }
}   # bands are defined as follows: 'scan name': [['R' or 'F', center, half-width], ['R' or 'F', center, half-width]...]

log_scale = True
linregress_ROI_points_only = False

if Intensity_correlation == True:
    print
    print "doing intensity correlation..."
    for i in range(0, len(data['scan'])):
        sol = data['sol'][i]
        scan = data['scan'][i]
        
        # determine which set of band positions to correlate
        if sol in Default_intensity_correlations.keys():
            if scan in Default_intensity_correlations[sol].keys():
                band_pairs = combinations(Default_intensity_correlations[sol][scan], 2)
            else:
                band_pairs = combinations(Default_intensity_correlations['Default'], 2)
        else:
            band_pairs = combinations(Default_intensity_correlations['Default'], 2)
            
        # proceed with correlation
        for band1, band2 in band_pairs:
            print band1, band2
            # generate band 1 map
            if band1[0] == 'R':
                print "band 1: Raman %0.f +/- %0.f cm-1" % (band1[1], band1[2])
                map1, min1, max1 = map_spectral_band(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], band1[1], band1[2], clipping=True, clip=0.04, vmin=0.)
                xlabel = "%0.f cm$^{-1}$" % band1[1]
                print "plotting %s range: %0.f - %0.f counts" % (xlabel, min1, max1)
            else:
                print "band 1: Fluor %0.f +/- %0.f nm" % (band1[1], band1[2])
                map1, min1, max1 = map_spectral_band(data['wavelength'][i], data['y'][i], data['grid_index'][i], band1[1], band1[2], clipping=True, clip=0.04, vmin=0.)
                xlabel = "%0.f nm" % band1[1]
                print "plotting %s range: %0.f - %0.f counts" % (xlabel, min1, max1)
            # generate band 2 map
            if band2[0] == 'R':
                print "band 2: Raman %0.f +/- %0.f cm-1" % (band2[1], band2[2])
                map2, min2, max2 = map_spectral_band(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], band2[1], band2[2], clipping=True, clip=0.04, vmin=0.)
                ylabel = "%0.f cm$^{-1}$" % band2[1]
                print "plotting %s range: %0.f - %0.f counts" % (ylabel, min2, max2)
            else:
                print "band 2: Fluor %0.f +/- %0.f nm" % (band2[1], band2[2])
                map2, min2, max2 = map_spectral_band(data['wavelength'][i], data['y'][i], data['grid_index'][i], band2[1], band2[2], clipping=True, clip=0.04, vmin=0.)
                ylabel = "%0.f nm" % band2[1]
                print "plotting %s range: %0.f - %0.f counts" % (ylabel, min2, max2)
            
            # generate ratio map and figure out appropriate axes for plotting
            ratio_map = map1/map2
            print "%s/%s ratio range:" % (xlabel, ylabel), np.amin(ratio_map), np.amax(ratio_map)
            filter_map = np.logical_and(map1 > 0, map2 > 0)
            print "    %s/%s points have >0 signal" % (np.count_nonzero(filter_map), np.size(filter_map))
            
            # check for pre-defined ROIs and assign colours
            colors = [(0., 0., 1.), (1., 0., 0.), (0., 1., 0.), (1., 0., 1.), (1., 1., 0.), (0., 1., 1.)]
            num = 0
            labels = []
            handles = []
            print len(data['ROI_names'][i]), len(data['ROI_indices'][i])
            ROI_map = np.full((np.size(ratio_map, axis=1), np.size(ratio_map, axis=0), 3), 0.)
            if np.any(filter_map) == True:
                ROI_map[~filter_map] = (0.2, 0.2, 0.2)
            if len(data['ROI_names'][i]) > 0 and len(data['ROI_indices'][i]) > 0:
                # add ROIs to map
                for roi_name, roi_indices in zip(data['ROI_names'][i], data['ROI_indices'][i]):
                    if roi_name == "non-ROI":
                        color = (0., 0., 0.)
                    else:
                        color = colors[num]
                        num += 1
                        for point in roi_indices:
                            serp_indx = np.unravel_index(point, np.shape(ROI_map[:,:,0]))
                            map_indx = data['grid_index'][i][serp_indx]
                            map_indx_2d = np.unravel_index(map_indx, np.shape(ROI_map[:,:,0]))
                            ### print map_indx, map_indx_2d
                            ROI_map[map_indx_2d] = color
                    labels.append(mpl.lines.Line2D([], [], linewidth=0., marker='o', color=color, label=roi_name))
                    handles.append(mpl.patches.Patch(color=color, label=roi_name))
            print "%s ROIs:" % len(data['ROI_names'][i]), data['ROI_names'][i]
            
            # plot maps against each other
            plt.figure(figsize=(12,6))
            # ax1: map of band 1, ax2: map of band 2, ax3: map of band 1/band 2 ratio, ax0: map of ROIs, ax4: scatter plot
            ax1 = plt.subplot2grid((2,4), (0,0))
            ax2 = plt.subplot2grid((2,4), (0,1))
            ax3 = plt.subplot2grid((2,4), (1,0))
            ax0 = plt.subplot2grid((2,4), (1,1))
            ax4 = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=2)
            ax0.set_title("ROIs")
            ax0.imshow(ROI_map, aspect='equal')
            text = ""
            if band1[0] == 'R':
                label1 = "%0.f cm$^{-1}$" % band1[1]
                ax1.set_title("%0.f-%0.f cm$^{-1}$" % (band1[1]-band1[2], band1[1]+band1[2]))
                text += "%0.fcm-v-" % band1[1]
            else:
                label1 = "%0.f nm" % band1[1]
                ax1.set_title("%0.f-%0.f nm" % (band1[1]-band1[2], band1[1]+band1[2]))
                text += "%0.fnm-v-" % band1[1]
            if band2[0] == 'R':
                label2 = "%0.f cm$^{-1}$" % band2[1]
                ax2.set_title("%0.f-%0.f cm$^{-1}$" % (band2[1]-band2[2], band2[1]+band2[2]))
                text += "%0.fcm" % band2[1]
            else:
                label2 = "%0.f nm" % band2[1]
                ax2.set_title("%0.f-%0.f nm" % (band2[1]-band2[2], band2[1]+band2[2]))
                text += "%0.fnm" % band2[1]
            # determine x,y limits
            xmin = np.amin(map1[filter_map])
            xmax = np.amax(map1[filter_map])
            ymin = np.amin(map2[filter_map])
            ymax = np.amax(map2[filter_map])
            print "x range: %0.f - %0.f counts" % (xmin, xmax)
            print "y range: %0.f - %0.f counts" % (ymin, ymax)
            if log_scale == True:
                ax4.set_xscale('log')
                ax4.set_yscale('log')
                ax4.set_xlim(10**np.floor(np.log10(np.amax([1,xmin]))), 10**np.ceil(np.log10(xmax)))
                ax4.set_ylim(10**np.floor(np.log10(np.amax([1,ymin]))), 10**np.ceil(np.log10(ymax)))
            else:
                ax4.set_xlim(xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin))
                ax4.set_ylim(ymin-0.2*(ymax-ymin), ymax+0.2*(ymax-ymin))
            ax3.set_title("%0.f/%0.f Intensity Ratio" % (band1[1], band2[1]))
            ax4.set_title("%s vs %s" % (label1, label2))
            ax4.set_xlabel("%s Intensity (counts)" % label1)
            ax4.set_ylabel("%s Intensity (counts)" % label2)
            imR = ax1.imshow(map1, aspect='equal', vmin=0, vmax=max1, cmap=Cmap)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(imR, cax=cax)
            imF = ax2.imshow(map2, aspect='equal', vmin=0, vmax=max2, cmap=Cmap)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(imF, cax=cax)
            # plot R/F ratio map
            imRF = ax3.imshow(ratio_map, aspect='equal', vmin=0., vmax=np.sort(np.ravel(ratio_map))[int(0.95*np.size(ratio_map))], cmap=Cmap)
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(imRF, cax=cax)
            # plot R/F intensities as 2D scatter
            ROI_map_flat = np.reshape(ROI_map, newshape=(np.size(np.ravel(map1)),3))
            print np.shape(ROI_map), np.shape(ROI_map_flat)
            ax4.scatter(np.ravel(map1), np.ravel(map2), c=ROI_map_flat, alpha=0.2)
            if linregress_ROI_points_only == True:
                mask = np.sum(ROI_map, axis=2) > 0.
                print np.shape(mask)
                x_points = np.ravel(map1[mask])
                y_points = np.ravel(map2[mask])
            else:
                x_points = np.ravel(map1)
                y_points = np.ravel(map2)
            result = stats.linregress(x_points, y_points)
            xmin, xmax = ax4.get_xlim()
            print "x scale: %0.f - %0.f" % (xmin, xmax)
            print "y scale: %0.f - %0.f" % (ymin, ymax)
            x_temp = np.linspace(xmin, xmax, 1000)
            y_temp = result.slope * x_temp + result.intercept
            ax4.plot(x_temp, y_temp, 'k:', alpha=0.5)
            plt.figtext(0.66, 0.87, 'slope: %0.1E +/- %0.3E\nR$2$: %0.4f' % (result.slope, result.stderr, result.rvalue**2))
            if len(labels) > 0:
                ax4.legend(handles=labels, loc=2)
            plt.tight_layout()
            plt.savefig('%sSol-%s_%s_%s-v-%s_%s.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], band1[0], band2[0], text), dpi=300)
            plt.show()
                    
            # plot correlation graph on its own
            plt.figure(figsize=(6,6))
            plt.title("%s\n%s vs %s" % (data['scan_name'][i], label1, label2))
            plt.xlabel("%s Intensity (counts)" % label1)
            plt.ylabel("%s Intensity (counts)" % label2)
            if log_scale == True:
                plt.xscale('log')
                plt.yscale('log')
                plt.xlim(10**np.floor(np.log10(np.amax([1,xmin]))), 10**np.ceil(np.log10(xmax)))
                plt.ylim(10**np.floor(np.log10(np.amax([1,ymin]))), 10**np.ceil(np.log10(ymax)))
            else:
                plt.xlim(xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin))
                plt.ylim(ymin-0.2*(ymax-ymin), ymax+0.2*(ymax-ymin))
            plt.scatter(np.ravel(map1), np.ravel(map2), c=ROI_map_flat, alpha=0.2)
            plt.plot(x_temp, y_temp, 'k:', alpha=0.5)
            plt.figtext(0.45, 0.85, 'slope: %0.1E +/- %0.3E' % (result.slope, result.stderr))
            plt.figtext(0.45, 0.8, 'R$^2$: %0.4f' % (result.rvalue**2))
            if len(labels) > 1:
                plt.legend(handles=labels, loc=2)
            plt.tight_layout()
            plt.savefig('%sSol-%s_%s_%s-v-%s_%s_scatter.png' % (data['figdir'][i], data['sol'][i], data['scan'][i], band1[0], band2[0], text), dpi=300)
            plt.show()
    
    plot_SNR = False
    SNR_threshold = 5.
    log_scale = True
    target_correlations = [['F', 350, 5], ['R', 1080, 25], ['R', 1610, 50]]
    group_by = 'ppp'

    if group_by == 'ppp':
        norm = mpl.colors.LogNorm(vmin=np.amin(data['laser_pulses']), vmax=np.amax(data['laser_pulses']))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=Cmap)
        cmap.set_array([])
    elif group_by == 'sol':
        norm = mpl.colors.Normalize(vmin=np.amin(data['sol']), vmax=np.amax(data['sol']))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=Cmap)
        cmap.set_array([])

    # plot all data points for each target
    for target in np.unique(data['target']):
            # determine which set of band positions to correlate
            band_pairs = combinations(target_correlations, 2)
            for band1, band2 in band_pairs:
                # plot correlation scatter
                plt.figure(figsize=(8,6))
                all_x = []
                all_y = []
                count = 0
                for i in range(0, len(data['scan'])):
                    if data['target'][i] == target:
                        sol = data['sol'][i]
                        scan = data['scan'][i]
                        if band1[0] == 'R':
                            map1, min1, max1 = map_spectral_band(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], band1[1], band1[2], clipping=True, clip=0.04, vmin=0.)
                            map1 = np.ravel(map1)
                            if plot_SNR == True:
                                noise = np.std(data['y'][i][:,np.ravel(np.where((265 - band1[2] <= data['raman_shift'][i]) & (data['raman_shift'][i] <= 265 + band1[2])))], axis=1)
                                map1 /= noise
                                print np.amax(map1)
                            xlabel = "%0.f cm$^{-1}$" % band1[1]
                            text = "%0.fcm-v-" % band1[1]
                        else:
                            map1, min1, max1 = map_spectral_band(data['wavelength'][i], data['y'][i], data['grid_index'][i], band1[1], band1[2], clipping=True, clip=0.04, vmin=0.)
                            map1 = np.ravel(map1)
                            if plot_SNR == True:
                                noise = np.std(data['y'][i][:,np.ravel(np.where((265 - band1[2] <= data['wavelength'][i]) & (data['wavelength'][i] <= 265 + band1[2])))], axis=1)
                                map1 /= noise
                                print np.amax(map1)
                            xlabel = "%0.f nm" % band1[1]
                            text = "%0.fnm-v-" % band1[1]
                        # generate band 2 map
                        if band2[0] == 'R':
                            map2, min2, max2 = map_spectral_band(data['raman_shift'][i], data['y_r_sub'][i], data['grid_index'][i], band2[1], band2[2], clipping=True, clip=0.04, vmin=0.)
                            map2 = np.ravel(map2)
                            if plot_SNR == True:
                                noise = np.std(data['y_r_sub'][i][:,np.ravel(np.where((2100 - band2[2] <= data['raman_shift'][i]) & (data['raman_shift'][i] <= 2100 + band2[2])))], axis=1)
                                map2 /= noise
                                print np.amax(map2)
                            ylabel = "%0.f cm$^{-1}$" % band2[1]
                            text += "%0.fcm" % band2[1]
                        else:
                            map2, min2, max2 = map_spectral_band(data['wavelength'][i], data['y'][i], data['grid_index'][i], band2[1], band2[2], clipping=True, clip=0.04, vmin=0.)
                            map2 = np.ravel(map2)
                            if plot_SNR == True:
                                noise = np.std(data['y'][i][:,np.ravel(np.where((265 - band2[2] <= data['wavelength'][i]) & (data['wavelength'][i] <= 265 + band2[2])))], axis=1)
                                map2 /= noise
                                print np.amax(map2)
                            ylabel = "%0.f nm" % band2[1]
                            text += "%0.fnm" % band2[1]
                        if plot_SNR == True:
                            filter_map = np.logical_and(map1 > SNR_threshold, map2 > SNR_threshold)
                        else:
                            filter_map = np.full_like(map1, True, dtype=bool)
                        print "%s / %s above threshold" % (np.count_nonzero(filter_map), np.size(map1))
                        if group_by == 'ppp':
                            color = cmap.to_rgba(data['laser_pulses'][i])
                            label = "%d ppp" % data['laser_pulses'][i]
                        elif group_by == 'sol':
                            color = cmap.to_rgba(data['sol'][i])
                            label = "Sol-%s" % (sol)
                        else:
                            color = Color_list[count % (len(Color_list))]
                            label = "Sol-%s %s" % (sol, scan)
                        plt.scatter(map1[filter_map], map2[filter_map], c=color, label=label, alpha=0.2)
                        all_x += list(map1[filter_map])
                        all_y += list(map2[filter_map])
                        count += 1
                if plot_SNR == True:
                    plt.title("%s by scan (SNR > %s)\n%s vs %s" % (target, xlabel, ylabel, SNR_threshold))
                    plt.xlabel("%s SNR" % xlabel)
                    plt.ylabel("%s SNR" % ylabel)
                else:
                    plt.title("%s by scan\n%s vs %s" % (target, xlabel, ylabel))
                    plt.xlabel("%s Intensity (counts)" % xlabel)
                    plt.ylabel("%s Intensity (counts)" % ylabel)
                xmin = np.amin(all_x)
                xmax = np.amax(all_x)
                ymin = np.amin(all_y)
                ymax = np.amax(all_y)
                if log_scale == True:
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlim(10**np.floor(np.log10(np.amax([1,xmin]))), 10**np.ceil(np.log10(xmax)))
                    plt.ylim(10**np.floor(np.log10(np.amax([1,ymin]))), 10**np.ceil(np.log10(ymax)))
                else:
                    plt.xlim(np.amax([0,xmin]), xmax+0.2*(xmax-xmin))
                    plt.ylim(np.amax([0,ymin]), ymax+0.2*(ymax-ymin))
                result = stats.linregress(all_x, all_y)
                x_temp = np.linspace(xmin, xmax, 1000)
                y_temp = result.slope * x_temp + result.intercept
                plt.plot(x_temp, y_temp, 'k:', alpha=0.5)
                plt.figtext(0.65, 0.85, 'slope: %0.1E +/- %0.3E' % (result.slope, result.stderr))
                plt.figtext(0.65, 0.8, 'R$^2$: %0.4f' % (result.rvalue**2))
                plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                plt.savefig('./figures by target/%s/%s_%s-v-%s_%s_scatter.png' % (target, target, band1[0], band2[0], text), dpi=300)
                plt.show()
            
"""
==================================================
save processed data
==================================================
"""

print
print "saving processed data to output files"

for i in range(0, len(data['scan'])):
    sol = data['sol'][i]
    scan = data['scan'][i]
    print
    print data['scan_name'][i]
    save_name = "Sol-%s_%s" % (sol, scan)
    
    # save recalibrated, raw composite spectra
    save_data = np.vstack((data['wavelength'][i], data['raman_shift'][i], data['y'][i]))
    print "    composite spectrum, raw:", np.shape(save_data)
    header = ["Wavelength (nm)", "Raman shift (cm-1)"]
    for i2 in range(0, data['table_size'][i]):
        header.append("Spectrum %s" % i2)
    np.savetxt("%s%s_composite_recal.txt" % (data['outdir'][i], save_name), save_data.transpose(), delimiter='\t', header="\t".join(header))
    np.savetxt("%s%s_composite_recal.csv" % (data['outdir'][i], save_name), save_data.transpose(), delimiter=',', header=",".join(header))
    
    # save recalibrated, raw region 1 spectra
    save_data = np.vstack((data['wavelength'][i], data['raman_shift'][i], data['y_r'][i]))
    print "    reg1 spectrum, raw:", np.shape(save_data)
    header = ["Wavelength (nm)", "Raman shift (cm-1)"]
    for i2 in range(0, data['table_size'][i]):
        header.append("Spectrum %s" % i2)
    np.savetxt("%s%s_reg1_recal.txt" % (data['outdir'][i], save_name), save_data.transpose(), delimiter='\t', header="\t".join(header))
    np.savetxt("%s%s_reg1_recal.csv" % (data['outdir'][i], save_name), save_data.transpose(), delimiter=',', header=",".join(header))
    
    # save processed region 1 data
    save_data = np.vstack((data['wavelength'][i], data['raman_shift'][i], data['y_r_sub'][i]))
    print "    reg1 spectrum, baselined:", np.shape(save_data)
    header = ["Wavelength (nm)", "Raman shift (cm-1)"]
    for i2 in range(0, data['table_size'][i]):
        header.append("Spectrum %s" % i2)
    np.savetxt("%s%s_reg1_processed.txt" % (data['outdir'][i], save_name), save_data.transpose(), delimiter='\t', header="\t".join(header))
    np.savetxt("%s%s_reg1_processed.csv" % (data['outdir'][i], save_name), save_data.transpose(), delimiter=',', header=",".join(header))
    
print
print "DONE"
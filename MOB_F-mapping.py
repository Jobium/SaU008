"""
==================================================

This script processes Fluorescence spectral maps from the MOBIUS instrument. Processing includes:
    1) spectral recalibration and atmospheric correction
    2) baseline fitting and subtraction
    3) region and point of interest averaging
    
This script then generates figures, including:
    1) intensity maps at user-defined wavelengths
    2) comparison of user-defined ROI average spectra to reference materials
    3) comparison of user-defined POI spectra to reference materials
    4) intensity correlation maps

==================================================
"""

import os
import glob
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import lmfit as lmfit

from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from itertools import combinations
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==================================================
# initial variables

# define the directories for input/output:
Data_dir = './data/MOBIUS/Fluor/'
Output_dir = './output/MOBIUS/'
Figure_dir = './figures/MOBIUS/'
ref_data_dir = '/refs/MOBIUS/output/Fluor/*/'

# define the sample and scan to process, use '*' to run all
Group = "SAU-008"       # name of group
Sample = 'SAU-008'      # name of sample
Scan = '*'              # name/type of scan
Pulses = '*'            # number of laser pulses per point
Spacing = '*'           # spacing, in um
Laser_correction_factor = 1.

# processes to run
Plot_ROIs = True            # plot Region of Interest average spectra
Plot_POIs = True            # plot Region of Interest average spectra
Compare = True              # plot Region of Interest average spectra
Plot_intensity_ratios = True    # plot comparison maps for intensity ratio pairs

# define Fluorescence bands for individual mapping, by position and half-width
Fluor_bands = [275, 300, 320, 340, 350, 375, 400]
Fluor_widths = [5, 5, 5, 5, 5, 5, 5]

# define Fluorescence bands for RGB mapping, by position and half-width
RGB_bands = [350, 340, 300]
RGB_widths = [5, 5, 5]

# specify reference standards for comparison (leave list empty to skip)
ref_samples = []
ref_pulses = 25
ref_current = 15

# define x limits for Fluorescence inset figures
inset_limits = (255, 280)

# plot +/-1 st. dev.? True/False
Plot_std = False

# specify map normalisation: 'full' = all bands normalised to same min/max; 'separate': each band normalised individually; 'None': no normalisation
band_normalisation = 'all'

# color parameters for plotting
Cmap = 'viridis'
Color_list =  ['r', 'g', 'b', 'm', 'y', 'tab:gray', 'c', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:purple']
Marker_list = ['o', 's', 'v', '^', 'D', '*']

# ==================================================
# functions

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

def serpentine(table_size, shape='square'):
    # function for indexing a 1D array into a 2D array according to a serpentine (back and forth) pattern
    indices = np.arange(table_size)
    # determine shape of 2D array
    if len(shape) == 2:
        dims = shape
    else:
        # assume a square array
        dims = (int(np.sqrt(table_size)), int(np.sqrt(table_size)))
    print dims
    # create 2D array
    index_array = np.reshape(indices, dims)
    print index_array
    # reverse every other row
    index_array[1::2,:] = np.flip(index_array, axis=1)[1::2,:]
    print index_array
    return index_array

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

def baseline(x, y, x_list, base='poly', max_order=15, window=25, hydration=False, find_minima=True, fixed_ends=True, debug=False, plot=False, name=None):
    global groups, samples, scans, sample_figdirs, i
    # calculate baseline and subtract it
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
            ax1.set_title("%s:\nBaseline-Corrected Raman Spectrum" % (str(name)))
        else:
            ax1.set_title("%s:\nBaseline-Corrected Raman Spectrum" % (scans[i]))
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
        plt.savefig("%s%s_av_base.png" % (sample_figdirs[i], scans[i]), dpi=300)
        plt.show()
    return y_B

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
# functions for handling spectral map generation

def map_spectral_band(x_values, spectra, position_array, band_position, band_width, function='sum', vmin=None, vmax=None, clipping=False, clip=0.02):
    # function for turning a list of spectra into a 2D array of band values that can be plotted as an image
    global Cmin, Cmax
    print "input:", np.shape(x_values), np.shape(spectra), np.shape(position_array)
    print "band compression function:", function
    if clipping == True:
        # check if input clipping value makes sense and correct if necessary
        if clip >= 50:
            print "input clipping value exceeds limit, defaulting to 2%%"
            clip = 0.02
        elif clip >= 0.5:
            print "input clipping value greater than 0.5, converting to percentage"
            clip = float(clip)/100.
        elif clip < 1./len(spectra):
            print "input clipping value equates to less than 1 spectrum, defaulting to 0"
            clip = 0.
    else:
        clip = 0.
    # determine which x_values fall within the target band
    band_pass = np.absolute(x_values - band_position) <= band_width
    print np.shape(band_pass)
    if np.any(band_pass == True) == False:
        print "spectral band outside of data range"
    # reindex spectra into a 2D map according to the position array
    spectral_map = spectra[position_array]
    print "spectral map:", np.shape(spectral_map)
    # get the mean band value
    print "masked spectral map:", np.shape(spectral_map[:, :, band_pass])
    if function == 'sum':
        band_map = np.sum(spectral_map[:, :, band_pass], axis=2)
    else:
        band_map = np.mean(spectral_map[:, :, band_pass], axis=2)
    print "band value map:", np.shape(band_map)
    if vmin == None or vmax == None:
        # determine min/max values for color mapping
        if clipping == True and clip > 0.:
            indx = int(np.ceil(clip*len(spectra)))
            print "    clip percentage: %0.1f%%" % (100.*indx/len(spectra))
            print "    min clipping indx:", indx, "of", len(spectra)
            print "    max clipping indx:", len(spectra)-indx, "of", len(spectra)
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
    print "nans:", np.any(np.isnan(band_map))
    print "output data range: %0.1f to %0.1f" % (np.amin(band_map), np.amax(band_map))
    print "vmin, vmax range: %0.1f to %0.1f" % (vmin, vmax)
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

def clip_this_map(absolute_map, floor=0.02, ceiling=0.98, norm='full'):
    # function for clipping and normalising the values of a 2D map (RGB or single-channel) for image generation
    print
    if absolute_map.ndim > 3:
        print "input map is RGB"
    elif absolute_map.ndim == 2:
        print "input map is single-channel"
    if absolute_map.ndim > 2:
        # handling RGB map
        R_map = absolute_map[:,:,0]
        G_map = absolute_map[:,:,1]
        B_map = absolute_map[:,:,2]
        print "R channel range: %0.1f to %0.1f" % (np.amin(R_map), np.amax(R_map))
        print "G channel range: %0.1f to %0.1f" % (np.amin(G_map), np.amax(G_map))
        print "B channel range: %0.1f to %0.1f" % (np.amin(B_map), np.amax(B_map))
        RGB_hists = np.stack((np.sort(R_map, axis=None), np.sort(G_map, axis=None), np.sort(B_map, axis=None)))
        print "sorted values:",np.shape(RGB_hists)
        RGB_limits = np.zeros((3,2))
        if norm == 'full':
            # normalise all three RGB channels to overall min/max
            floor_indx = int(np.ceil(floor*3.*np.size(absolute_map)))
            print "    floor: %0.1f%%" % (100.*floor_indx/(3.*np.size(absolute_map)))
            print "    floor indx:", floor_indx, "of", 3.*np.size(absolute_map)
            ceil_indx = int(np.floor(ceiling*3.*np.size(absolute_map)))
            print "    ceiling: %0.1f%%" % (100.*ceil_indx/(3.*np.size(absolute_map)))
            print "    ceiling indx:", ceil_indx, "of", 3.*np.size(absolute_map)
            channel_sort = np.sort(RGB_hists, axis=None)
            channel_min = channel_sort[floor_indx]
            channel_max = channel_sort[ceil_indx]
            print "min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            R_map = (R_map - channel_min)/(channel_max - channel_min)
            G_map = (G_map - channel_min)/(channel_max - channel_min)
            B_map = (B_map - channel_min)/(channel_max - channel_min)
            RGB_limits = np.tile(np.asarray([channel_min, channel_max]), (3,1))
        else:
            # normalise and clip intensity values in each RGB channel individually
            floor_indx = int(np.ceil(floor*np.size(absolute_map)))
            print "    floor: %0.1f%%" % (100.*floor_indx/(np.size(absolute_map)))
            print "    floor indx:", floor_indx, "of", np.size(absolute_map)
            ceil_indx = int(np.floor(ceiling*3.*np.size(absolute_map)))
            print "    ceiling: %0.1f%%" % (100.*ceil_indx/(np.size(absolute_map)))
            print "    ceiling indx:", ceil_indx, "of", np.size(absolute_map)
            # handle R channel
            channel_sort = RGB_hists[0]
            channel_min = channel_sort[floor_indx]
            channel_max = channel_sort[ceil_indx]
            print "R min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            R_map = (R_map - channel_min)/(channel_max - channel_min)
            RGB_limits[0] = np.asarray([channel_min, channel_max])
            # handle G channel
            channel_sort = RGB_hists[1]
            channel_min = channel_sort[floor_indx]
            channel_max = channel_sort[ceil_indx]
            print "G min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            G_map = (G_map - channel_min)/(channel_max - channel_min)
            RGB_limits[1] = np.asarray([channel_min, channel_max])
            # handle B channel
            channel_sort = RGB_hists[2]
            channel_min = channel_sort[floor_indx]
            channel_max = channel_sort[ceil_indx]
            print "B min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            B_map = (B_map - channel_min)/(channel_max - channel_min)
            RGB_limits[2] = np.asarray([channel_min, channel_max])
        # create RGB map array
        RGB_map = np.stack((R_map, G_map, B_map), axis=-1)
        print np.shape(RGB_map)
        if full_norm == True or channel_norm == True:
            # clean up any values outside clipping range
            RGB_map[RGB_map > 1.] = 1.
            RGB_map[RGB_map < 0.] = 0.
        print "output data range: %0.1f to %0.1f" % (np.amin(RGB_map), np.amax(RGB_map))
        print "output:", np.shape(RGB_map), np.shape(RGB_hists), np.shape(RGB_limits)
        return RGB_map, RGB_hists, RGB_limits
    else:
        # handling single-channel map
        hist = np.sort(absolute_map, axis=None)
        print "sorted values:", np.shape(hist)
        # determine upper, lower limits for clipping
        floor_indx = int(np.ceil(floor*3.*np.size(absolute_map)))
        print "    floor: %0.1f%%" % (100.*floor_indx/(3.*np.size(absolute_map)))
        print "    floor indx:", floor_indx, "of", 3.*np.size(absolute_map)
        ceil_indx = int(np.floor(ceiling*3.*np.size(absolute_map)))
        print "    ceiling: %0.1f%%" % (100.*ceil_indx/(3.*np.size(absolute_map)))
        print "    ceiling indx:", ceil_indx, "of", 3.*np.size(absolute_map)
        channel_min = hist[floor_indx]
        channel_max = hist[ceil_indx]
        limits = np.asarray([channel_min, channel_max])
        print "clipping min/max: %0.1f to %0.1f" % (channel_min, channel_max)
        # normalise map based on clipping limits
        clipped_map = (absolute_map - channel_min)/(channel_max - channel_min)
        print "clipped map:", np.shape(clipped_map)
        print "data type:", clipped_map.dtype
        # clean up any values outside clipping range
        clipped_map[clipped_map > 1.] = 1.
        clipped_map[clipped_map < 0.] = 0.
        print "nans:", np.any(np.isnan(clipped_map))
        print "output data range: %0.1f to %0.1f" % (np.amin(clipped_map), np.amax(clipped_map))
        return clipped_map, hist, limits
    
def get_clip_limits(absolute_map, clip=0.02, norm='full'):
    # function for determining the upper and lower clipping limits for a 2D spectral map (RGB or single-channel)
    print
    if absolute_map.ndim == 3:
        print "input map is RGB"
    elif absolute_map.ndim == 2:
        print "input map is single-channel"
    # check if input clipping value makes sense and correct if necessary
    if clip >= 50:
        print "input clipping value exceeds limit, defaulting to 2%%"
        clip = 0.02
    elif clip >= 0.5:
        print "input clipping value greater than 0.5, converting to fraction"
        clip = float(clip)/100.
    elif clip < 1./np.size(absolute_map):
        print "input clipping value rounds to 0, defaulting to minimum"
        clip = 1./np.size(absolute_map)
    if absolute_map.ndim > 2:
        # handling RGB map
        RGB_limits = np.zeros((3,2))
        if norm == 'full':
            # normalise all three RGB channels to overall min/max
            indx = int(np.ceil(clip*3.*len(spectra)))
            print "    clip percentage: %0.1f%%" % (100.*indx/(3.*np.size(absolute_map)))
            print "    min clipping indx:", indx, "of", 3.*np.size(absolute_map)
            print "    max clipping indx:", len(spectra)-indx, "of", 3.*np.size(absolute_map)
            channel_sort = np.sort(RGB_hists, axis=None)
            channel_min = channel_sort[indx]
            channel_max = channel_sort[-indx-1]
            print "min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            R_map = (R_map - channel_min)/(channel_max - channel_min)
            G_map = (G_map - channel_min)/(channel_max - channel_min)
            B_map = (B_map - channel_min)/(channel_max - channel_min)
            RGB_limits = np.tile(np.asarray([channel_min, channel_max]), (3,1))
        else:
            indx = int(np.ceil(clip*np.size(absolute_map)))
            print "    clip percentage: %0.1f%%" % (100.*indx/np.size(absolute_map))
            print "    min clipping indx:", indx, "of", np.size(absolute_map)
            print "    max clipping indx:", np.size(absolute_map)-indx, "of", np.size(absolute_map)
            # handle R channel
            channel_sort = RGB_hists[0]
            channel_min = channel_sort[indx]
            channel_max = channel_sort[-indx-1]
            print "R min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            RGB_limits[0] = np.asarray([channel_min, channel_max])
            # handle G channel
            channel_sort = RGB_hists[1]
            channel_min = channel_sort[indx]
            channel_max = channel_sort[-indx-1]
            print "G min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            RGB_limits[1] = np.asarray([channel_min, channel_max])
            # handle B channel
            channel_sort = RGB_hists[2]
            channel_min = channel_sort[indx]
            channel_max = channel_sort[-indx-1]
            print "B min/max: %0.1f to %0.1f" % (channel_min, channel_max)
            RGB_limits[2] = np.asarray([channel_min, channel_max])
        print "output:", np.shape(RGB_limits)
        return RGB_limits
    else:
        # handling single-channel map
        hist = np.sort(absolute_map, axis=None)
        print "sorted values:", np.shape(hist)
        # determine upper, lower limits for clipping
        indx = int(np.ceil(clip*len(hist)))
        print "    clip percentage: %0.1f%%" % (100.*indx/len(hist))
        print "    min clipping indx:", indx, "of", len(hist)
        print "    max clipping indx:", len(hist)-indx, "of", len(hist)
        channel_min = hist[indx]
        channel_max = hist[-indx-1]
        print "min/max values: %0.1f/%0.1f" % (channel_min, channel_max)
        limits = np.asarray([channel_min, channel_max])
        print "output:", np.shape(limits)
        return limits

def area_mean(x_values, spectra, position_array, xy_coords, target, shape='circle', width=5):
    # find the mean spectrum for an area around a targeted point expressed in x,y space
    print "input:", np.shape(x_values), np.shape(spectra), np.shape(position_array)
    # check values
    width = float(width)
    # check coordinate arrays
    x_coords, y_coords = (xy_coords)
    # check target location (tranposed coords)
    x_loc, y_loc = target
    print "x,y coord arrays:", np.shape(x_coords), np.shape(y_coords)
    print "x range: %0.1f - %0.1f" % (np.amin(x_coords), np.amax(x_coords))
    print "y range: %0.1f - %0.1f" % (np.amin(y_coords), np.amax(y_coords))
    print "x,y coords of target:", x_loc, y_loc
    ### print x_coords
    ### print y_coords
    if x_loc < np.amin(x_coords):
        print "    target x coordinate outside map area!"
        x_loc = np.amin(x_coords)
    elif x_loc > np.amax(x_coords):
        print "    target x coordinate outside map area!"
        x_loc = np.amax(x_coords)
    if y_loc < np.amin(y_coords):
        print "    target y coordinate outside map area!"
        y_loc = np.amin(y_coords)
    elif y_loc > np.amax(y_coords):
        print "    target y coordinate outside map area!"
        y_loc = np.amax(y_coords)
    # mask the map to exclude points outside the targeted area
    if shape == 'square':
        mask = np.where(np.absolute(x_coords - x_loc)**2 + np.absolute(y_coords - y_loc)**2 <= width**2)
        print np.shape(mask)
    else:
        mask = np.where(np.absolute(x_coords - x_loc)**2 + np.absolute(y_coords - y_loc)**2 <= width**2)
        print np.shape(mask)
    target_spectra = spectra[position_array][mask]
    print "    spectra within ROI:", np.size(target_spectra, axis=0)
    mean_spectrum = np.mean(target_spectra, axis=0)
    print np.shape(mean_spectrum)
    return mean_spectrum

"""
# ==================================================
# import data
# ==================================================
"""

# dict for reference standards
ref_data = {
    'standards': [],
    'splits': [],
    'wavelength': [],
    'raman_shift': [],
    'y_av': [],
    'std': [],
    'y_av_norm': [],
    'std_norm': [],
    'y_av_sub': [],
    'std_sub': [],
    'y_av_sub_norm': [],
    'std_sub_norm': [],
    'R_pulses': [],
    'R_current': [],
    'R_dose': [],
    'F_pulses': [],
    'F_current': [],
    'F_dose': [],
    'fitted_peaks': []
}

# ==================================================
# attempt to import each spectrum file

Groups = sorted(glob.glob('%s%s/' % (Data_dir, Group)))

print Groups

sample_figdirs = []
sample_outdirs = []
groups = []
samples = []
scans = []
table_sizes = []
datetimes = []
laser_pulses = []
laser_currents = []
laser_energies = []
map_positions = []
xy_coords = []
wavelength = []
raman_shift = []
y = []

num = 0
for group in Groups:
    group = group.split("/")[-2]
    print
    print group
    # create figure/output folders for this group
    figdir = "%s%s/" % (Figure_dir, group)
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    outdir = "%s%s/" % (Output_dir, group)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # find individual data folders for this group
    folders = sorted(glob.glob('%s%s/%s_%s*_%sppp_%sum/' % (Data_dir, group, Sample, Scan, Pulses, Spacing)))
    print len(folders)
    for folder in folders:
        folder_name = folder.split("/")[-2]
        sample_name = folder_name.split("_")[0]
        print
        print "attempting to import data for scan %s..." % folder_name
        print "    ", folder
        while True:
            try:
                spec_dir = glob.glob("%s*_NC.txt" % folder)[0]
                spec = np.genfromtxt(spec_dir, delimiter="\t")
                print "    spectra array:", np.shape(spec)
                if np.size(spec, axis=0) in [1024,1025,2048,2049]:
                    print "        transposing..."
                    spec = np.transpose(spec)
                if np.size(spec, axis=1) in [1025,2049]:
                    print "        removing extraneous column..."
                    spec = np.delete(spec, -1, axis=1)
                if np.size(spec, axis=1) == 2048:
                    print "        reducing array to 1024 points..."
                    x_temp = (spec[0,::2] + spec[0,1::2])/2
                    y_temp = spec[1:,::2] + spec[1:,1::2]
                else:
                    x_temp = spec[0]
                    y_temp = spec[1:]
                print "        x,y arrays:", np.shape(x_temp), np.shape(y_temp)
                print "    extracting timestamp..."
                original_dir = glob.glob("%s* Fluor_*.txt" % folder)[0]
                with open(original_dir) as fp:
                    for i, line in enumerate(fp):
                        if i == 33:
                            datestr = line.split("\t")[-1][:-1]
                        elif i > 33:
                            break
                datetime_temp = datetime.datetime.strptime(datestr, "%d.%m.%Y %H:%M:%S")
                print "        timestamp:", datetime_temp
                print "        importing point position array..."
                positions_dir = glob.glob("%sworking_files/points_array.csv" % folder)
                ### print "            ", position_dir
                positions_temp = np.genfromtxt(positions_dir[0], delimiter=',', dtype='int')[1:,1:]
                print "            position array", np.shape(positions_temp)
                grid = np.mgrid[0:np.size(positions_temp, axis=0), 0:np.size(positions_temp, axis=1)]
                print "            grid:", np.shape(grid)
                if len(np.ravel(positions_temp)) != np.size(y_temp, axis=0):
                    print "            point array size does not match y array size, trimming to match y!"
                    max_rows = np.size(y_temp, axis=0)/np.size(positions_temp, axis=1)
                    print "                complete rows found:", max_rows
                    max_spec = np.size(y_temp, axis=1)*max_rows
                    y_temp = y_temp[:max_spec]
                    positions_temp = positions_temp[:max_rows,:]
                    grid = np.mgrid[0:np.size(positions_temp, axis=0), 0:np.size(positions_temp, axis=1)]
                    print "                trimmed spectra:", np.shape(y_temp)
                    print "                trimmed positions:", np.shape(positions_temp)
                spacing_temp = 100.
                for temp in folder_name.split("_"):
                    if len(temp) > 2:
                        if temp[-2:] == 'um':
                            spacing_temp = float(temp[:-2])
                coords_temp = (spacing_temp/1000.) * np.roll(grid, 1, axis=0)
                print "        dimensions of scan: %0.1f x %0.1f mm" % (coords_temp[0,0,-1], coords_temp[1,-1,0])
                print "        importing laser energy array..."
                laser_dir = glob.glob("%sworking_files/laser_array.csv" % folder)
                ### print "            ", laser_dir
                laser_temp = Laser_correction_factor * np.genfromtxt(laser_dir[0], delimiter=',')/1000
                print "            laser energy array:", np.shape(laser_temp)
                laser_energies_temp = np.ravel(laser_temp)[:np.size(positions_temp)]
                print "            matched to position array:", np.shape(laser_energies_temp)
                pulse_temp = 1200.
                current_temp = 15.
                for temp in folder_name.split("_"):
                    if len(temp) > 3:
                        if temp[-3:] == 'ppp':
                            pulse_temp = float(temp[:-3])
                dose_temp = np.mean(laser_energies_temp)
                print "        laser: %0.0f ppp, %0.0f A, %0.2f uJ" % (pulse_temp, current_temp, dose_temp)
                print "        mean pulse energy: %0.2f uJ" % (dose_temp/pulse_temp)
                print "    all data found, adding to arrays"
                groups.append(group)
                samples.append(sample_name)
                scans.append(folder_name)
                wavelength.append(x_temp)
                raman_shift.append(wavelength2shift(x_temp))
                y.append(y_temp)
                table_sizes.append(np.size(y_temp, axis=0))
                datetimes.append(datetime_temp)
                laser_pulses.append(pulse_temp)
                laser_currents.append(current_temp)
                laser_energies.append(laser_energies_temp)
                map_positions.append(positions_temp)
                xy_coords.append(coords_temp)
                # make sample folders
                print "    making output folders"
                figdir = '%s%s/%s/%s/' % (Figure_dir, group, sample_name, folder_name)
                if not os.path.exists(figdir):
                    os.makedirs(figdir)
                outdir = '%s%s/%s/%s/' % (Output_dir, group, sample_name, folder_name)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                print "        ", figdir
                sample_figdirs.append(figdir)
                sample_outdirs.append(outdir)
                print "    success!"
                break
            except IOError:
                print "    no file found!"
                break
            except:
                print "    something went wrong!"
                break
            
datetimes = np.asarray(datetimes)
laser_pulses = np.asarray(laser_pulses)
laser_currents = np.asarray(laser_currents)
laser_energies = np.asarray(laser_energies)
wavelength = np.asarray(wavelength)
raman_shift = np.asarray(raman_shift)
y = np.asarray(y)

print
print "sample array:", np.shape(samples)
print "pulse, current arrays:", np.shape(laser_pulses), np.shape(laser_currents)
print "laser energies array:", np.shape(laser_energies)
print laser_pulses
print laser_currents

print
print
print "assembling group dict, generating output directories..."
print "standards found:"

print samples

unique_samples = []

for i in range(0, len(samples)):
    sample = samples[i]
    if sample not in unique_samples:
        unique_samples.append(sample)
        
print
print "unique samples:", len(unique_samples)

print unique_samples
        
print
print "sample array:", np.shape(samples)
print "sample figure directory array:", np.shape(sample_figdirs)

"""
==================================================
begin data processing
==================================================
"""

print
print "finding average spectrum and standard deviation..."

y_av = []
y_av_norm = []
std = []
std_norm = []
y_smooth = []

for i in range(0, len(samples)):
    y_av.append(np.mean(y[i], axis=0))
    y_smooth.append(smooth_f(y_av[i], 5, 3))
    std.append(np.std(y[i], axis=0))
    plt.title("Average Spectrum:\n%s" % scans[i])
    plt.fill_between(wavelength[i], y_av[i]+std[i], y_av[i]-std[i], color='k', alpha=0.2)
    plt.plot(wavelength[i], y_av[i], 'k')
    plt.plot(wavelength[i], y_smooth[i], 'r')
    y_max = find_max(wavelength[i], y_av[i], 255, 410)[1]
    y_av_norm.append(y_av[i]/y_max)
    std_norm.append(std[i]/y_max)
    plt.ylim(-0.2*y_max,1.2*y_max)
    plt.xlim(np.amin(wavelength[i]),np.amax(wavelength[i]))
    plt.show()

"""
==================================================
generate Fluorescence maps
==================================================
"""

colorbar_factor = 8
histcolors = ['r', 'g', 'b', 'k', 'c', 'tab:orange', 'tab:purple', 'tab:gray', 'tab:brown', 'tab:pink']

print
print "generating Fluorescence intensity maps..."

for i in range(0, len(scans)):
    print
    print scans[i]
    plt.figure(figsize=(6, 4))
    plt.title("Individual Fluorescence Spectra: %s" % scans[i])
    plt.xlim(inset_limits[0], inset_limits[1])
    y_max = 0.
    for i2 in range(0, len(y[i])):
        plt.plot(wavelength[i], y[i][i2])
        if find_max(wavelength[i], y[i][i2], inset_limits[0], inset_limits[1])[1] > y_max:
            y_max = find_max(wavelength[i], y[i][i2], inset_limits[0], inset_limits[1])[1]
    plt.ylim(-0.2*y_max, 1.2*y_max)
    plt.show()
    x_coords, y_coords = (xy_coords[i])
    bands = np.asarray(Fluor_bands)
    widths = np.asarray(Fluor_widths)
    if len(Fluor_bands) > 0 and len(Fluor_widths) == len(Fluor_bands):
        # add predefined bands to list
        bands, sorti = np.unique(np.concatenate((bands, Fluor_bands)), return_index=True)
        widths = np.concatenate((widths, Fluor_widths))[sorti]
    print "bands for mapping:", bands
    colors = []
    best_R = []
    for i2 in range(0, len(bands)):
        band = bands[i2]
        width = widths[i2]
        print
        print "band %0.f: %0.f to %0.f" % (band, band-width, band+width)
        aspect = float(np.size(map_positions[i], axis=0)) / float(np.size(map_positions[i], axis=1))
        print "map aspect ratio: %0.f" % aspect
        plt.figure(figsize=(10,4*aspect))
        ax1 = plt.subplot2grid((1,2), (0,0))
        ax2 = plt.subplot2grid((1,2), (0,1), colspan=1)
        ax1.set_title("%0.f-%0.f nm Intensity Map" % (band-width, band+width))
        ax2.set_title(scans[i])
        x_start, x_end = (255, 410)
        ax2.set_xlim(x_start, x_end)
        band_map, vmin, vmax = map_spectral_band(wavelength[i], y[i], map_positions[i], band, width, clipping=True, clip=0.04, vmin=0.)
        plt.imsave("%s%s_%0.fnm_map.png" % (sample_figdirs[i], scans[i], band), np.repeat(np.repeat(band_map, 5, axis=0), 5, axis=1), vmin=vmin, vmax=vmax, cmap=Cmap)
        im = ax1.imshow(band_map, extent=(0., np.amax(x_coords), np.amax(y_coords), 0.), vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # find hottest point and add to list for plotting separately
        print
        print "    hottest point:"
        point = np.argmax(band_map.flatten())
        print "        point", point
        serp_indx = np.unravel_index(point, np.shape(band_map))
        print "        serpentine index:", serp_indx
        map_indx = map_positions[i][serp_indx]
        print "        map index:", map_indx
        print "        integrated intensity: %0.f counts" % (band_map.flatten()[point])
        best_R.append(y[i][map_indx])

        # find top 3 points for plotting in this figure
        y_max = find_max(wavelength[i], np.mean(y[i], axis=0), x_start, x_end)[1]
        ax2.plot(wavelength[i], np.mean(y[i], axis=0), 'k', label='mean')
        ax2.axvspan(band-width, band+width, color='k', alpha=0.1)
        hottest_points = np.flip(np.argsort(band_map.flatten())[-3:])
        print
        print "    3 hottest points:", hottest_points
        for point in hottest_points:
            print "    point", point
            serp_indx = np.unravel_index(point, np.shape(band_map))
            print "        serpentine index:", serp_indx
            map_indx = map_positions[i][serp_indx]
            print "        map index:", map_indx
            map_indx_2d = np.unravel_index(map_indx, np.shape(band_map))
            print "           2D:", map_indx_2d
            point_x = x_coords[map_indx_2d]
            point_y = y_coords[map_indx_2d]
            print "        x,y coords:", point_x, point_y
            ax1.text(point_x, point_y, "%0.f" % (map_indx), color='w')
            ax2.plot(wavelength[i], y[i][map_indx], label='point %s' % (map_indx))
            print "        integrated intensity: %0.f counts" % (band_map.flatten()[point])
            if find_max(wavelength[i], y[i][map_indx], x_start, x_end)[1] > y_max:
                y_max = find_max(wavelength[i], y[i][map_indx], x_start, x_end)[1]
        ax2.legend()
        ax2.set_ylim(-0.2*y_max, 1.2*y_max)
        ax2.minorticks_on()
        plt.tight_layout()
        plt.savefig('%s%s_%0.fnm.png' % (sample_figdirs[i], scans[i], band), dpi=300)
        plt.show()
        print '    saving figure to %s%s_%0.fnm.png' % (sample_figdirs[i], scans[i], band)
            
    aspect = float(np.size(map_positions[i], axis=0)) / float(np.size(map_positions[i], axis=1))
    print "map x/y aspect ratio:", (1./aspect)
    plt.figure(figsize=(4.*len(bands), 4.*aspect+2.))
    plt.suptitle(scans[i])
        
    clipping = False
    clip = 0.04
    vmins = []
    vmaxs = []
    if band_normalisation == 'fixed' and Cmin != None and Cmax != None:
        print "using pre-assigned upper/lower intensity limits for map normalisation"
        norm = 'fixed'
        vmins = np.full((len(bands)), 0.)
        vmaxs = np.full((len(bands)), Cmax)
    elif band_normalisation in ['all', 'all_bands', 'full', 'norm']:
        print "normalising maps using %0.2f and %0.2f percentiles of all band values" % (clip, 1.-clip)
        norm = 'fixed'
        band_maps = []
        for i2 in range(0, len(bands)):
            temp, vmin, vmax = map_spectral_band(wavelength[i], y[i], map_positions[i], bands[i2], widths[i2])
            band_maps.append(temp)
        temp = np.sort(np.ravel(band_maps))
        print np.shape(temp)
        indx = int(np.ceil(clip*len(temp)))
        print "    clip percentage: %0.1f%%" % (100.*indx/len(temp))
        print "    min clipping indx:", indx, "of", len(temp), "value:", temp[indx]
        print "    max clipping indx:", len(temp)-indx, "of", len(temp), "value:", temp[-indx]
        vmins = np.full((len(bands)), 0.)
        vmaxs = np.full((len(bands)), temp[-indx])
    else:
        print "defaulting to individual band normalisation, using %0.2f - %0.2f clipping" % (clip, 1.-clip)
        norm = 'full'
        vmins = np.full((len(bands)), None)
        vmaxs = np.full((len(bands)), None)
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
            
    plt.figure(figsize=(1+2*len(bands), 4))
    for i2 in range(0, len(bands)):
        band = bands[i2]
        width = widths[i2]
        print "    band %0.f, from %0.f to %0.f nm" % (band, band-width, band+width)
        if i2 == 0:
            ax1 = plt.subplot2grid((3, colorbar_factor*len(bands)+1), (0, colorbar_factor*i2), colspan=colorbar_factor)
            ax1.set_title("%0.f-%0.f nm" % (band-width, band+width))
            ax1.hist(np.ravel(band_maps[i2]), bins=bins, range=(vmins[i2], vmaxs[i2]), color=histcolors[i2])
        else:
            plt.subplot2grid((3, colorbar_factor*len(bands)+1), (0, colorbar_factor*i2), colspan=colorbar_factor, sharex=ax1)
            plt.title("%0.f-%0.f nm" % (band-width, band+width))
            plt.hist(np.ravel(band_maps[i2]), bins=bins, range=(vmins[i2], vmaxs[i2]), color=histcolors[i2])
            plt.yticks([])
        plt.subplot2grid((3, colorbar_factor*len(bands)+1), (1, colorbar_factor*i2), colspan=colorbar_factor, rowspan=2)
        plt.imshow(band_maps[i2], extent=(0., np.amax(x_coords), np.amax(y_coords), 0.), vmin=vmins[i2], vmax=vmaxs[i2], cmap=Cmap)
        if i2 != 0:
            plt.yticks([])
    ax2 = plt.subplot2grid((3, colorbar_factor*len(bands)+1), (1, colorbar_factor*len(bands)), rowspan=2)
    plt.colorbar(ax=ax2, shrink=0.9, label="intensity (counts)")
    plt.savefig("%s%s_Fluor_maps.png" % (sample_figdirs[i], scans[i]), dpi=300)
    plt.show()
    
    # generate fully normalised map and save to file
    sort = np.argsort(RGB_bands)[::-1]
    band_centers = np.asarray(RGB_bands)[sort]
    band_widths = np.asarray(RGB_widths)[sort]
    print '%s%s_R=%0.f_G=%0.f_B=%0.f_map_abs.png' % (sample_figdirs[i], scans[i], band_centers[0], band_centers[1], band_centers[2])
    RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(wavelength[i], y[i], map_positions[i], band_centers, band_widths, norm='full', clip=0.02)
    plt.imsave('%s%s_R=%0.f_G=%0.f_B=%0.f_map_abs.png' % (sample_figdirs[i], scans[i], band_centers[0], band_centers[1], band_centers[2]), np.repeat(np.repeat(RGB_map, 5, axis=0), 5, axis=1))
    # generate channel-normalised map and save to file
    RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(wavelength[i], y[i], map_positions[i], band_centers, band_widths, norm='channel', clip=0.02)
    plt.imsave('%s%s_R=%0.f_G=%0.f_B=%0.f_map_norm.png' % (sample_figdirs[i], scans[i], band_centers[0], band_centers[1], band_centers[2]), np.repeat(np.repeat(RGB_map, 5, axis=0), 5, axis=1))
    
"""
# ==================================================
# plot average spectra for Regions of Interest
# ==================================================
"""

# specify which ROIs to plot for each scan, as follows: 'scan_name': [radius, (x0,y0), (x0,y0), (x0,y0)...]
ROIs = {
    'FM2_Area1': [20.0, (27., 29.)],
    'FM2_Area2': [0.25, (3.8, 0.4), (2.6, 1.3), (1.2, 1.3)],
    'FM2_Area3': [0.5, (4.7, 1.8), (3.2, 5.5), (6.3, 6.7), (4.7, 4.8)],
    'FM2_Area4': [0.25, (3.9, 0.6), (4.4, 0.9), (2.3, 1.4), (1.5, 1.0), (0.8, 1.1)],
    'SAU-008_Survey1': [0.5, (6.0, 12.), (6.0, 2.5), (10., 7.5), (2.0, 12.)],
    'SAU-008_Survey2': [0.5, (1.8, 9.7), (5.8, 5.7), (12.7, 2.3), (1.3, 0.7)],
    'SAU-008_Survey3': [0.3, (2.7, 1.2), (5.0, 1.0), (2.7, 2.8), (4.1, 0.1)],
    'SAU-008_Survey4': [0.3, (4.2, 2.2), (2.0, 3.6), (6.0, 4.8), (5.5, 0.8)],
    'SAU-008_Survey5': [0.3, (2.2, 2.2), (1.7, 0.6), (4.6, 3.7), (0.8, 3.5), (3.3, 2.2)],
    'SAU-008_Region1': [0.2, (1.3, 1.8), (2.6, 1.7), (2.3, 0.9), (4.3, 0.8), (4.1, 1.1)],
    'SAU-008_Region2': [0.2, (0.8, 0.9), (2.8, 1.6), (2.2, 0.2), (1.7, 0.8)],
    'SAU-008_Region3': [0.2, (4.0, 2.0), (3.0, 0.4), (1.7, 0.8), (3.8, 0.9), (3.5, 1.3), (0.8, 2.1), (1.1, 2.1), (1.8, 1.6), (0.6, 1.6), (1.3, 1.2)]
}

target_shape = 'circle'

if Plot_ROIs == True:
    print
    print "plotting manually-specified ROIs"
    for i in range(0, len(scans)):
        print
        print i, scans[i]
        sample = samples[i]
        scan = scans[i]
        print np.shape(xy_coords[i])
        x_coords, y_coords = xy_coords[i]
        print np.shape(x_coords), np.shape(y_coords)
        colors = []
        aspect = float(np.size(map_positions[i], axis=0)) / float(np.size(map_positions[i], axis=1))
        print "aspect ratio:", aspect
        plt.figure(figsize=(8,4*aspect))
        ax1 = plt.subplot2grid((1,2), (0,0))
        ax2 = plt.subplot2grid((1,2), (0,1))
        ax1.set_title("R=%0.f, G=%0.f, B=%0.f cm-1" % (RGB_bands[0], RGB_bands[1], RGB_bands[2]))
        ax2.set_title(scan)
        ### ax1.set_xlim(-0.5, np.size(x_coords, axis=1)-0.5)
        ### ax1.set_ylim(np.size(x_coords, axis=0)-0.5, -0.5)
        x_start = np.amax([RGB_bands[0]-RGB_widths[0]-100, 255])
        x_end = np.amin([RGB_bands[-1]+RGB_widths[-1]+100, 410])
        ax2.set_xlim(x_start, x_end)
        ax2.set_xlabel("Wavelength (nm)")
        print "x, y index arrays:", np.shape(x_coords), np.shape(y_coords)
        print "map position array:", np.shape(map_positions[i])
        # generate fully normalised map and save to file
        RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(wavelength[i], y[i], map_positions[i], RGB_bands, RGB_widths, norm='full', clip=0.02)
        plt.imsave('%s%s_R=%0.f_G=%0.f_B=%0.f_map.png' % (sample_figdirs[i], scans[i], RGB_bands[0], RGB_bands[1], RGB_bands[2]), np.repeat(np.repeat(RGB_map, 5, axis=0), 5, axis=1))
        # generate channel-normalised map and save to file
        RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(wavelength[i], y[i], map_positions[i], RGB_bands, RGB_widths, norm='channel', clip=0.02)
        plt.imsave('%s%s_R=%0.f_G=%0.f_B=%0.f_map_norm.png' % (sample_figdirs[i], scans[i], RGB_bands[0], RGB_bands[1], RGB_bands[2]), np.repeat(np.repeat(RGB_map, 5, axis=0), 5, axis=1))
        print np.shape(RGB_map)
        for i2 in range(0, len(RGB_bands)):
            ax2.axvspan(RGB_bands[i2]-RGB_widths[i2], RGB_bands[i2]+RGB_widths[i2], color=['r', 'g', 'b'][i2], alpha=0.1)
        im = ax1.imshow(RGB_map, extent=(0., np.amax(x_coords), np.amax(y_coords), 0.))
        # find mean F/R maxima for settings y limits
        F_max = find_max(wavelength[i], np.mean(y[i], axis=0), x_start, x_end)[1]
        # plot mean spectra for whole scan
        ### ax3.plot(wavelength[i], np.mean(y[i], axis=0), 'k', label='mean')
        num = 0
        mask = np.full_like(x_coords, True)
        ROI_F = []
        temp = sample + "_" + scan.split("_")[1]
        print temp
        if temp in ROIs.keys():
            width = ROIs[temp][0]
            for target in ROIs[temp][1:]:
                num += 1
                print "        ROI %s: (x,y) =" % num, target
                # plot target area on ax1 image
                circle_azi = np.linspace(-np.pi, np.pi, 100)
                circle_x = target[0] + width * np.cos(circle_azi)
                circle_y = target[1] + width * np.sin(circle_azi)
                ax1.plot(circle_x, circle_y, 'w')
                ax1.text(target[0], target[1], num, color='w', ha='center', va='center')
                target_mean = area_mean(wavelength[i], y[i], map_positions[i], (x_coords, y_coords), target, shape='circle', width=width)
                ROI_F.append(target_mean)
                ax2.plot(wavelength[i], target_mean, label="ROI #%s" % num)
                if find_max(wavelength[i], target_mean, x_start, x_end)[1] > F_max:
                    F_max = find_max(wavelength[i], target_mean, x_start, x_end)[1]
                ### print np.shape(mask)
                ### print np.shape(np.absolute(xy_coords[0] - target[1])**2 + np.absolute(xy_coords[1] - target[0])**2 > width**2)
                mask = np.logical_and(mask, np.absolute(x_coords - target[1])**2 + np.absolute(y_coords - target[0])**2 > width**2)
        if np.all(mask) == True:
            # plot mean spectra for all points
            ax2.plot(wavelength[i], np.mean(y[i], axis=0), 'k', label='non-ROI')
        else:
            # plot mean spectra for all points outside ROIs
            print "    spectra outside ROIs:", np.shape(y[i][map_positions[i]][mask])
            ax2.plot(wavelength[i], np.mean(y[i][map_positions[i]][mask], axis=0), 'k', label='non-ROI')
        ax2.set_ylim(-0.2*F_max, 1.2*F_max)
        ax2.minorticks_on()
        plt.tight_layout()
        plt.savefig('%s%s_ROIs.png' % (sample_figdirs[i], scans[i]), dpi=300)
        plt.savefig('%s%s_ROIs.svg' % (sample_figdirs[i], scans[i]), dpi=300)
        plt.show()
        print '    saving figure to %s%s_ROIs.png' % (sample_figdirs[i], scans[i])
        
        norm = False
        norm_wavelength = 400
        
        if len(ROI_F) > 0:
            x_range = (255., 410.)
            print x_range
            plt.figure(figsize=(8,4))
            plt.suptitle("%s" % (scans[i]))
            ax1 = plt.subplot(111)
            ax1.set_xlabel("Wavelength (nm)")
            ax1.set_xlim(x_range)
            # add wavelength ticks to top of ax1
            ax1_top = ax1.twiny()
            x_range = ax1.get_xlim()
            major_labels, major_locations, minor_locations = get_wavelength_ticks(x_range)
            ax1_top.set_xticks(major_locations)
            ax1_top.set_xticklabels(major_labels)
            ax1_top.xaxis.set_minor_locator(plt.FixedLocator(minor_locations))
            ax1_top.set_xlabel("Wavelength (nm)")
            ax1.plot(wavelength[i], np.mean(y[i], axis=0), 'k', label="Average")
            if norm == True:
                ax1.set_ylabel("Relative Intensity")
                F_max = 1.
                for i2 in range(0, len(ROI_F)):
                    F_norm = ROI_F[i2][np.argmin(np.abs(wavelength[i] - norm_wavelength))]
                    ax1.plot(wavelength[i], ROI_F[i2]/F_norm, label="ROI %s" % (i2+1))
                    y_max = find_max(wavelength[i], ROI_F[i2]/F_norm, x_range[0], x_range[1])[1]
                    if y_max > F_max:
                        F_max = y_max
            else:
                ax1.set_ylabel("Average Intensity (counts)")
                F_max = 1.
                for i2 in range(0, len(ROI_F)):
                    ax1.plot(wavelength[i], ROI_F[i2], label="ROI %s" % (i2+1))
                    y_max = find_max(wavelength[i], ROI_F[i2], x_range[0], x_range[1])[1]
                    if y_max > F_max:
                        F_max = y_max
            ax1.legend(loc=2)
            ax1.set_ylim(-0.2*F_max, 1.2*F_max)
            ax1.minorticks_on()
            plt.tight_layout(rect=[0, 0., 1, 0.93])
            plt.savefig('%s%s_ROIs-Fluor.png' % (sample_figdirs[i], scans[i]), dpi=300)
            plt.savefig('%s%s_ROIs-Fluor.svg' % (sample_figdirs[i], scans[i]), dpi=300)
            plt.show()
            print '    %s%s_ROIs-Fluor.png' % (sample_figdirs[i], scans[i])
            
            # save ROI spectra to file
            save_name = "%s" % (scans[i])
            save_data = np.vstack((wavelength[i], wavelength2shift(wavelength[i])))
            header = ["Wavelength (nm)", "Raman shift (cm-1)"]
            for i2 in range(0, len(ROI_F)):
                header += ["ROI %s" % i2]
                save_data = np.vstack((save_data, ROI_F[i2]))
            print "    saving %s ROI spectra to file, shape:" % len(ROI_F), np.shape(save_data)
            np.savetxt("%s%s_Fluor_ROIs.txt" % (sample_outdirs[i], save_name), save_data.transpose(), delimiter='\t', header="\t".join(header))
            np.savetxt("%s%s_Fluor_ROIs.csv" % (sample_outdirs[i], save_name), save_data.transpose(), delimiter=',', header=",".join(header))
        
"""
# ==================================================
# plot spectra for Points of Interest
# ==================================================
"""

# specify which POIs to plot for each scan, as follows: 'scan_name': [[list of spec indices], [list of references to compare to]]
POIs = {
    'SAU-008_Region1': [[], ['Gypsum_powder', 'Calcite_powder', 'Enstatite_C7758', 'Humic Acid_powder', 'Anthracite_powder']]
}

ROI_data = [{'count': 0, 'names': [], 'spectra': []} for i in range(len(scans))]

if Plot_POIs == True:
    print
    print "plotting manually-specified (0-indexed) point spectra and comparing to spectral standards"
    for i in range(0, len(scans)):
        print
        print scans[i]
        sample = samples[i]
        scan = scans[i]
        x_coords, y_coords = (xy_coords[i])
        ROI_data.append({'count': 0, 'names': [], 'spectra': []})
        if scan in POIs.keys():
            points = POIs[scan][0]
            print "    points for plotting:", points
            standards = POIs[scan][1]
            print "    standards for plotting:", standards
            point_names = []
            point_spectra = []
            if len(points) > 0:
                for i2 in range(0, len(points)):
                    print points[i2], np.size(points[i2]),
                    if np.size(points[i2]) > 1:
                        ROI_data[i]['count'] += 1
                        print "ROI:", np.shape(np.mean(y[i][points[i2]], axis=0))
                        point_spectra.append(np.mean(y[i][points[i2]], axis=0))
                        point_names.append("ROI %s" % ROI_data[i]['count'])
                        ROI_data[i]['spectra'].append(np.mean(y[i][points[i2]], axis=0))
                        ROI_data[i]['names'].append("ROI %s" % ROI_data[i]['count'])
                    else:
                        print "single point:", np.shape(y[i][points[i2]])
                        point_spectra.append(y[i][points[i2]])
                        point_names.append("spec %s" % points[i2])
                if ROI_data[i]['count'] > 0:
                    # create ROI 0 (non-ROI points)
                    nonROI = np.asarray([i3 for i3 in range(len(y[i])) if i3 not in np.concatenate(points)])
                    print nonROI
                    points.insert(0, [nonROI])
                    print "%s out of %s points are not assigned to an ROI" % (len(nonROI), len(y[i]))
                    point_spectra.insert(0, np.mean(y[i][nonROI], axis=0))
                    point_names.insert(0, "non-ROI")
                    ROI_data[i]['spectra'].insert(0, np.mean(y[i][nonROI], axis=0))
                    ROI_data[i]['names'].insert(0, "non-ROI")
                point_spectra = np.asarray(point_spectra)
                print "    point Fluor spectra array:", np.shape(point_spectra)
            else:
                print "    no points specified!"
                    
            if ROI_data[i]['count'] > 0:
                    # generate figure showing ROI locations
                    plt.figure(figsize=(8,6))
                    plt.suptitle(" %s ROIs" % (scan))
                    ax1 = plt.subplot2grid((1,2), (0,0))
                    ax2 = plt.subplot2grid((1,2), (0,1))
                    ax1.set_title("Fluorescence\nR=%0.f, G=%0.f, B=%0.f nm" % (RGB_bands[0], RGB_bands[1], RGB_bands[2]))
                    ax1.set_xlim(-0.5, np.size(xy_coords[0], axis=1)-0.5)
                    ax1.set_ylim(np.size(xy_coords[0], axis=0)-0.5, -0.5)
                    ax2.set_title("ROIs")
                    ax2.set_xlim(-0.5, np.size(xy_coords[0], axis=1)-0.5)
                    ax2.set_ylim(np.size(xy_coords[0], axis=0)-0.5, -0.5)
                    print "xy index array:", np.shape(xy_coords)
                    print "map position array:", np.shape(map_positions[i])
                    print np.shape(RGB_map)
                    RGB_map, RGB_hists, RGB_limits = map_spectral_RGB(wavelength[i], y[i], map_positions[i], RGB_bands, RGB_widths, norm='full', clip=0.02)
                    print np.shape(RGB_map)
                    im = ax1.imshow(RGB_map, extent=(0., np.amax(x_coords), np.amax(y_coords), 0.))
                    ROI_map = np.zeros_like(RGB_map)
                    colors = [(0., 0., 1.), (1., 0., 0.), (0., 1., 0.), (1., 0., 1.), (1., 1., 0.), (0., 1., 1.)]
                    num = 0
                    handles = []
                    text = 'single points'
                    print points
                    for i2 in range(0, len(point_names)):
                        color = colors[num]
                        if "ROI" in point_names[i2]:
                            print points[i2]
                            text = 'ROI averages'
                            if point_names[i2] == 'non-ROI':
                                color = (0., 0., 0.)
                            else:
                                color = colors[num]
                                num += 1
                            for point in points[i2]:
                                serp_indx = np.unravel_index(point, np.shape(RGB_map[:,:,0]))
                                print point, serp_indx
                                map_indx = map_positions[i][serp_indx]
                                map_indx_2d = np.unravel_index(map_indx, np.shape(RGB_map[:,:,0]))
                                ### print map_indx, map_indx_2d
                                ROI_map[map_indx_2d] = color
                            handles.append(mpl.patches.Patch(color=color, label=point_names[i2]))
                    im = ax2.imshow(ROI_map, extent=(0., np.amax(x_coords), np.amax(y_coords), 0.))
                    plt.imsave('%s%s_ROI_map.png' % (sample_figdirs[i], scans[i]), np.repeat(np.repeat(ROI_map, 5, axis=0), 5, axis=1))
                    ax2.legend(handles=handles)
                    plt.savefig('%s%s_ROI_maps.png' % (sample_figdirs[i], scans[i]), dpi=300)
                    plt.show()

            if len(standards) > 0:
                    # attempt to import reference standards
                    Refs = False
                    for standard in standards:
                        if standard not in mob_data['standards']:
                            print "    attempting to import reference spectra for %s" % standard
                            ref_F = False
                            while True:
                                try:
                                    # import Fluorescence
                                    spec_dirs = glob.glob('%s%s/*_*ppp_*A_*_Fluor_spectra.txt' % (ref_data_dir, standard))
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
                                    both_test = np.logical_and(np.asarray(current_temp) == ref_current, np.asarray(pulse_temp) == ref_pulses)
                                    print "        both_test:", both_test
                                    if np.any(both_test) == True:
                                        # then at least one spectrum has matching settings
                                        indx = np.arange(0, len(spec_dirs))[both_test][-1]
                                        print "        match found: %s, %0.f ppp %0.f A" % (indx, pulse_temp[indx], current_temp[indx])
                                    elif np.any(pulse_temp == ref_pulses) == True:
                                        # then at least one spectrum has a matching pulse count
                                        indx = np.arange(0, len(spec_dirs))[pulse_temp == ref_pulses][-1]
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
                                    print "        Reference laser values: %0.0f ppp, %0.0f A, %0.2f uJ" % (pulse_temp, current_temp, dose_temp)
                                    # check saturation
                                    test = spec[2] > 60000.
                                    saturation_temp = float(np.sum(test)) / float(np.size(spec[2]))
                                    print "        spectrum saturation: %0.2f" % saturation_temp
                                    # store data in brassboard dict
                                    ref_F = True
                                    ref_data['wavelength'].append(spec[1])
                                    ref_data['y_av_norm'].append(spec[4])
                                    ref_data['std_norm'].append(spec[5])
                                    ref_data['F_pulses'].append(pulse_temp)
                                    ref_data['F_current'].append(current_temp)
                                    ref_data['F_dose'].append(dose_temp)
                                    print "        successfully imported Reference Fluor spectrum for %s!" % standard
                                    break
                                except Exception as e:
                                    print "        something went wrong! %s" % e
                                    break
                            if ref_R == True:
                                ref_data['standards'].append(standard)
                    if len(mob_data['standards']) > 0:
                        Refs = True
                        print "%s Reference standards imported:" % len(ref_data['standards'])
                        print "    ", ref_data['standards']
                
            print "   generating POI figure..."
                
            length = len(points) + len([standard for standard in standards if standard in ref_data['standards']])
            width = 5.
            print
            print "      spectra to plot: %s SHERLOC %s points and %s standards" % (len(points), text, len([standard for standard in standards if standard in ref_data['standards']]))
                
            plt.figure(figsize=(width, np.amax([6., 0.5+0.5*length])))
            plt.suptitle("%s" % (scan))
            ax1 = plt.subplot(111)
            ax1.set_xlabel("Wavelength (nm)")
            ax1.set_ylabel("Normalised Intensity")
            ax1.set_xlim(x_start, x_end)
            ax1.set_yticks([])
            cumulative_offset = 0.
            count = 0
            print "points:", points
            for i2 in range(0, len(point_names)):
                if np.size(point_spectra[i2], axis=0) == 1:
                    y_temp = np.ravel(point_spectra[i2])
                else:
                    y_temp = point_spectra[i2]
                print i2, np.shape(wavelength[i]), np.shape(y_temp), x_start, x_end
                y_max = find_max(wavelength[i], y_temp, x_start, x_end)[1]
                ax1.plot(wavelength[i], y_temp/y_max+cumulative_offset, 'k', label=point_names[i2])
                ax1.text(x_start+0.98*(x_end-x_start), cumulative_offset+0.3, point_names[i2], ha='right')
                cumulative_offset += 1.2
            results = np.asarray([s_i for s_i,item in enumerate(ref_data['standards']) if item in standards])
            for s_i in results:
                standard = ref_data['standards'][s_i]
                color = Color_list[count % len(Color_list)]
                print "        Reference standard:", standard
                y_max = find_max(ref_data['wavelength'][s_i], ref_data['y_av_norm'][s_i], x_start, x_end)[1]
                ax1.plot(ref_data['wavelength'][s_i], ref_data['y_av_norm'][s_i]/y_max+cumulative_offset, color, label=standard.split("_")[0])
                ax1.text(x_start+0.98*(x_end-x_start), cumulative_offset+0.3, standard.split("_")[0], ha='right')
                y_max = find_max(ref_data['wavelength'][s_i], ref_data['y_av_norm'][s_i], x_start, x_end)[0]
                ax1.axvline(y_max, c=color, ls=':', alpha=0.5)
                cumulative_offset += 1.2
                count += 1
            if x_start < 1550 and 1550 < x_end:
                ax1.axvline(1550, c='k', ls=':', alpha=0.5)
                ax1.text(1550+20, cumulative_offset+0.1, 'O$_2$')
            if x_start < 2330 and 2330 < x_end:
                ax1.axvline(2330, c='k', ls=':', alpha=0.5)
                ax1.text(2330+20, cumulative_offset+0.1, 'N$_2$')
                    
            ax1.set_ylim(-0.5, cumulative_offset+0.3)
            ax1.minorticks_on()
            plt.tight_layout()
            plt.savefig('%s%s_POI-comparison.png' % (sample_figdirs[i], scans[i]), dpi=300)
            plt.savefig('%s%s_POI-comparison.svg' % (sample_figdirs[i], scans[i]), dpi=300)
            plt.show()
            print '    saving figure to %s%s_POI-comparison.png' % (sample_figdirs[i], scans[i])
                
            # save point spectra to file
            save_name = "%s" % (scans[i])
            save_data = np.vstack((wavelength[i], wavelength2shift(wavelength[i])))
            header = ["Wavelength (nm)", "Raman shift (cm-1)"]
            for i2 in range(0, len(points)):
                header += point_names[i2]
                if np.size(point_spectra[i2], axis=0) == 1:
                    y_temp = np.ravel(point_spectra[i2])
                    F_temp = np.ravel(point_Fspectra[i2])
                else:
                    y_temp = point_spectra[i2]
                    F_temp = point_Fspectra[i2]
                save_data = np.vstack((save_data, y_temp))
            print "    saving %s point spectra to file, shape:" % len(points), np.shape(save_data)
            np.savetxt("%s%s_Fluor_POIs.txt" % (sample_outdirs[i], save_name), save_data.transpose(), delimiter='\t', header="\t".join(header))
            np.savetxt("%s%s_Fluor_POIs.csv" % (sample_outdirs[i], save_name), save_data.transpose(), delimiter=',', header=",".join(header))
                
            # Raman spectra comparison plots
            for i2 in range(0, len(point_names)):
                    if np.size(point_spectra[i2], axis=0) == 1:
                        y_temp = np.ravel(point_spectra[i2])
                    else:
                        y_temp = point_spectra[i2]
                    plt.figure(figsize=(5,4))
                    plt.title("Sol %s, %s %s" % (sol, sequence, point_names[i2]))
                    ax1 = plt.subplot(111)
                    ax1.set_xlabel("Wavelength (nm)")
                    ax1.set_ylabel("Normalised Intensity")
                    ax1.set_xlim(x_start, x_end)
                    ax1.set_ylim(-0.5, 1.5)
                    # plot each standard
                    count = 0
                    for s_i in results:
                        standard = ref_data['standards'][s_i]
                        color = Color_list[count % len(Color_list)]
                        print "        Reference standard:", standard
                        mask = np.logical_and(600 <= ref_data['wavelength'][s_i], ref_data['wavelength'][s_i] <= 750)
                        print "            ", np.shape(ref_data['wavelength'][s_i]), np.count_nonzero(mask)
                        y_max = find_max(ref_data['wavelength'][s_i][~mask], ref_data['y_av_norm'][s_i][~mask], x_start, x_end)[1]
                        ax1.plot(ref_data['wavelength'][s_i], ref_data['y_av_norm'][s_i]/y_max, color, alpha=0.5, label=standard.split("_")[0])
                        y_max = find_max(ref_data['raman_shift'][s_i], ref_data['y_av_norm'][s_i], x_start, x_end)[0]
                        ax1.axvline(y_max, c=color, ls=':', alpha=0.5)
                        count += 1
                    # plot point spectrum
                    print "Fluor spectrum %s" % (point_names[i2])
                    y_max = find_max(wavelength[i], y_temp, x_start, x_end)[1]
                    ax1.plot(wavelength[i], y_temp/y_max, 'k', label=point_names[i2])
                    
                    ax1.legend()
                    ax1.minorticks_on()
                    plt.tight_layout()
                    print "saving figure to %s%s_%s-comp.png" % (sample_figdirs[i], scans[i], point_names[i2])
                    plt.savefig('%s%s_%s-comp.png' % (sample_figdirs[i], scans[i], point_names[i2]), dpi=300)
                    plt.savefig('%s%s_%s-comp.svg' % (sample_figdirs[i], scans[i], point_names[i2]), dpi=300)
                    plt.show()
                    
"""
# ==================================================
# plot Fluorescence intensity ratios
# ==================================================
"""

# define bands for intensity comparison in each scan
Band_ratios = {
    'SAU-008_Region1': [[300, 5], [325, 5], [400, 5]],
    'SAU-008_Region2': [[300, 5], [325, 5], [400, 5]],
    'SAU-008_Region3': [[300, 5], [325, 5], [400, 5]]
}   # bands are defined as follows: 'scan name': [[center, half-width], [center, half-width], [center, half-width]...]

# do a linear fit for I1/I2 of each ROI?
fit_ROI_line = True

if Plot_intensity_ratios == True:
    print
    print "plotting intensity ratios"
    print
    for i in range(0, len(scans)):
        sample = samples[i]
        scan = scans[i]
        x_coords, y_coords = (xy_coords[i])
        temp = sample + "_" + scan.split("_")[1]
        if temp in Band_ratios.keys():
                band_pairs = combinations(Band_ratios[temp], 2)
                print band_pairs
                for band1, band2 in band_pairs:
                    print band1, band2
                    # generate band 1 map
                    print "band 1: Fluor %0.f +/- %0.f nm" % (band1[0], band1[1])
                    map1, min1, max1 = map_spectral_band(wavelength[i], y[i], map_positions[i], band1[0], band1[1], clipping=True, clip=0.04, vmin=0.)
                    xlabel = "%0.f nm" % band1[0]
                    print "    plotting range: %0.f - %0.f counts" % (min1, max1)
                    # generate band 2 map
                    print "band 2: Fluor %0.f +/- %0.f nm" % (band2[0], band2[1])
                    map2, min2, max2 = map_spectral_band(wavelength[i], y[i], map_positions[i], band2[0], band2[1], clipping=True, clip=0.04, vmin=0.)
                    ylabel = "%0.f nm" % band2[0]
                    print "    plotting range: %0.f - %0.f counts" % (min2, max2)
                    # generate ratio map and figure out appropriate axes for plotting
                    ratio_map = map1/map2
                    print "%s/%s ratio range:" % (xlabel, ylabel), np.amin(ratio_map), np.amax(ratio_map)
                    # check for pre-defined ROIs and assign colours
                    colors = [(0., 0., 1.), (1., 0., 0.), (0., 1., 0.), (1., 0., 1.), (1., 1., 0.), (0., 1., 1.)]
                    num = 0
                    if ROI_data[i]['count'] > 0:
                        labels = []
                        handles = []
                        for point in ROI_data[i]['names']:
                            if point == "non-ROI":
                                color = 'k'
                            else:
                                color = colors[num]
                                num += 1
                            labels.append(mpl.lines.Line2D([], [], linewidth=0., marker='o', color=color, label=point))
                            handles.append(mpl.patches.Patch(color=color, label=point))
                    else:
                        ROI_map = np.zeros_like(RGB_map)
                        colors = [(0., 0., 1.), (1., 0., 0.), (1., 0., 1.), (1., 1., 0.)]
                        num = 0
                        labels = [mpl.lines.Line2D([], [], linewidth=0., marker='o', color='k', label="non-ROI")]
                        handles = []
                    # plot maps against each other
                    plt.figure(figsize=(12,6))
                    ax1 = plt.subplot2grid((2,4), (0,0))
                    ax2 = plt.subplot2grid((2,4), (0,1))
                    ax3 = plt.subplot2grid((2,4), (1,0))
                    ax0 = plt.subplot2grid((2,4), (1,1))
                    ax4 = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=2)
                    ax0.set_title("ROIs")
                    ax0.imshow(ROI_map, extent=(0., np.amax(x_coords), np.amax(y_coords), 0.))
                    text = ""
                    label1 = "%0.f nm" % band1[1]
                    ax4.set_xlim(1, 10**np.ceil(np.log10(np.amax(map1))))
                    ax1.set_title("%0.f-%0.f nm" % (band1[0]-band1[1], band1[0]+band1[1]))
                    text += "%0.fnm-v-" % band1[0]
                    label2 = "%0.f nm" % band2[1]
                    ax4.set_ylim(1, 10**np.ceil(np.log10(np.amax(map2))))
                    ax2.set_title("%0.f-%0.f nm" % (band2[0]-band2[1], band2[0]+band2[1]))
                    text += "%0.fnm" % band2[0]
                    ax3.set_title("%0.f/%0.f Intensity Ratio" % (band1[0], band2[0]))
                    ax4.set_title("%s vs %s" % (ylabel, xlabel))
                    ax4.set_xlabel("%s Intensity (counts)" % xlabel)
                    ax4.set_ylabel("%s Intensity (counts)" % ylabel)
                    ax4.set_xscale('log')
                    ax4.set_yscale('log')
                    # plot band1 intensity map
                    im1 = ax1.imshow(map1, extent=(0., np.amax(x_coords), np.amax(y_coords), 0.), vmin=0, vmax=max1, cmap=Cmap)
                    divider = make_axes_locatable(ax1)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im1, cax=cax)
                    # plot band2 intensity map
                    im2 = ax2.imshow(map2, extent=(0., np.amax(x_coords), np.amax(y_coords), 0.), vmin=0, vmax=max2, cmap=Cmap)
                    divider = make_axes_locatable(ax2)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im2, cax=cax)
                    # plot ratio map
                    imratio = ax3.imshow(ratio_map, extent=(0., np.amax(x_coords), np.amax(y_coords), 0.), vmin=0., vmax=np.sort(np.ravel(ratio_map))[int(0.95*np.size(ratio_map))], cmap='plasma')
                    plt.imsave('%s%s_%0.f-%0.f_ratio-map.png' % (sample_figdirs[i], scans[i], band1[0], band2[0]), np.repeat(np.repeat(ratio_map, 5, axis=0), 5, axis=1), vmin=0., vmax=np.sort(np.ravel(ratio_map))[int(0.95*np.size(ratio_map))], cmap='plasma')
                    divider = make_axes_locatable(ax3)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(imratio, cax=cax)
                    # plot point-by-point intensities as 2D scatter
                    ROI_map_flat = np.reshape(ROI_map, newshape=(-1,3))
                    print np.shape(ROI_map), np.shape(ROI_map_flat)
                    ax4.scatter(np.ravel(map1), np.ravel(map2), c=ROI_map_flat, alpha=0.2)
                    linregress_ROI_points_only = False
                    if linregress_ROI_points_only == True:
                        mask = np.sum(ROI_map, axis=2) == 0.
                        print np.shape(mask)
                        x_points = np.ravel(map1[~mask])
                        y_points = np.ravel(map2[~mask])
                    else:
                        x_points = np.ravel(map1)
                        y_points = np.ravel(map2)
                    result = stats.linregress(x_points, y_points)
                    xmin, xmax = ax4.get_xlim()
                    print xmin, xmax
                    x_temp = np.linspace(xmin, xmax, 1000)
                    y_temp = result.slope * x_temp + result.intercept
                    ax4.plot(x_temp, y_temp, 'k:', alpha=0.5)
                    plt.figtext(0.66, 0.87, 'slope: %0.1E +/- %0.3E' % (result.slope, result.stderr))
                    plt.figtext(0.66, 0.82, 'R$^2$: %0.4f' % (result.rvalue**2))
                    if len(labels) > 1:
                        ax4.legend(handles=labels, loc=2)
                    plt.tight_layout()
                    plt.savefig('%s%s_%0.fnm-v-%0.fnm.png' % (sample_figdirs[i], scans[i], band1[0], band2[0]), dpi=300)
                    plt.show()
                    
                    # plot correlation graph on its own
                    plt.figure(figsize=(6,6))
                    plt.title("%s\n%0.f nm vs %0.f nm" % (scans[i], band1[0], band2[0]))
                    plt.xlabel("%s Intensity (counts)" % xlabel)
                    plt.ylabel("%s Intensity (counts)" % ylabel)
                    ### plt.xscale('log')
                    ### plt.yscale('log')
                    plt.xlim(1, np.amax(map1))
                    plt.ylim(1, np.amax(map2))
                    plt.scatter(np.ravel(map1), np.ravel(map2), c=ROI_map_flat, alpha=0.2)
                    plt.plot(x_temp, y_temp, 'k:', alpha=0.5)
                    plt.figtext(0.45, 0.85, 'slope: %0.1E +/- %0.3E' % (result.slope, result.stderr))
                    plt.figtext(0.45, 0.8, 'R$^2$: %0.4f' % (result.rvalue**2))
                    if len(labels) > 1:
                        plt.legend(handles=labels, loc=2)
                    plt.tight_layout()
                    plt.savefig('%s%s_%0.fnm-v-%0.fnm_scatter.png' % (sample_figdirs[i], scans[i], band1[0], band2[0]), dpi=300)
                    plt.show()
                    
for i in range(0, len(scans)):
    map1, min1, max1 = map_spectral_band(wavelength[i], y[i], map_positions[i], 340, 5, clipping=True, clip=0.04, vmin=0.)
    map2, min2, max2 = map_spectral_band(wavelength[i], y[i], map_positions[i], 300, 5, clipping=True, clip=0.04, vmin=0.)
    ratio_map = map2/map1
    indx = int(0.04 * np.size(ratio_map))
    floor = np.amax([0, np.sort(np.ravel(ratio_map))[indx]])
    ceil = np.sort(np.ravel(ratio_map))[-indx-1]
    print "vmin, vmax:", floor, ceil
    plt.title("%s:\n 300/340 nm Intensity Ratio" % (scans[i]))
    plt.imshow(ratio_map, vmin=0.2, vmax=0.8)
    plt.colorbar()
    plt.savefig("%s%s_340-300_ratio.png" % (sample_figdirs[i], scans[i]))
    plt.show()

"""
# ==================================================
# save processed spectra to files
# ==================================================
"""

for i in range(0, len(samples)):
    # prepare data arrays to save to file
    save_data = np.vstack((wavelength[i], wavelength2shift(wavelength[i]), y_av[i], std[i], y_av_norm[i], std_norm[i]))
    header = ["Scan size: %0.f points" % len(laser_energies),
                          "raman shift determined using excitation wavelength of 248.56 nm",
                          "average intensity = average of all spectra in scan after laser correction (adjusting values of each individual spectrum to account for any variations in laser energy output) + automatic cosmic ray removal",
                          "standard deviation for average spectrum",
                          "Normalised intensity: average spectrum normalised by dividing by the maximum intensity between 255 and 410 nm",
                          "standard deviation for normalised spectrum",
                          "\nWavelength (nm)", "Raman shift (cm-1)", "Average Intensity (counts)", "St. Dev. (counts)", "Normalised Intensity", "Normalised St. Dev."
             ]
    print np.shape(save_data.transpose())
    save_name = "%s_%0.fppp_%0.fA_%0.2fuJ_%s" % (scans[i], laser_pulses[i], laser_currents[i], np.mean(laser_energies[i]), datetimes[i].strftime('%Y-%m-%d'))
    print "        ", save_name
    np.savetxt("%s%s_Fspectra.txt" % (sample_outdirs[i], save_name), save_data.transpose(), delimiter='\t', header="\t".join(header))
    np.savetxt("%s%s_Fspectra.csv" % (sample_outdirs[i], save_name), save_data.transpose(), delimiter=',', header=",".join(header))
    
    # save all processed spectra
    header = np.concatenate((["Wavelength (nm)", "Raman Shift (cm-1)"], ["point %s" % i2 for i2 in range(0, len(y[i]))]))
    save_data = np.vstack((shift2wavelength(raman_shift[i]), raman_shift[i], y[i]))
    save_name = "%s_%sx%s_%0.fppp_%0.fA_%0.2fuJ_%s" % (scans[i], np.size(xy_coords[i][0], axis=0), np.size(xy_coords[i][0], axis=1), laser_pulses[i], laser_currents[i], np.mean(laser_energies[i]), datetimes[i].strftime('%Y-%m-%d'))
    np.savetxt("%s%s_Fmap.txt" % (sample_outdirs[i], save_name), save_data.transpose(), delimiter='\t', header="\t".join(header))
    np.savetxt("%s%s_Fmap.csv" % (sample_outdirs[i], save_name), save_data.transpose(), delimiter=',', header=",".join(header))
    
print
print
print "DONE"
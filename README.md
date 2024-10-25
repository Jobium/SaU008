# SaU008
Python scripts for processing and analysis of Raman &amp; fluorescence spectra of meteorite SaU 008

These scripts process Raman and fluorescence spectra and spectral maps acquired using the MOBIUS and SHERLOC instruments.
 - MOB_R-mapping: handles Raman maps from MOBIUS, does baseline subtraction, peak fitting, RegionOfInterest (ROI) plotting, figure generation
 - MOB_F-mapping: handles Fluorescence maps from MOBIUS, does baseline subtraction, peak fitting, ROI plotting, figure generation
 - MOB_DG_fitting: does specialised organic D&G band region fitting of Raman spectra from MOBIUS
 - MOB_ROI-comparison: does comparison of previously generated ROIs, including specialised D&G region fitting
 - SRLC_RF-mapping: handles combined Raman+Fluorescence maps from SHERLOC, does baseline subtraction, peak fitting, ROI plotting, figure generation

Any updates to this script will be made available online at www.github.com/Jobium/SaU008/

Last updated by Dr Joseph Razzell Hollis on 2024-10-25. Tested on Python 2.7.18.

These scripts require Python 2.7 and the following packages: os, math, glob, datetime, numpy, pandas, matplotlib, lmfit, itertools, scipy.

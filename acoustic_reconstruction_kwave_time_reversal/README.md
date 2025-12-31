# Time-Reversal Reconstructions (Study 2)

This folder contains MATLAB code used to generate **time-reversal
reconstructions** that serve as inputs to the neural network in **Study 2**
of the accompanying qPACT paper.

## Overview
The script:
- Loads simulated photoacoustic sensor measurements
- Applies noise and frequency filtering consistent with the study design
- Runs 3D time-reversal reconstruction using k-Wave
- Saves reconstructed initial pressure fields for each wavelength

## Requirements
- MATLAB
- k-Wave toolbox

## Files
- `time_reversal_recon2.m`: time-reversal reconstruction script used in the study

# EMG Processing Steps - Kayla Russell-Bertucci (last modified: Dec 12, 2023)
# Output this file into any file containing emg data
# variables can be changed depending on processing preferences

# Outputs:
#       - remove_bias = removes DC bias
#       - butter_bandpass = creates a bandpass filter
#       - full_wave_rectify = rectifies the signal 
#       - process_muscle = processes signal with the steps above


import os
import pandas as pd
import numpy as np
import scipy


def remove_bias(signal: np.ndarray):
    """Remove DC bias from signal.
    
    Args:
        signal: raw signal from MVC data.
    """
    MVC_signal_bias = np.mean(signal)
    MVC_signal_no_bias = signal - MVC_signal_bias
    return MVC_signal_no_bias

# def butter_bandpass(signal: np.ndarray):
#     """Function to design and apply Butterworth Filter

#     Args:
#         signal: This is the non-biased signal after remove_bias
#     """
#     # Butterworth bandpass filter specifications
#     fs = 2000 # Sampling frequency
#     a = 30 # Highpass cut off @ 30 Hz to remove HR (Drake and Callaghan 2006)
#     b = 500 # Lowbass cutoff @ 500 Hz 
#     Wn = np.array([a, b]) 
#     sos = scipy.signal.butter(4, Wn, btype='bandpass', fs=fs, output='sos') 
#     MVC_filtered_signal = scipy.signal.sosfiltfilt(sos, signal)
#     # MVC_filtered_signal = scipy.signal.filtfilt(b, a, signal)
#     return MVC_filtered_signal

def butter_high(signal:np.ndarray):
    fs = 2000 # Sampling frequency
    a = 30 # Highpass cut off @ 30 Hz to remove HR (Drake and Callaghan 2006)
    b = 500 # Lowbass cutoff @ 500 Hz 
    Wn = (a/(0.5*fs))
    sos = scipy.signal.butter(4, Wn, btype='highpass', fs=fs, output='sos')
    highpass_filtered_signal = scipy.signal.sosfiltfilt(sos, signal)
    return highpass_filtered_signal

def butter_low(signal: np.ndarray):
    fs = 2000 # Sampling frequency
    a = 30 # Highpass cut off @ 30 Hz to remove HR (Drake and Callaghan 2006)
    b = 500 # Lowpass cutoff @ 500 Hz 
    Wn = (b/(0.5*fs))
    sos = scipy.signal.butter(4, Wn, btype='lowpass', fs=fs, output='sos')
    lowpass_filtered_signal = scipy.signal.sosfiltfilt(sos, signal)
    return lowpass_filtered_signal


    
def full_wave_rectify(signal: np.ndarray):
    """Rectify filtered signal.

    Args:
        signal: Bandpass filtered signal.
    """
    MVC_FWR = np.absolute(signal)
    return MVC_FWR

# def process_muscle(signal: np.ndarray):
#     unbiased_signal = remove_bias(signal)
#     bandpass_signal = butter_bandpass(unbiased_signal)
#     rectified_signal = full_wave_rectify(bandpass_signal)
#     return rectified_signal


def process_signal(signal: np.ndarray):
    unbiased_signal = remove_bias(signal)
    highpass_signal = butter_high(unbiased_signal)
    rectified_signal = full_wave_rectify(highpass_signal)
    lowpass_signal = butter_low(rectified_signal) 
    return lowpass_signal 
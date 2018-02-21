# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:55:11 2018

@author: pdzwiniel
@python_version: 3.6
"""
import scipy.signal as scisig
import numpy as np
import scipy.fftpack as scifft

# FILTERING, SMOOTHING, UP- AND DOWNSAMPLING
def filtfilt_butterworth(sig, sf, cf, order=1, btype='bandpass'):
    """Two-sided Butterworth filter.
    
    Parameters
    ----------
    sig : numpy.ndarray
        Signal to filter.
    sf : int
        Signal Sampling frequecy.
    cf : float | list of float of length 2
        Filter frequencies. When using btype 'lowpass' or 'highpass' use single float. When using btype 'bandstop'
        or 'bandpass' use list of float of length 2.
    order : int in range of 1-5.
        Order of the filter. Default value is 1.
    btype : str
        One of the four filter types, ie. 'lowpass', 'highpass', 'bandstop', 'bandpass'. Default value is 'bandpass'.
    
    Returns
    -------
    filtered : numpy.ndarray
        Filtered sig.
    """
    if isinstance(sig, np.ndarray) and isinstance(sf, int) and sf >= 1 and isinstance(cf, list) 
        and len(cf) in range(1, 3) and isinstance(order, int) and order in range(1, 6)
        and btype in ['lowpass', 'highpass', 'bandstop', 'bandpass']:
        if btype == 'highpass' or btype == 'lowpass':
            b, a = scisig.butter(order, Wn=cf / (0.5 * sf), btype=btype, analog=0, output='ba')
            return scisig.filtfilt(b, a, sig)
        elif btype == 'bandstop' or btype == 'bandpass':
            b, a = scisig.butter(order, Wn=(cf[0] / (0.5 * sf), cf[1] / (0.5 * sf)), btype=btype, analog=0, output='ba')
            return scisig.filtfilt(b, a, sig)
    else:
        raise ValueError("Inappropriate type or value of one of the arguments. Please read carefully function docstring.")

def upsample(sig, i_factor):
    """Upsample one-dimensional signal with the use of linear interpolation.

    Parameters
    ----------
    sig : 1D numpy.ndarray
        One-dimensional signal for interpolation.
    i_factor : int, range(1, inf)
        Interpolation factor. Must be higher than 0.
    
    Returns
    -------
    i_sig : 1D numpy.ndarray
        One-dimensional signal interpolated lineary by factor equal to 'i_factor'.
    """
    if isinstance(sig, np.ndarray) and sig.ndim == 1 and isinstance(i_factor, int) and i_factor >= 1:
        x = np.linspace(0, sig.size, sig.size)
        y = sig
        i_x = np.linspace(0, sig.size, sig.size * i_factor)
        i_y = np.interp(i_x, x, y)
        i_sig = i_y
        return i_sig
    else:
        raise ValueError("Inappropriate type, shape or value of one of the arguments. Please read carefully function docstring.")

def downsample(sig, d_factor):
    """Downsample one-dimensional signal with the use of reshaping.

    Parameters
    ----------
    sig : 1D numpy.ndarray
        One-dimensional signal for downsampling.
    d_factor : int, range(1, inf)
        Downsampling factor. Must be higher than 0.

    Returns
    -------
    d_sig : 1D numpy.ndarray
        One-dimensional signal downsampled lineary by factor equal to 'd_factor'.
    """
    if isinstance(sig, np.ndarray) and sig.ndim == 1 and isinstance(d_factor, int) and d_factor >= 1:
        d_sig = sig.reshape(-1, d_factor).mean(axis=1)
        return d_sig
    else:
        raise ValueError("Inappropriate type, shape or value of one of the arguments. Please read carefully function docstring.")

# EXPLORATION AND MARKING
def mark_photodiode_changes(sig, threshold, wait_n_samples, direction='left-to-right'):
    """Create 1D-array of zeros and ones, where ones indicate places where photodiode signal exceeds some
    specific threshold value. This 1D-array is the same length as photodiode signal.
    
    Parameters
    ----------
    sig : 1D numpy.ndarray
        Photodiode signal.
    threshold : float
        Threshold value above which photodiode signal will be marked.
    wait_n_samples : int
        Wait n samples after last marker before trying to put next marker. Must be >= 0.
    direction : str
        Direction in which photodiode signal course will be analyzed and marked. There are three directions, ie.
        'left-to-right', 'right-to-left', 'both'. In case of 'both' photodiode signal course will be first analyzed
        'left-to-right' and than 'right-to-left'. Default value is 'left-to-right'.
    
    Returns
    -------
    markers : 1D numpy.ndarray
        Array of zeros and ones, where ones are markers.
    """
    if isinstance(sig, np.ndarray) and sig.ndim == 1 and isinstance(threshold, float) and isinstance(wait_n_samples, int)
        and wait_n_samples >= 0 and direction in ['left-to-right', 'right-to-left', 'both']:
        if direction == 'left-to-right':
            markers = np.zeros(len(sig))
            iterator = 0
            wait_until_next_mark = wait_n_samples
            for sample in sig:
                if sample > threshold and wait_until_next_mark >= wait_n_samples:
                    markers[iterator] = 1
                    wait_until_next_mark = 0
                iterator += 1
                wait_until_next_mark += 1
            return markers
        elif direction == 'right-to-left':
            markers = np.zeros(len(sig))
            iterator = len(sig)
            wait_until_next_mark = wait_n_samples
            for sample in reversed(sig):
                if sample > threshold and wait_until_next_mark >= wait_n_samples:
                    markers[iterator] = 1
                    wait_until_next_mark = 0
                iterator -= 1
                wait_until_next_mark += 1
            return markers
        else:
            markers_left_to_right = mark_photodiode_changes(sig, threshold, wait_n_samples, direction='left-to-right')
            markers_right_to_left = mark_photodiode_changes(sig, threshold, wait_n_samples, direction='right-to-left')
            markers = markers_left_to_right + markers_right_to_left
            return markers
    else:
        raise ValueError("Inappropriate type, shape or value of one of the arguments. Please read carefully function docstring.")

# TRANSFORMATIONS
def spectrum(sig, time_scale, abs=True):
    """Compute the one-dimensional Discrete Fourier Transform (DFT) for given N-dimensional signal.

    Parameters
    ----------
    sig : numpy.ndarray
        Signal for DFT (can be complex).
    time_scale : 1D numpy.ndarray
        One-dimensional time scale in seconds.
    abs : boolean
        If True, the result of DFT will be absolute. If False, the result of DFT will be complex. Default value is True.

    Returns
    -------
    freqs : 1D numpy.ndarray
        One-dimensional array containing information about the signal frequencies.
    fft : numpy.ndarray or complex numpy.ndarray
        One-dimensional array containing the result of DFT. If parameter 'abs' is equal to False, the result will
        be complex numpy.ndarray.
    """
    if isinstance(sig, np.ndarray) and isinstance(time_scale, np.ndarray) and time_scale.ndim == 1:
        freqs = scifft.fftfreq(sig.size, d=time_scale[1]-time_scale[0])
        fft = np.fft.fft(sig)
        if abs:
            return freqs, np.abs(fft)
        else:
            return freqs, fft
    else:
        raise ValueError("Inappropriate type or shape of one of the arguments. Please read carefully function docstring.")


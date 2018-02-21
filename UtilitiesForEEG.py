# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:55:11 2018

@author: pdzwiniel
@python_version: 3.6
"""
import scipy.signal as scisig
import numpy as np

# FILTERS
def filtfilt_butterworth(sig, sf, cf, order=1, btype='bandpass'):
    """Two-sided Butterworth filter.
    
    Parameters
    ----------
    sig : ndarray
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
    filtered : ndarray
        Filtered sig.
    """
    if btype == 'highpass' or btype == 'lowpass':
        b, a = scisig.butter(order, Wn=cf / (0.5 * sf), btype=btype, analog=0, output='ba')
        return scisig.filtfilt(b, a, sig)
    elif btype == 'bandstop' or btype == 'bandpass':
        b, a = scisig.butter(order, Wn=(cf[0] / (0.5 * sf), cf[1] / (0.5 * sf)), btype=btype, analog=0, output='ba')
        return scisig.filtfilt(b, a, sig)

# EXPLORE AND MARK
def mark_photodiode_changes(sig, threshold, wait_n_samples, direction='left-to-right'):
    """Create 1D-array of zeros and ones, where ones indicate places where photodiode signal exceeds some
    specific threshold value. This 1D-array is the same length as photodiode signal.
    
    Parameters
    ----------
    sig : 1D-array
        Photodiode signal.
    threshold : float
        Threshold value above which photodiode signal will be marked.
    wait_n_samples : int
        Wait n samples after last marker before trying to put next marker.
    direction : str
        Direction in which photodiode signal course will be analyzed and marked. There are three directions, ie.
        'left-to-right', 'right-to-left', 'both'. In case of 'both' photodiode signal course will be first analyzed
        'left-to-right' and than 'right-to-left'. Default value is 'left-to-right'.
    
    Returns
    -------
    markers : 1D-array
        Array of zeros and ones, where ones are markers.
    """
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
    elif direction == 'both':
        markers_left_to_right = mark_photodiode_changes(sig, threshold, wait_n_samples, direction='left-to-right')
        markers_right_to_left = mark_photodiode_changes(sig, threshold, wait_n_samples, direction='right-to-left')
        markers = markers_left_to_right + markers_right_to_left
        return markers
    else:
        raise ValueError("Incorrect value of 'direction' parameter. Available values: 'left-to-right', 'right-to-left', 'both'")

    
    
    
    
    
    
    
    
    
    
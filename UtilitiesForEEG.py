# -*- coding: utf-8 -*-
"""
@author: Piotr Dzwiniel
@python_version: 3.6.4
"""
from scipy import signal as scisig
import numpy as np
import scipy.fftpack as scifft
import math

# ARTIFACTS REMOVAL
def remove_current_pulse_artifacts(sig, markers, window, n_draws, return_artifacts=False):
    """Remove current pulse artifacts from one-dimensional signal based on artifacts occurences represented
    by one-dimensional markers signal. Current pulse artifacts removal is performed in following steps:
    1. Extract current pulse artifacts from 'sig' based on 'markers' which contains ones and zeros, whereas ones
    indicate current pulse artifact occurences. 
    2. Extraction is performed around the artifact occurence in accordance to range described by 'window'.
    3. Extracted artifacts are stored in two-dimensional numpy.ndarray.
    4. We draw from stored artifacts 'n_draws' without repetition and average them in order to get averaged
    representation of current pulse artifact.
    5. We substract this averaged artifact representation from the first occurence of the artifact in 'sig'.
    6. We now repeat steps 4 and 5 for all next subsequent artifact occurences in 'sig'.

    Parameters
    ----------
    sig : 1D numpy.ndarray
        One-dimensional signal with the occurences of current pulse artifacts.
    markers : 1D numpy.ndarray
        One-dimensional signal consisted of ones and zeros, where ones correspond to the exact sample occurences 
        of current pulse artifact in 'sig'. That's why 'markers'.size' must equal to 'sig.size'.
    window : list of int of length 2
        List consisted of two values describing sample range (window) of the current pulse artifact around 
        its occurence.
    n_draws : int
        Number of draws from the collection of stored artifacts. Must be >= 1.
    return_artifacts : boolean
        If True, beside of cleared signal, function will return also collection of the stored artifacts. Default value
        is False.

    Returns
    -------
    cleared : 1D numpy.ndarray
        Cleared signal.
    artifacts : 2D numpy.ndarray
        Collection of the stored artifacts.
    """
    if (isinstance(sig, np.ndarray) and sig.ndim == 1 and isinstance(markers, np.ndarray) and markers.ndim == 1 
        and ndarray_contains_only(markers, np.array([0, 1])) and sig.size == markers.size and len(window) in range(1, 3) 
        and list_is_int(window) and isinstance(n_draws, int) and n_draws >= 1):

        # Extract artifacts.
        artifacts = []
        iterator = 0
        for marker in markers:
            if marker == 1:
                artifacts.append(sig[iterator-window[0]:iterator+window[1]])
            iterator += 1
        artifacts = np.asarray(artifacts)

        # Remove artifacts from the signal.
        iterator = 0
        for marker in markers:
            if marker == 1:

                if n_draws > np.shape(artifacts)[0]:
                    n_draws = np.shape(artifacts)[0]

                random_artifact_indices = np.random.choice(np.arange(np.shape(artifacts)[0]), n_draws, replace=False)
                avg_artifact = np.mean(np.take(artifacts, random_artifact_indices, axis=0), axis=0)
                sig[iterator-window[0]:iterator+window[1]] -= avg_artifact
            iterator += 1
        cleared = sig
        
        # Return cleared signal and extracted artifacts.
        if return_artifacts:
            return cleared, artifacts
        else:
            return cleared
    else:
        raise ValueError("Inappropriate type or value of one of the arguments. Please read carefully function docstring.")

# EXPLORATION AND MARKING
def mark_photodiode_changes(sig, threshold, wait_n_samples, direction='left-to-right'):
    """Create one-dimensional array of zeros and ones, where ones indicate places where photodiode signal exceeds some
    specific threshold value. This one-dimensional array is the same length as photodiode signal.
    
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
    if (isinstance(sig, np.ndarray) and sig.ndim == 1 and isinstance(threshold, float) and isinstance(wait_n_samples, int) and wait_n_samples >= 0 and direction in ['left-to-right', 'right-to-left', 'both']):
        if direction == 'left-to-right':
            markers = np.zeros(len(sig))
            wait_until_next_mark = wait_n_samples
            iterator = 0
            for sample in sig:
                if sample > threshold and wait_until_next_mark >= wait_n_samples:
                    markers[iterator] = 1
                    wait_until_next_mark = 0
                iterator += 1
                wait_until_next_mark += 1
            return markers
        elif direction == 'right-to-left':
            markers = np.zeros(len(sig))
            iterator = len(sig)-1
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

# FILTERING, SMOOTHING, UP- AND DOWNSAMPLING
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
    if (isinstance(sig, np.ndarray) and sig.ndim == 1 and isinstance(d_factor, int) and d_factor >= 1):
        d_sig = sig.reshape(-1, d_factor).mean(axis=1)
        return d_sig
    else:
        raise ValueError("Inappropriate type, shape or value of one of the arguments. Please read carefully function docstring.")

def filtfilt_butterworth(sig, sf, cf, order=1, btype='bandpass'):
    """Two-sided Butterworth filter.
    
    Parameters
    ----------
    sig : numpy.ndarray
        Signal to filter.
    sf : float
        Signal sampling frequecy (number of samples per second).
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
    if (isinstance(sig, np.ndarray) and isinstance(sf, float) and sf > 0 and isinstance(cf, list) 
        and len(cf) in range(1, 3) and isinstance(order, int) and order in range(1, 6)
        and btype in ['lowpass', 'highpass', 'bandstop', 'bandpass']):
        
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
    if (isinstance(sig, np.ndarray) and sig.ndim == 1 and isinstance(i_factor, int) and i_factor >= 1):
        x = np.linspace(0, sig.size, sig.size)
        y = sig
        i_x = np.linspace(0, sig.size, sig.size * i_factor)
        i_y = np.interp(i_x, x, y)
        i_sig = i_y
        return i_sig
    else:
        raise ValueError("Inappropriate type, shape or value of one of the arguments. Please read carefully function docstring.")

# SIGNAL CREATION
def create_sawtooth_pulse(freq, sf, amp, first_peak='positive'):
    """Create one-period sawtooth pulse.

    Parameters
    ----------
    freq : float
        Frequency of the pulse wave in Hz. Must be > 0.
    sf : int
        Sampling frequency of the pulse (number of samples per second). Must be > 0.
    amp : float
        Amplitude of the pulse in microamperes (uA). Must be > 0.
    first_peak : str
        Polarity of the first pulse hillock. Available options: 'positive', 'negative'. Default value is 'positive'.

    Returns
    -------
    pulse : 1D numpy.ndarray
        One-period sawtooth pulse.
    """
    if (isinstance(freq, float) and freq > 0 and isinstance(sf, int) and sf > 0 and isinstance(amp, float) 
        and amp > 0 and first_peak in ['positive', 'negative']):

        duration = 1 / freq
        time_scale = np.arange(0, duration, 1 / sf)
        pulse = scisig.sawtooth(2 * np.pi * freq * time_scale) * (amp / 2)
        if first_peak == 'negative':
            pulse *= -1
        return pulse
    else:
        raise ValueError("Inappriopriate type or value of one of the arguments. Please read carefully function docstring.")

def create_sin_pulse(freq, sf, amp, first_peak='positive'):
    """Create one-period sinusoidal pulse.

    Parameters
    ----------
    freq : float
        Frequency of the pulse wave in Hz. Must be > 0.
    sf : int 
        Sampling frequency of the pulse (number of samples per second). Must be > 0.
    amp : float
        Amplitude of the pulse in microapers (uA). Must be > 0.
    first_peak : str
        Polarity of the first pulse hillock. Available options: 'positive', 'negative'. Default value is 'positive'.

    Returns
    -------
    pulse : 1D numpy.ndarray
        One-period sinusoidal pulse.
    """
    if (isinstance(freq, float) and freq > 0 and isinstance(sf, int) and sf > 0 and isinstance(amp, float) 
        and amp > 0 and first_peak in ['positive', 'negative']):

        duration = 1 / freq
        time_scale = np.arange(0, duration, 1 / sf)
        pulse = np.sin(2 * np.pi * freq * time_scale) * (amp / 2)
        if first_peak == 'negative':
            pulse *= -1
        return pulse
    else:
        raise ValueError("Inappriopriate type or value of one of the arguments. Please read carefully function docstring.")

def create_square_pulse(freq, sf, amp, first_peak='positive'):
    """Create one-period square pulse.

    Parameters
    ----------
    freq : float
        Frequency of the pulse wave in Hz. Must be > 0.
    sf : int
        Sampling frequency of the pulse (number of samples per second). Must be > 0.
    amp : float
        Amplitude of the pulse in microamperes (uA). Must be > 0.
    first_peak : str
        Polarity of the first pulse hillock. Available options: 'positive', 'negative'. Default value is 'positive'.

    Returns
    -------
    pulse : 1D numpy.ndarray
        One-period squarewave pulse.
    """
    if (isinstance(freq, float) and freq > 0 and isinstance(sf, int) and sf > 0 and isinstance(amp, float) 
        and amp > 0 and first_peak in ['positive', 'negative']):
        
        duration = 1 / freq
        time_scale = np.arange(0, duration, 1 / sf)
        pulse = scisig.square(2 * np.pi * freq * time_scale) * (amp / 2)
        if first_peak == 'negative':
            pulse *= -1
        return pulse
    else:
        raise ValueError("Inappropriate type or value of one of the arguments. Please read carefully function docstring.")

def create_alternating_signal(duration, sf, freq, amp, s_type='sinusoidal', first_peak='positive'):
    """Create one-dimensional alternating signal using sawtooth, sinusoidal or square wave.

    Parameters
    ----------
    duration : float
        Duration of the signal in seconds. Must be > 0.
    sf : int
        Sampling frequency of the pulse (number of samples per second). Must be > 0.
    freq : float
        Frequency of the signal in Hz.
    amp : float
        Amplitude of the pulse in microampers (uA). Must be > 0.
    s_type : str
        Type of the wave used in the signal creation. Available types: 'sawtooth', sinusoidal', 'square'. 
        Default value is 'sinusoidal'.
    first_peak : str
        Polarity of the first pulse hillock. Available options: 'positive', 'negative'. Default value is 'positive'.

    Returns
    -------
    sig : 1D numpy.ndarray
        Created one-dimensional alternating signal.
    """
    if (isinstance(duration, float) and duration > 0 and isinstance(sf, int) and sf > 0 and isinstance(freq, float) 
        and freq > 0 and isinstance(amp, float) and amp > 0 and s_type in ['sawtooth', 'sinusoidal', 'square']
        and first_peak in ['positive', 'negative'] and duration * sf >= 1):
        
        temp_sig = []
        pulse_time_in_s = 1.0 / freq
        n_pulses = int(math.ceil(duration / pulse_time_in_s))
        if s_type == 'sawtooth':
            for i in np.arange(n_pulses):
                pulse = create_sawtooth_pulse(freq, sf, amp, first_peak=first_peak)
                temp_sig.append(pulse)
        elif s_type == 'sinusoidal':
            for i in np.arange(n_pulses):
                pulse = create_sin_pulse(freq, sf, amp, first_peak=first_peak)
                temp_sig.append(pulse)
        else:
            for i in np.arange(n_pulses):
                pulse = create_square_pulse(freq, sf, amp, first_peak=first_peak)
                temp_sig.append(pulse)
        temp_sig = np.asarray(temp_sig).reshape(-1)

        sig = np.zeros(int(np.around(duration * sf, decimals=0)))
        sig = temp_sig[:sig.size]

        return sig
    else:
        raise ValueError("Inappropriate type or value of one of the arguments. Please read carefully function docstring.")

# SIMPLE CALCULATIONS
def z_score(x, avg, sd):
    """Calculate z-score.

    Parameters
    ----------
    x : float
        Standardized variable..
    avg : float
        Average from population.
    sd : float
        Standard deviation from population.

    Returns
    -------
    z : float
        Z-score.
    """
    return (x - avg) / sd


def create_time_scale(n_samples, sf, unit='s'):
    """Create one-dimensional time scale.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the signal for which time scale has to be created.
    sf : int
        Sampling frequency of the signal, ie. number of samples per second.
    unit : str
        Time unit in which time scale has to be expressed. Available units: hours 'h', minutes 'min', seconds 's',
        milliseconds 'ms', microseconds 'us', nanoseconds 'ns'. Default value is 's'.
    
    Returns
    -------
    time_scale : 1D np.ndarray
        One-dimensional time scale with values expressed in a specific time unit.
    """
    if (isinstance(n_samples, int) and isinstance(sf, int) and unit in ['h', 'min', 's', 'ms', 'us', 'ns']):
        unit_convertion = {'h':3600, 'min':60, 's':1, 'ms':0.001, 'us':0.000001, 'ns':0.000000001}
        total_time_in_unit = (n_samples / sf) / unit_convertion[unit]
        dt = (1 / sf) / unit_convertion[unit]
        time_scale = np.arange(0, total_time_in_unit, dt)
        return time_scale
    else:
        raise ValueError("Innapriopriate type or value of one of the arguments. Please read carefully function docstring.")

def list_is_int(list_of_ints):
    """Check whether given list contains only int values.

    Parameters
    ----------
    list_of_ints : list
        List of presumably only int values.

    Returns
    -------
    verdict : boolean
        Return True, if 'list_of_ints" contains only in values. Otherwise, return False.
    """
    if (isinstance(list_of_ints, list) and len(list_of_ints) > 0):
        for item in list_of_ints:
            if not isinstance(item, int):
                return False
        return True 
    else:
        raise ValueError("Inappropriate type or size of the argument.")

def ndarray_contains_only(ndarray, values):
    """Check whether numpy.ndarray contains only some specific values.

    Parameters
    ----------
    ndarray : numpy.ndarray
        One-dimensional array.
    values : 1D numpy.ndarray
        One-dimensional array with values to check whether they occur in 'ndarray'.

    Returns
    -------
    verdict : boolean
        Return True, if 'ndarray' contains only 'values'. Otherwise, return False.

    """
    if (isinstance(ndarray, np.ndarray) and ndarray.ndim == 1 and isinstance(values, np.ndarray) and values.ndim == 1):
        mask = np.isin(ndarray, values)
        matches = np.sum(mask)
        if matches != ndarray.size:
            return False
        else:
            return True
    else:
        raise ValueError("Inappropriate type or shape of the argument.")

# TRANSFORMATIONS AND CORRECTIONS
def baseline_correction(sig, b_window, c_window, b_type='absolute'):
    """Perform baseline correction on a given one-dimensional signal.

    Parameters
    ----------
    sig : 1D numpy.ndarray
        One-dimensional signal for which baseline correction has to be performed.
    b_window : list of int of length 2
        Range of the 'sig' samples from which baseline should be calculated. Minimum and maximum range 
        is [0, sig.size-1].
    c_window : list of int of length 2
        Range of the 'sig' samples which should be baseline-corrected. Minimum and maximum range is [0, sig.size-1].
    b_type : str
        Type of baseline. Available options: 'absolute', 'relative', 'relchange', 'decibel' (based on 
        http://bjornherrmann.com/baseline_correction.html). Default values is 'absolute'. For 'X' is the signal 
        and for 'B' is the baseline calculated as mean(sig[window[0]:window[1]]):
        1. 'absolute' - absolute baseline, range of possible values: [-inf, inf], calculated as X - B;
        2. 'relative' - relative baseline, range of possible values: [0, inf], calculated as X / B;
        3. 'relchange' - relative change baseline, range of possible values: [-1, inf], calculated as (X - B) / B;
        4. 'decibel' - decibel baseline (defined only for power), range of possible values: [-inf, inf], calculated as
        10 * log10(X / B).

    Returns
    -------
    corrected : numpy.ndarray
        Baseline-corrected signal.
    """
    if (isinstance(sig, np.ndarray) and sig.ndim == 1 and isinstance(b_window, list) and list_is_int(b_window) 
        and len(b_window) in range(1, 3) and isinstance(c_window, list) and list_is_int(c_window) 
        and len(c_window) in range(1, 3) and b_type in ['absolute', 'relative', 'relchange', 'decibel']):
        
        baseline = np.mean(sig[b_window[0]:b_window[1]])

        if b_type == 'absolute':
            sig[c_window[0]:c_window[1]] -= baseline
        elif b_type == 'relative':
            sig[c_window[0]:c_window[1]] /= baseline
        elif b_type == 'relchange':
            sig[c_window[0]:c_window[1]] = (sig[c_window[0]:c_window[1]] - baseline) / baseline
        else:
            sig[c_window[0]:c_window[1]] = 10 * np.log10(sig[c_window[0]:c_window[1]] / baseline)

        return sig
    else:
        raise ValueError("Inappropriate type, value or shape of one of the arguments. Please read carefully function docstring.")

def hanning_correction(sig, c_window, mode='full'):
    """Perform Hanning window correction on a given one-dimensional signal.

    Parameters
    ----------
    sig : 1D numpy.ndarray
        One-dimensional signal.
    c_window : list of int of length 2
        Range of the 'sig' samples which should be Hanning-corrected. Minimum and maximum range is [0, sig.size-1].
    mode : str
        Mode of the Hanning correction. There are three available modes: 'half-left', 'half-right', 'full'. 
        Default value is 'full'. Modes description:
        1. 'half-left' - only half left part of the Hanning window is used for the correction.
        2. 'half-right' - only half right part of the Hanning window is used for the correction.
        3. 'full' - full Hanning window is used for the correction.

    Returns
    -------
    corrected : 1D numpy.ndarray
        Hanning-corrected signal.
    """
    if (isinstance(sig, np.ndarray) and sig.ndim == 1 and list_is_int(c_window) and len(c_window) in range(1, 3)
        and mode in ['half-left', 'half-right', 'full']):
        
        c_window_size = sig[c_window[0]:c_window[1]].size
        if mode == 'half-left':
            hann = np.hanning(c_window_size * 2)[:c_window_size]
            sig[c_window[0]:c_window[1]] = np.multiply(sig[c_window[0]:c_window[1]], hann)
        elif mode == 'half-right':
            hann = np.hanning(c_window_size * 2)[c_window_size:]
            sig[c_window[0]:c_window[1]] = np.multiply(sig[c_window[0]:c_window[1]], hann)
        else:
            hann = np.hanning(c_window_size)
            sig[c_window[0]:c_window[1]] = np.multiply(sig[c_window[0]:c_window[1]], hann)
        return sig
    else:
        raise ValueError("Inappropriate type, value or shape of one of the arguments. Please read carefully function docstring.")

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
    if (isinstance(sig, np.ndarray) and isinstance(time_scale, np.ndarray) and time_scale.ndim == 1):
        freqs = scifft.fftfreq(sig.size, d=time_scale[1]-time_scale[0])
        fft = np.fft.fft(sig)
        if abs:
            return freqs, np.abs(fft)
        else:
            return freqs, fft
    else:
        raise ValueError("Inappropriate type or shape of one of the arguments. Please read carefully function docstring.")

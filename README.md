# Utilities-For-EEG
This repository contains python (v3.6.4) scripts with various utilities for EEG signal preprocessing and analysis, as well as bunch of different functionalities, like generating artificial signals. Below is list of scripts and key defined functions with short description.

<h3>Script UtilitiesForEEG.py</h3>
<h4>Artifacts Removal</h4>
`remove_current_pulse_artifacts` - Remove current pulse artifacts from one-dimensional signal based on artifacts occurences represented by one-dimensional markers signal.

<h4>Exploration and Marking</h4>
`mark_photodiode_changes` - Create one-dimensional array of zeros and ones, where ones indicate where photodiode exceeds some specific threshold value. This one-dimensional array is the same length as photodiode signal.

<h4>Filtering, Smoothing, Up- and Downsampling</h4>
`downsample` - Downsample one-dimensional signal with the use of reshaping.
`filtfilt_butterworth` - Two-sided Butterworth filter.
`upsample` - Upsample one-dimensional signal with the use of linear interpolation.

<h4>Signal Creation</h4>
`create_sawtooth_pulse` - Create one-period sawtooth pulse.
`create_sin_pulse` - Create one-period sinusoidal pulse.
`create_square_pulse` - Create one-period square pulse.
`create_alternating_signal` - Create one-dimensional alternating signal using sawtooth, sinusoidal or square wave.

<h4>Simple Calculations</h4>
`create_time_scale` - Create one-dimensional time scale.
`list_is_int` - Check whether given list contains only int values.
`ndarray_contains_only` - Check whether numpy.ndarray contains only some specific values.

<h4>Transformations and Corrections</h4>
`baseline_correction` - Perform baseline correction on a given one-dimensional signal.
`hanning_correction` - Perform Hanning window correction on a given one-dimensional signal.
# Utilities-For-EEG
This repository contains python (v3.6.4) scripts with various utilities for EEG signal preprocessing and analysis, as well as bunch of different functionalities, like generating artificial signals. Below is list of scripts and key defined functions with short description.

<h2>Script UtilitiesForEEG.py</h2>
<h4>Artifacts Removal</h4>
<code>remove_current_pulse_artifacts</code> - Remove current pulse artifacts from one-dimensional signal based on artifacts occurences represented by one-dimensional markers signal.

<h4>Exploration and Marking</h4>
<code>mark_photodiode_changes</code> - Create one-dimensional array of zeros and ones, where ones indicate where photodiode exceeds some specific threshold value. This one-dimensional array is the same length as photodiode signal.

<h4>Filtering, Smoothing, Up- and Downsampling</h4>
<code>downsample</code> - Downsample one-dimensional signal with the use of reshaping.<br>
<code>filtfilt_butterworth</code> - Two-sided Butterworth filter.<br>
<code>upsample</code> - Upsample one-dimensional signal with the use of linear interpolation.

<h4>Mutual Information</h4>
<code>shannon_entropy</code> - Calculate Shannon's entropy.

<h4>Signal Creation</h4>
<code>create_sawtooth_pulse</code> - Create one-period sawtooth pulse.<br>
<code>create_sin_pulse</code> - Create one-period sinusoidal pulse.<br>
<code>create_square_pulse</code> - Create one-period square pulse.<br>
<code>create_alternating_signal</code> - Create one-dimensional alternating signal using sawtooth, sinusoidal or square wave.

<h4>Simple Calculations</h4>
<code>z_score</code> - Calculate Z-Score.<br>
<code>create_time_scale</code> - Create one-dimensional time scale.<br>
<code>list_is_int</code> - Check whether given list contains only int values.<br>
<code>ndarray_contains_only</code> - Check whether numpy.ndarray contains only some specific values.

<h4>Transformations and Corrections</h4>
<code>baseline_correction</code> - Perform baseline correction on a given one-dimensional signal.<br>
<code>hanning_correction</code> - Perform Hanning window correction on a given one-dimensional signal.<br>
<code>spectrum</code> - Compute Discrete Fourier Transform for given N-dimensional signal.

#!/usr/bin/env python
__author__ = "Thomas Martineau"
__copyright__ = "Copyright 2023, tf-stream"
__license__ = "FPA"
__version__ = "0.0.1"
__email__ = "tlc_martineau@gmail.com"
__status__ = "Pre-Alpha"
__doc__= "Utility functions."

# %% Imports
import numpy as np

# %% Push axis for broacasting.
def push_axis(ndim, axis):
    if axis >= 0:
        return ndim - axis  - 1
    return -axis - 1

# %% Is numeric.
def is_numeric_dtype(arr):
    
    # thank-you chatgpt
    numeric_dtypes = [
        np.int_, np.intc, np.intp, 
        np.int8, np.int16, np.int32, 
        np.int64, np.uint8, np.uint16, 
        np.uint32, np.uint64, 
        np.float_, np.float16,
        np.float32, np.float64,
        ]
    
    return any(np.issubdtype(arr.dtype, dt) for dt in numeric_dtypes)

# %% Frequency Handler
def check_freq_input(
        f, fs,
        two_dim=True,
        return_exception=True): 
    
    def output(e):
        if return_exception:
            return False, e
        else:
            return False
    
    # (a) check the sampling frequency
    check_fsample(fs)
        
    # (b) convert to an array, error raised.
    try:
        f = np.asarray(f) 
    except:
        return output(ValueError('f needs to be convertible to a numpy array.'))
        
    # (c) check for dimensions.
    # TODO: use at_least2d
    
    if f.ndim >= 3:
        if two_dim:
            return output(ValueError('f can only support 2 or 1 dimensions.'))
        else:
            return output(ValueError('f can only support 1 dimension.'))
    else:
        if f.ndim == 2 and not two_dim:
            return output(ValueError('f can only support 1 dimension.'))

    # (d) check the array is correctly constructred.
    # empty.
    if f.size == 0:
        return output(ValueError('f needs to contain at least one frequency point.'))
    
    # type.
    if not is_numeric_dtype(f):
        return output(ValueError('f needs to support a numeric dtype.'))
        
    # Nyquist.
    if f.max() > fs/2:
        return output(ValueError('All elements f need to be inferior to the Nyquist frequency.'))
    
    if f.min() < 0:
        return output(ValueError('All elements f need to be positive'))

    if return_exception:
        return True, None
    else:
        return False
        
def check_fsample(fs, token='fs'):
    
    # check the sampling frequency.
    if not isinstance(fs, (int, float)):
        raise TypeError('{} needs to be either a float or an integer.'.format(
            token))
    
    if fs <= 0:
        raise ValueError('{} needs to be strictly positive.'.format(
            token))
        
# %% Input Handler.
def handle_input(x, axis):
    if not isinstance(axis, int):
        raise TypeError('Axis needs to be an int.')

    try:
        x = np.asarray(x)
    except:
        raise ValueError('x is not a valid array.')

    return x
    
# %% Frequency Scales.
def freq_linear_scale(
        fs: int|float, df: int|float, 
        include_nysquit: bool=True):
    """
    Generate linearly spaced frequency between 0 and fs/2.

    Parameters
    ----------
    fs : int|float
        Sampling frequency.
    df : int|float
        Frequency spacing.
    include_nysquit : bool, optional
        Add the fs/2 as the last element of the vector. The default is True.

    Returns
    -------
    f  : np.ndarray
        Frequency vector.

    """

    check_fsample(fs)
    check_fsample(fs, token='df')
    
    end = fs/2
    if include_nysquit:
        end += df
    return np.arange(0, end, df)

def freq_dyadic_scale(
        fs: int|float, per_octave: int=2, fmin: int|float=1, 
        include_nysquit=True):
    """
    Generate exponentially (base-2) spaced frequency between fmin and fs/2.

    Parameters
    ----------
    fs : int|float
        Sampling frequency.
    per_octave : int, optional
        Number of sub-division between every other division. The default is 2.
    fmin : int|float, optional
        Minimum frequency to stop the scale. The default is 1.
     include_nysquit : bool, optional
         Add the fs/2 as the last element of the vector. The default is True.

     Returns
     -------
     f  : np.ndarray
         Frequency vector.
         
    """
    
    check_fsample(fs)
    check_fsample(fmin, token='fmin')
    check_fsample(per_octave, token='per_octave')
    
    fn = fs/2
    N = np.floor(-per_octave/np.log(2)*np.log(fmin/fn))
    f = np.array([fn/2**(n/per_octave) for n in np.arange(N, -1, -1)])
    if not include_nysquit:
        f = f[:-1]
    return f

# %% Frequency Bands
def handle_freq_bands(f, fs):
    
    # check that frequency make sense
    check, e = check_freq_input(f, fs)
    if not check:
        raise e
        
    # check for the dimensions.
    f = np.asarray(f)
    
    if f.size == 1:
        if f.item() == 0 or f.item() == fs/2:
            raise ValueError('Select at least single value between 0, fs/2')
        # assume edge.
        f = np.array([0, f.item(), fs/2])
    
    if f.ndim == 1:
        f = zip(f[:-1], f[1:]) # wrap around a generator
        # [fa, fb, fc, fd] -> [(fa, fb), (fb, fc), (fc, fd)]
        
    # for every band.
    bands, labels = [], []

    for f1, f2 in f:
        # Arange bands.
        labels.append(str(f"{f1:.2f}-{f2:.2f}"))
        
        # centre-frequency edge.
        if f1 == 0:
            bands.append((f2, 'lowpass'))
            
        # Nyquist-frequency edge.
        elif f2 == fs/2:
            bands.append((f1, 'highpass'))
        
        # Band.
        else:
            bands.append(((f1, f2), 'bandpass'))
            
    return bands, labels

# %% Other.
# decibels.
def dB(x):
    x = np.abs(x)
    return 10*np.log10(x, where = x > 0)

# simple argmax phase.
def argmax_freq(f, x, axis=-1): # TODO broadcast properly
    map_ = np.vectorize(lambda idx: f[idx])
    return map_(np.argmax(x, axis=axis))

# downsample time vector
def downsample_time_vector(t, y, *args, axis=0):
    # TODO: control arguments.
    
    # down-sample t if necessary.
    if (n := y.shape[axis]) < (m := t.size):
        R = m//n
        t = t[::R][1:n] # account for additional delay

        # other arrays.
        args = [arg[::R][1:n] for arg in args]
    
        # cut-down on the original
        y = y[:len(t)]
        
    return (t, y, *args)

def split_evenly(x, R, axis=-1):
    
    n, r = x.shape[axis]//R, x.shape[axis]%R

    return np.split(x[:-r], n, axis=axis)



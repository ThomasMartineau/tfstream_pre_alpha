#!/usr/bin/env python
__author__ = "Thomas Martineau"
__copyright__ = "Copyright 2023, tf-stream"
__license__ = "FPA"
__version__ = "0.0.1"
__email__ = "tlc_martineau@gmail.com"
__status__ = "Pre-Alpha"
__doc__= "Fourier transform module."

# %% Imports
import numpy as np
from scipy.fft import fft
from scipy.signal.windows import get_window

from .epoch import buffer_epoch, view_epoch, get_epoch_length
from .utils import push_axis, check_freq_input, check_fsample

# %% STFT Function
# fourier handling fucntion.
def stft(
        x: np.ndarray, R:int, n:int, 
        zi: np.ndarray|None= None, 
        window: str|tuple|None=None, 
        offline: bool=True, axis: int=-1):
    """
    STFT function with both online and offiline toggle.

    Parameters
    ----------
    x : np.ndarray
        Signal to decompose.
    R : int
        Epoch stride length (or downsampling ratio).
    n : int
        Number of overlapping strides.
    zi : np.ndarray|None, optional
        Epoch buffer. The default is None.
    window : str|tuple|None, optional
        Window function name. The default is None.
    offline : bool, optional
        Toggle between using `view_epoch` and `buffer_epoch`. The default is True.
    axis : int, optional
        Axis along which to carry the decomposition. The default is -1.

    Returns
    -------
    X : np.ndarray
        STFT decomposition matrix of shape (L//2, ..., len(x)/R).

    """
    
    # apply the epoching.
    if offline:
        x = view_epoch(x, R, n, axis=axis)
    else:
        x, zi = buffer_epoch(x, R, n, zi, axis=axis)
    
    # find the dimension of the epoch.
    L = x.shape[axis]
    
    # apply windowing function.
    if window is not None:
        w = get_window(window, L)
        x = x*np.reshape(w, (L, *push_axis(x.ndim, axis)*(1,)))
 
    # extract fourier transform.
    X = fft(x, axis = axis)
    
    # only positive right side of the specrta.
    X = np.take(X, np.arange(0, L//2), axis)

    # move resulting axis to the front.
    X = np.moveaxis(X, axis, 0)

    # depending on state
    if offline:
        return X
    else:
        return X, zi
    
def get_stft_freq(R:int, n:int, fs:int|float):
    
    # check the sampling frequency.
    check_fsample(fs)
    
    # get the epoch lenght.
    L = get_epoch_length(R, n)

    return np.linspace(0, fs/2, L//2)

# %% IIR-Style Fourier Transform
# swift
def swift(
    f: list|np.ndarray, fs: int|float,
    tau: float = 14.4):
    """
    SWIFT filterbank constructor. Based on 
    L. L. Grado, M. D. Johnson and T. I. Netoff, 
    "The Sliding Windowed Infinite Fourier Transform [Tips & Tricks]," 
    in IEEE Signal Processing Magazine, vol. 34, no. 5, pp. 183-188, Sept. 2017, 
    doi: 10.1109/MSP.2017.2718039.

    Parameters
    ----------
    f : list|np.ndarray
        Frequency vector.
    fs : int|float
        Sampling frequency.
    tau : float, optional
        Exponential decay constant. The default is 14.4.

    Raises
    ------
    ValueError
        When tau is not stricly positive.

    Returns
    -------
    b : np.ndarray|int|pd.DataFrame
        Numerator filter coefficient bank. Shape is (n_filter, n_order)
    a : np.ndarray|int|pd.DataFrame
        Denominator filter coefficient bank. Shape is (n_filter, n_order)

    """
    
    # check the frequency.
    check_freq_input(f, fs, two_dim=False)
    
    # check tau
    if tau <= 0:
        raise ValueError('tau needs to be strictly positive')
    
    # normalise frequency by Nysquit
    w = 2j*np.pi*np.asarray(f)/fs
    N = len(f)
    
    # b - coefficients
    b = np.zeros((1, 2), dtype = complex)
    b[:, 0] = 1
    
    # a - coefficients
    a = np.zeros((N, 2), dtype = complex)  
    a[:, 0], a[:, 1] = 1, -np.exp(-1/tau)*np.exp(w)

    return b, a

# alpha-swift
def aswift(
    f, fs,
    tau_slow = 14.4,
    tau_fast = 2.89):
    """
    Alpha-SWIFT filterbank constructor. Based on 
    L. L. Grado, M. D. Johnson and T. I. Netoff, 
    "The Sliding Windowed Infinite Fourier Transform [Tips & Tricks]," 
    in IEEE Signal Processing Magazine, vol. 34, no. 5, pp. 183-188, Sept. 2017, 
    doi: 10.1109/MSP.2017.2718039.

    Parameters
    ----------
    f : list|np.ndarray
        Frequency vector.
    fs : int|float
        Sampling frequency.
    tau_slow : float, optional
        Exponential slow decay constant. The default is 14.4.
    tau_fast : float, optional
        Exponential fast decay constant. The default is 2.89.

    Raises
    ------
    ValueError
        When either tau is not stricly positive.

    Returns
    -------
    b : np.ndarray|int|pd.DataFrame
        Numerator filter coefficient bank. Shape is (n_filter, n_order)
    a : np.ndarray|int|pd.DataFrame
        Denominator filter coefficient bank. Shape is (n_filter, n_order)

    """
    
    # check the frequency.
    check_freq_input(f, fs, two_dim=False)
    
    # check tau
    if tau_slow <= 0:
        raise ValueError('tau_slow needs to be strictly positive')
        
    # check tau
    if tau_fast <= 0:
        raise ValueError('tau_fast needs to be strictly positive')

    # normalise frequency by Nysquit
    w = 2j*np.pi*np.asarray(f)/fs
    N = len(f)
    
    gamma = np.exp(-1/tau_slow)*np.exp(w)
    beta  = np.exp(-1/tau_fast)*np.exp(w)
    
    # b - coefficients
    b = np.zeros((N, 3), dtype = complex)
    b[:, 0] = 0
    b[:, 1] = beta - gamma
    b[:, 2] = 0
    
    # a - coefficients
    a = np.zeros((N, 3), dtype = complex)  
    a[:, 0] = 1
    a[:, 1] = -(beta + gamma)
    a[:, 2] = beta*gamma
    
    return b, a

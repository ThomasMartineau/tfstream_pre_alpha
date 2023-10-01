#!/usr/bin/env python
__author__ = "Thomas Martineau"
__copyright__ = "Copyright 2023, tf-stream"
__license__ = "FPA"
__version__ = "0.0.1"
__email__ = "tlc_martineau@gmail.com"
__status__ = "Pre-Alpha"
__doc__= "Signal generator."

# %% Imports
import numpy as np
from scipy.signal import chirp
from scipy.stats import norm 

# %% Chirp
mirror_array = lambda x: np.concatenate((x, x[::-1]))

def chirp_signal(
        T: int|float=10, 
        fs: int|float=200, 
        f0: int|float=10, f1: int|float=75, 
        method='linear', mirror=False, **kwargs):
    """
    Function wrapping around scipy.signal.chirp to generate a chirp signal and 
    its phase function.

    Parameters
    ----------
    T : int|float, optional
        Signal duration in s. The default is 10.
    fs : int|float, optional
        Sampling frequency in Hz. The default is 200.
    f0 : int|float, optional
        Initial frequency in Hz. The default is 10.
    f1 : int|float, optional
        Final frequency in Hz. The default is 75.
    method : TYPE, optional
        Phase function shape. The default is 'linear'.
    mirror : TYPE, optional
        Mirror and append the chirp signals. The default is False.
    **kwargs : TYPE
        Keyword arguments for scipy.signal.chirp .

    Returns
    -------
    t : np.ndarray
        Time vector.
    f : np.ndarray
        Frequency vector.
    x : np.ndarray
        Chirp signal.

    """
    
    t = np.arange(0, int(T*fs))/fs
    
    if method == 'linear':
        f = (f1 - f0)*t/T + f0
    
    elif method =='logarithmic':
        f = f0*(f1/f0)**(t/T)
    
    # call the chirp funciton.
    x = chirp(t, f0, T, f1, method=method, **kwargs)
    
    if mirror:
        t = np.concatenate((t, t[-1] + t))
        f = mirror_array(f)
        x = mirror_array(x)
    
    return t, f, x

# %% Even Gaussian.
def gaussian_am_modes(
        n: int=3, T: int|float=10, 
        fs: int|float=200, scale: float=0.25):
    """
    Generate amplitude modulated signals using a gaussian envelope.
    Modes are equally spaced in time and frequency.

    Parameters
    ----------
    n : int, optional
        Number of modes. The default is 3.
    T : int|float, optional
        Signal length in seconds. The default is 10.
    fs : int|float, optional
        Sampling frequency. The default is 200.
    scale : float, optional
        S. The default is 0.25.

    Returns
    -------
    x : 
        Gaussian am-modes. Shape is (int(T*fs), n).

    """
    
    # locations.
    locs = np.arange(0, n)/(n-1)*T
    freqs = np.arange(1, n+1)/(n+1)

    N = int(T*fs)
    edge = int(scale*T/n*fs*3)
    k = np.arange(-edge, N+edge)
    t = k/fs

    x, As = [], []
    for loc, f in zip(locs, freqs):
        A = norm.pdf(t, loc=loc, scale=scale*T/n)
        A /= np.max(A)
        x.append(A*np.sin(np.pi*k*f))
        As.append(A)

    return t, np.stack(x, axis=0), np.stack(As, axis=-1)


# %% Imports
import numpy as np
from scipy.signal import freqz

from .epoch import view_epoch, buffer_epoch, get_epoch_length
from .utils import check_freq_input

# %% CWT
def cwt(
    h: np.ndarray, x: np.ndarray, 
    R: int, n: int, 
    zi:  np.ndarray|None=None, 
    offline: bool=True, axis: int=-1):
    """
    Continuous wavelet transform

    Parameters
    ----------
    h : np.ndarray
        Wavelet kernels array, shape (len(f), R(n+1)).
    x : np.ndarray
        Signal to decompose.
    R : int
        Epoch stride length (or downsampling ratio).
    n : int
        Number of overlapping strides.
    zi : None|np.ndarray
        Buffer memory of size R.
    offline : bool, optional
        Toggle between using `view_epoch` and `buffer_epoch`. The default is True.
    axis : int, optional
        Axis along which to carry the decomposition. The default is -1.

    Raises
    ------
    ValueError
        If kernel is not properly formatted.

    Returns
    -------
    X : np.ndarray
        Wavelet decomposition matrix of shape (L//2, ..., len(x)/R).

    """
    
    
    # check that h can be properly build.
    try:
        h = np.asarray(h)
    except:
        raise ValueError('h cannot be properly assembled as an array')
        
    # check that the kernel is correct size.
    if h.shape[-1] != get_epoch_length(R, n):
        raise ValueError('h does not match epoch length.')
    
    # apply the epoching.
    if offline:
        x = view_epoch(x, R, n, axis=axis)
    else:
        x, zi = buffer_epoch(x, R, n, zi, axis=axis)
    
    # apply kernel to epoch (k/R, L) @ (L, len(f)).
    return np.matmul(x, h.T).T
    

def morlet(
        f: list|np.ndarray, fs: int|float, 
        R:int, n:int, cycles: list|int|np.ndarray=6):
    """
    Morelet wavelet filter bank constructor.

    Parameters
    ----------
    f : list|np.ndarray
        Frequency vector.
    fs : int|float
        Sampling frequency.
    x : np.ndarray
        Signal to decompose.
    R : int
        Epoch stride length (or downsampling ratio).
    n : int
        Number of overlapping strides.
    cycles : list|int|np.ndarray, optional
        Number of cycles to use for each wavelet. The default is 6.

    Returns
    -------
    h : np.ndarray
        Wavelet kernel filter bank.

    """
    # check the frequency vector.
    check_freq_input(f, fs, two_dim=False)
    
    # epoch length.
    L = get_epoch_length(R, n)
    
    # normalised frequency vector.
    w = 2*np.asarray(f)/fs
    
    # handle cycles.
    s = 2*cycles/(np.pi*w)
    
    # centre discre count.
    k = (np.arange(0, L) - L//2)
    
    # discrete frequency kernel.
    return np.exp(1j*np.pi*w[:, None]*k)*np.exp(-(k/s[:, None])**2)
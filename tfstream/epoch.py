#!/usr/bin/env python
__author__ = "Thomas Martineau"
__copyright__ = "Copyright 2023, tf-stream"
__license__ = "FPA"
__version__ = "0.0.1"
__email__ = "tlc_martineau@gmail.com"
__status__ = "Pre-Alpha"
__doc__= "Epoching module."

# %% Imports
import numpy as np

from .utils import handle_input

# %% Support Function.
def get_epoch_length(R, n):
    
    if not isinstance(R, int):
        raise TypeError('R needs to be a int.')
    
    if R <= 0:
        raise ValueError('R needs to be strictly positive.')
    
    if not isinstance(n, int):
        raise TypeError('n needs to be a int.')
    
    if n < 0:
        raise ValueError('n needs to be positive')
    
    return R*(n + 1)

def padding(x, L, R, axis):
    # populate the first stride.
    shape = list(x.shape)
    shape[axis] = L - R    
    return np.zeros(shape, dtype = x.dtype)

default_axis = -1

# %% Epoch View.
# more efficient version than epoch_unfold
def view_epoch(
        x: np.ndarray, R:int, n:int, 
        zero_pad: bool=True, axis: int=default_axis):
    """
    Parse an array into epochs.

    Parameters
    ----------
    x : np.ndarray
        Signal to parse.
    R : int
        Epoch stride length (or downsampling ratio).
    n : int
        Number of overlapping strides.
    zero_pad : bool, optional
        Add a first zero stride (of size R) on the left of the signal. The default is True.
    axis : int, optional
        Axis along which to epoch. The default is -1.

    Returns
    -------
    y : np.ndarray
        Epoched array.

    """
    # check the input.
    x = handle_input(x, axis)
    
    # construct length.
    L = get_epoch_length(R, n)
    
    # zero-pad the input
    if n > 0:
        zi = padding(x, L, R, axis)
        x = np.append(zi, x, axis=axis)
    
    # wrap around numpy function.
    view = np.lib.stride_tricks.sliding_window_view(x, L, axis=axis)
    
    return view[..., ::R, :]

# %% Epoch Buffer.
def buffer_epoch(x, R:int, n:int, 
                 zi: None|np.ndarray, axis: int=default_axis):
    """
    Parse an array into epochs by buffering signal blocks of size R.

    Parameters
    ----------
    x : np.ndarray
        Signal block of size R (along axis).
    R : int
        Epoch stride length (or downsampling ratio).
    n : int
        Number of overlapping strides.
    zi : None|np.ndarray
        Buffer memory of size R.
    axis : int, optional
        Axis along which to epoch. The default is -1.

    Raises
    ------
    ValueError
        When signal block size is not R.

    Returns
    -------
    y : np.ndarray
        Epoched array.
    zi : np.ndarray
        Buffer memory of size R.

    """
    # check the input.
    x = handle_input(x, axis)
    
    # construct length.
    L = get_epoch_length(R, n)
    
    # return directly without any epoching.
    if n == 0:
        return x, None
    
    # check that x as the correct lenght
    if x.shape[axis] != R:
        raise ValueError(f"x size on axis {axis} is expected to be {R}, got {x.shape[axis]}")
    
    if zi is None:
        zi = padding(x, L, R, axis)
    
    # complete the array.
    y = np.append(zi, x, axis=-1)
    
    # acute the edge.    
    return y, np.take(y, np.arange(L-R, L), -1)
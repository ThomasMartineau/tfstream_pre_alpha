#!/usr/bin/env python
__author__ = "Thomas Martineau"
__copyright__ = "Copyright 2023, tf-stream"
__license__ = "FPA"
__version__ = "0.0.1"
__email__ = "tlc_martineau@gmail.com"
__status__ = "Pre-Alpha"
__doc__= "Filter bank module, wrapping around scipy.signal"

# %% Imports
import numpy as np
import pandas as pd

# from signal
from scipy.signal import butter, cheby1, cheby2, ellip, firwin
from scipy.signal import lfilter, lfilter_zi

# other
from .utils import handle_freq_bands, handle_input

# %% Packing & Handling Support
def handle_packing(b, a):

    # check supported.
    if not isinstance(b, (pd.DataFrame, np.ndarray)):
        raise TypeError('Only DataFrame and ndarray supported')

    is_a_one = isinstance(a, int) and a == 1

    # check that both b, a are the same
    if (_type := type(b)) != type(a) and not is_a_one:
        raise TypeError(
            f"Type error b, a need to have the same type, expect {_type}")

    if is_a_one:
        n = b.shape[0]
    else:
        n = max(b.shape[0], a.shape[0])

    return _type, n


def ba_pseudo_broadcast(b, a, k):

    # get bk, ak vector depending on some exceptions rules.
    bk = b[0] if b.shape[0] == 1 else b[k]
    if isinstance(a, int) and a == 1:
        ak = 1
    else:
        ak = a[0] if a.shape[1] == 1 else a[k]

    return bk, ak


def handle_zi(n, zi):
    if zi is None:
        return

    if n != (m := len(zi)):
        raise ValueError('zi is not the correct size, expected {}, got {}.'.format(
            n, m))

    if not isinstance(zi, dict):
        raise TypeError('zi needs to be a dictionary.')

    for zii in zi.values():
        if not isinstance(zii, np.ndarray):
            raise TypeError('All values of zi need to be arrays.')

# %% Bank Iterator.
def lfilter_bank(
        b: np.ndarray|int|pd.DataFrame, a: np.ndarray|int|pd.DataFrame,
        x: np.ndarray,  zi: dict|None=None, axis: int=-1):
    """
    Filter bank function wrapping around scipy.signal.lfilter
    
    Parameters
    ----------
    b : np.ndarray|int|pd.DataFrame
        Numerator filter coefficient bank. Shape is (n_filter, n_order)
    a : np.ndarray|int|pd.DataFrame
        Denominator filter coefficient bank. Shape is (n_filter, n_order)
    x : np.ndarray
        Array to filter.
    axis : int, optional
        Dimension on which to filter. The default is -1.
    zi : dict|None, optional
        Filter memory to persist between function calls. The default is None.

    Returns
    -------
    y : np.ndarray
        Filter output.
    zi : dict
        Filter memory dictionary.
    """   
    
    # handle arguments.
    coeff_type, n = handle_packing(b, a)
    handle_zi(n, zi)
    x = handle_input(x, axis)

    # unpack from pandas.
    if coeff_type is pd.DataFrame:
        b, a = b.to_numpy(), a.to_numpy()

    # create the output (cast to coefficient value)
    y = np.empty((n, *x.shape), dtype=b.dtype)

    # iterate over the array
    for k in range(n):
        # pseudo-broacasting.
        bk, ak = ba_pseudo_broadcast(b, a, k)
        zk = None if zi is None else zi[k]

        # filter function.
        output = lfilter(bk, ak, x, axis=axis, zi=zk)

        # manage hidden state.
        if zi is None:
            y[k] = output
        else:
            y[k], zi[k] = output

    if zi is None:
        return y
    else:
        return y, zi

def lfilter_zi_bank(
        b: np.ndarray|int|pd.DataFrame, a: np.ndarray|int|pd.DataFrame):
    """
    Initialize filter bank wrapping around scipy.signal.lfilter_zi_bank.

    Parameters
    ----------
    b : np.ndarray|int|pd.DataFrame
        Numerator filter coefficient bank. Shape is (n_filter, n_order)
    a : np.ndarray|int|pd.DataFrame
        Denominator filter coefficient bank. Shape is (n_filter, n_order)

    Returns
    -------
    zi : dict
        Filter memory dictionary.

    """

    # get the zi bank.
    coeff_type, n = handle_packing(b, a)

    # unpack from pandas.
    if coeff_type is pd.DataFrame:
        b, a = b.to_numpy(), a.to_numpy()

    zi = {}  # to account for multiple sized arrays.
    for k in range(n):
        zi[k] = lfilter_zi(b[k], a[k])

    return zi

# %% Standard Bank Iterator
iir = [butter, ellip, cheby1, cheby2]

fir = [firwin]

def handle_filter(backend):
    # check backend
    if backend in iir:
        return True
    
    elif backend in fir:
        return False
    
    else:
        raise ValueError('Filterbackend function not supported')
    
def get_filter_bank(
        N: int, f: list|np.ndarray, fs: int|float, *args,
        _filter=butter,
        pack_in_df: bool=True,
        **kwargs):
    """
    Build a filter bank using a scipy.signal backend function and band definition.

    Parameters
    ----------
    N : int
        Filter order.
    f : list|np.ndarray
        Band definition. A vector defines the frequencies at which the spectrum is partitioned.
        A list of frequency pairs defines specific bands to build.
    fs : int|float
        Sampling frequency.
    _filter : function, optional
        A scipy.signal backend. Either butter, ellip, cheby1, cheby2 or firwin. The default is butter.
    pack_in_df : bool, optional
        Pack into a DataFrame. The default is True.
    **kwargs : dict
        Keyword arguments for _filter.

    Returns
    -------
    b : np.ndarray|int|pd.DataFrame
        Numerator filter coefficient bank. Shape is (n_filter, n_order)
    a : np.ndarray|int|pd.DataFrame
        Denominator filter coefficient bank. Shape is (n_filter, n_order)

    """
    
    # (a) handle bands.
    bands, labels = handle_freq_bands(f, fs)

    # (b) iir.
    if handle_filter(_filter):
        # to collect
        Bs, As = [], []

        # bands
        for band, btype in bands:

            bi, ai = _filter(
                N, band,
                btype=btype, fs=fs, output='ba')

            Bs.append(bi)
            As.append(ai)

        # find min-max & pad.
        for X in [Bs, As]:
            n = max(map(len, X))
            for key, x in enumerate(X):
                if (m := len(x)) == n:
                    continue
                X[key] = np.append(x, (n - m)*[0,])

        # convert to array.
        b, a = np.stack(Bs, axis=0), np.stack(As, axis=0)

    # fir
    else:
        # look at the kwargs.
        if 'pass_zero' in kwargs:
            del kwargs['pass_zero']

        Bs = []
        for band, btype in bands:
            Bs.append(
                firwin(
                    N, band,
                    pass_zero=(btype == 'lowpass'),
                    fs=fs, **kwargs))

        b, a = np.stack(list(Bs.values())), 1

    if pack_in_df:
        b = pd.DataFrame(b, index=labels)
        a = pd.DataFrame(a, index=labels)

    return b, a




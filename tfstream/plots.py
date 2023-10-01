#!/usr/bin/env python
__author__ = "Thomas Martineau"
__copyright__ = "Copyright 2023, tf-stream"
__license__ = "FPA"
__version__ = "0.0.1"
__email__ = "tlc_martineau@gmail.com"
__status__ = "Pre-Alpha"
__doc__= "Useful plot functions for diagnostic and testing."

# %% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import freqz

from .bank import handle_packing 
from .utils import dB, argmax_freq, downsample_time_vector
from .generator import chirp_signal, gaussian_am_modes

# %% Settings.
figsize=(12,4)
plt.rc('lines', linewidth=2.5)
plt.rc('font', size=12.5)

# %% Typical Plots
def spectrogram(t, f, y, ax = None):
    
    if ax is None:
        plt.figure()
        ax = plt.gca()
    X, Y = np.meshgrid(t[:, None], f)
    im = ax.pcolormesh(X, Y, y.T, shading='gouraud')
    
    return im
       
# %% Frequency Response
def plot_freqz_bank(
        b: np.ndarray|int|pd.DataFrame, a: np.ndarray|int|pd.DataFrame,
        ylim: tuple=(-60,5),
        log_freq: bool=False,
        plot_phase: bool=False,
        fs: int|float=0.5,
        ax=None,
        **kwargs):
    """
    Plot frequency response of a filter bank using scipy.signal.freqz.

    Parameters
    ----------
    b : np.ndarray|int|pd.DataFrame
        Numerator filter coefficient bank. Shape is (n_filter, n_order)
    a : np.ndarray|int|pd.DataFrame
        Denominator filter coefficient bank. Shape is (n_filter, n_order)
    ylim : tuple, optional
        Plotting limit of filter response in dB. The default is (-60,5).
    log_freq : bool, optional
        Use log-scaling on frequency axis. The default is False.
    plot_phase : bool, optional
        Add phase reponse. The default is False.
    fs : int|float, optional
        Sampling frequency. The default is 0.5.
    ax : optional
        Axis of a matplotlib figure. The default is None.
    **kwargs : 
        Key word arguments for scipy.signal.freqz.

    Returns
    -------
    fig : 
        Matplotlib figure.

    """
            
    # handle the packing.
    coeff_type, n = handle_packing(b, a)

    # unpack from pandas.
    if coeff_type is pd.DataFrame:
        labels = b.index
        b, a = b.to_numpy(), a.to_numpy()
    else:
        labels = None

    if plot_phase:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True)
    else:
        # only supported for abs plot.
        if ax is None:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax1 = ax.get_figure(), ax

    # iterate over the array.
    for k in range(n):
        # pseudo broacast.
        bk = b[0] if b.shape[0] == 1 else b[k]
        if isinstance(a, int) and a == 1:
            ak = 1
        else:
            ak = a[0] if a.shape[1] == 1 else a[k]

        # frequency response.
        w, h = freqz(bk, ak, fs=fs, **kwargs)

        # magnitude.
        H = dB(h)
        ax1.plot(w, H, linewidth=2.5)
        ax1.set_xlim(w[0], w[-1])
        ax1.set_ylim(*ylim)
        if log_freq:
            ax1.set_xscale('log')
        ax1.grid()
        ax1.set_ylabel(
            'Frequency Response (dB)')
        if not plot_phase:
            ax1.set_xlabel(
                'Frequency (Hz)')

        # phase.
        if plot_phase:
            phi = np.unwrap(np.angle(h))
            ax2.plot(w, phi)
            if log_freq:
                ax2.set_xscale('log')
            ax2.grid()
            ax2.set_xlabel('Frequency (Hz)')

    # add labels.
    if labels is not None:
        ax1.legend(labels, title='Filters')

    plt.tight_layout()

    return fig

# %% Specific Generator test.
def plot_am_test(
        func, 
        fs: int|float=200,
        labels: list=None,
        ax=None,
        add_legend: bool=True,
        xlabel: None|str='Time (s)',
        ylabel: None|str='Filter Response',
        **kwargs):
    """
    Amplitude modulated signal test using Gaussian envelopes.

    Parameters
    ----------
    func : function
        Filtering function to test. Inputs x returns y.
    fs : int|float, optional
        Sampling frequency. The default is 200.
    labels : list, optional
        List of labels for the month. The default is None.
    ax : matplotlib axis, optional
        Optional axis from a matplotlib figure. The default is None.
    add_legend : bool, optional
        Display legend. The default is True.
    xlabel : None|str, optional
        Label on x-axis. The default is 'Time (s)'.
    ylabel : None|str, optional
        Label on y-axis. The default is 'Filter Response'.
    **kwargs : 
        Keyword arguments for tfstream.generator.am_gaussian_modes.

    Returns
    -------
    fig : 
        Matplotlib figure.

    """

    t_, x, a = gaussian_am_modes(fs=fs, **kwargs)
    y = func(x.sum(axis=0)).T  # transpose, time before.

    # down-sample t if necessary.
    t, y = downsample_time_vector(t_, y)

    # figure.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # the spectra.
    ax.plot(t_, x.sum(axis=0), 'k', alpha=0.25)
    lines = ax.plot(t, y, '-o', label=labels)
    ax.set_ylim(-1.2, 1.2)
    ax.grid()

    # plot the frequency trajectory
    for ai, line in zip(a.T, lines):
        ax.plot(
            t_, ai, '--',
            color=line.get_color())

    if add_legend:
        ax.legend(title='Response')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # tight layout.
    plt.tight_layout()

    return fig

def plot_chirp_test(
        func, 
        f: list|np.ndarray=None,
        fs=200,
        ax=None,
        add_colorbar: bool=True,
        add_legend: bool=True,
        xlabel: str='Time (s)',
        ylabel: str='Frequency (Hz)',
        **kwargs):
    """
    Frequency modulated test using Gaussian chirp phase function.

    Parameters
    ----------
    func : function
        Filtering function to test. Inputs x returns y of shape (len(x)/R, len(f)).
    f : list|np.ndarray, optional
        Frequency corresponding to the first dimension of y. The default is None.
    fs : int|float, optional
         Sampling frequency. The default is 200.
    ax : matplotlib axis, optional
        Optional axis from a matplotlib figure. The default is None.
    add_colorbar : bool, optional
        Add a colorbar. The default is True.
    add_legend : bool, optional
        Add the legends. The default is True.
    xlabel : str, optional
        Label on the x-axis. The default is 'Time (s)'.
    ylabel : str, optional
        Label on the y-axis. The default is 'Frequency (Hz)'.
    **kwargs : 
        Keyword arguments for tfstream.generator.am_gaussian_modes.

    Returns
    -------
    fig : 
        Matplotlib figure.

    """
    
    # produce the chirp signal
    t_, c, x = chirp_signal(fs=fs, **kwargs)

    # run the transform.
    y = func(x).T

    # down-sample t if necessary.
    t, y = downsample_time_vector(t_, y)

    # estimate using the argmax function.
    if f is None:
        f = np.arange(0, t.shape[0])

    c_est = argmax_freq(f, y)

    # figure.
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    im = spectrogram(t, f[None, :], y, ax)
    if add_colorbar:
        fig.colorbar(im)

    # plot the frequency trajectory
    ax.plot(t, c_est, 'r-o', label='Estimation')  # add chirp estimate
    ax.plot(t_, c, 'k', label='Trajectory')

    ax.set_ylim(0, min(fs/2, max(f)))
    ax.set_xlim(t[0], t[-1])

    if add_legend:
        ax.legend()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return fig


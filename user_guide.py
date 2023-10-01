#!/usr/bin/env python
__author__ = "Thomas Martineau"
__copyright__ = "Copyright 2023, tf-stream"
__license__ = "FPA"
__version__ = "0.0.1"
__email__ = "tlc_martineau@gmail.com"
__status__ = "Pre-Alpha"
__doc__= "Pre-Alpha user guide."

# %% Imports
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import butter
from pathlib import Path

from tfstream.bank import get_filter_bank, lfilter_bank
from tfstream.epoch import view_epoch
from tfstream.plots import plot_freqz_bank, plot_am_test
from tfstream import utils

# %% Gaussian Pulse.
# (1) Build & Inspect Filter Bank.
fs = 200
bands = [0, 35, 65, 100]
b, a = get_filter_bank(6, bands, _filter=butter, fs=fs)

fig = plot_freqz_bank(b, a, fs=fs)
plt.savefig(Path('.')/'examples/figures/butter_bank.svg', format='svg')

# define epoching settings.
R, n = 8, 1

# (2) Build pipeline and run example.
def _filter(x):
    # break signal the x-signal into 3-components.
    y = lfilter_bank(b, a, x) # y.shape = (3, x.shape[0])
    
    # downsample R into epoch window.
    y = view_epoch(y, R, n) # y.shape = (3, x.shape[0]/R, R(n+1))
    
    # root-mean square estimation of A(t), with correction sqrt(2) scaling.
    # https://en.wikipedia.org/wiki/Root_mean_square
    return np.sqrt(2)*y.std(axis=-1)  # y.shape = (3, x.shape[0]/R)

fig = plot_am_test(
    _filter, # pass on the filtering pipeline.
    labels=b.index,
    fs=fs, T=1, scale=0.5) # signal generator parameters.

plt.savefig(Path('.')/'examples/figures/am_test.svg', format='svg')

# %% Streaming.
from tfstream.bank import lfilter_zi_bank
from tfstream.epoch import buffer_epoch
from tfstream.generator import gaussian_am_modes

# epoch parameters.
R, n = 8, 1

def _filter(x, z):
    # check the chunk is the correct size.
    assert x.shape[-1] == R
    
    # break signal the x-signal into 3-components.
    y, z[0] = lfilter_bank(b, a, x, zi=z[0]) # 
     
    # downsample R into epoch window.
    if n > 0:
        y, z[1] = buffer_epoch(y, R, n, z[1]) # y.shape = (x.shape[0]/R,3,R(n+1))
        
    return np.sqrt(2)*y.std(axis=-1), z

# test signal.
_, x, _ = gaussian_am_modes(fs=fs, T=1, scale=0.5)
y = []
z = [lfilter_zi_bank(b, a), None]

for chunk in utils.split_evenly(x.sum(axis=0), R):
    yi, z = _filter(chunk, z)
    y.append(yi)

y = np.asarray(y)

# %% STFT.
from tfstream.fourier import stft, get_stft_freq
from tfstream.plots import plot_chirp_test
from tfstream import utils

fs = 200
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12,4))

for (R, n), ax, legend, cmb, tag in zip(
        [(16, 0), (25, 4), (50, 8)], axes.flatten(), 
        [True, False, False], [False, False, True],
        ['a', 'b', 'c']):
    
    plot_chirp_test(
        lambda x: utils.dB(stft(x, R, n, window = ('kaiser', 1))),
        f=get_stft_freq(R, n, fs),
        ax=ax,
        add_colorbar=cmb,
        add_legend=legend,
        ylabel='Frequency (Hz)' if legend else None,
        fs=fs, f0=10, f1=75)
    
    ax.set_title('({}) Window Length {} (ms)'.format(tag, R*(n+1)/fs*1000)) 

plt.tight_layout()
plt.savefig(Path('.')/'examples/figures/stft_test.png', format='png')

# %% SWIFT
from tfstream.fourier import swift, aswift
from tfstream.plots import plot_chirp_test

fs = 200
f = utils.freq_linear_scale(fs, 5)

# swift.
b, a = swift(f, fs)
fig = plot_freqz_bank(b, a, ylim=(0, 15), fs=fs)
plt.savefig(Path('.')/'examples/figures/swift_bank.svg', format='svg')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4), sharey=True)

plot_chirp_test(
    lambda x: utils.dB(lfilter_bank(b, a, x)),
    f = f,
    fs = fs,
    add_colorbar=False,
    ax = ax1)

ax1.set_title('SWIFT')

# alpha-swift.
b, a = aswift(f, fs)

plot_chirp_test(
    lambda x: utils.dB(lfilter_bank(b, a, x)),
    f = f,
    fs = fs,
    add_colorbar=True,
    add_legend=False,
    ylabel=None, 
    ax = ax2)

ax2.set_title(r'$\alpha$-SWIFT')

plt.tight_layout()
plt.savefig(Path('.')/'examples/figures/swift_test.png', format='png')

# %% Morlet's Wavelet.
from matplotlib.gridspec import GridSpec

from tfstream.wavelets import cwt, morlet
from tfstream.epoch import get_epoch_length
from tfstream.plots import plot_chirp_test, plot_freqz_bank
from tfstream import utils

# (a) Build a simple bank.
fs, R, n = 200, 50, 3
f = utils.freq_dyadic_scale(fs, per_octave=1, fmin=5,  include_nysquit=False)
h = morlet(f, fs, R, n, cycles=8)

# (b) layout for plot.
fig = plt.figure(figsize=(12,4))
gs = GridSpec(1, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
L = get_epoch_length(R, n)
t = (np.arange(0, L) - L//2)/fs
ax1.plot(t, h.T)
ax1.grid()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Kernel Real-Part')

ax2 = fig.add_subplot(gs[0, 1:])
plot_freqz_bank(
    h, 1, 
    ylim = (-150, 30),
    log_freq=True,
    fs=fs,
    ax=ax2)
ax2.legend(f, title='Frequency (Hz)')
ax2.grid()

plt.tight_layout()
plt.savefig(Path('.')/'examples/figures/morlet_bank.svg', format='svg')

fig, axes = plt.subplots(1, 3, figsize=(12,4),  sharey=True)
f = utils.freq_dyadic_scale(fs, per_octave=6, fmin=1,  include_nysquit=False)

for cycles, ax, legend, cmb, tag in zip(
        [6, 12, 24], axes,
        [True, False, False], [False, False, True],
        ['a', 'b', 'c']):
    
    h = morlet(f, fs, R, n, cycles=cycles)
    
    plot_chirp_test(
        lambda x: utils.dB(cwt(h, x, R, n)),
        f=f,
        fs=fs,
        method='logarithmic',
        ax=ax,
        add_colorbar=cmb,
        add_legend=legend,
        ylabel='Frequency (Hz)' if legend else None,
        )
    
    ax.set_title('({}) n={}'.format(tag, cycles))
    
plt.tight_layout()
plt.savefig(Path('.')/'examples/figures/morlet_test.png', format='png')


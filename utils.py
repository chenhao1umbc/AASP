##
# """This file constains all the necessary classes and functions"""
import os
import pickle
import time
import datetime
import wave
import bisect

import torch
import scipy.signal as sg
import numpy as np
import matplotlib.pyplot as plt


tt = datetime.datetime.now
# torch.set_default_dtype(torch.double)
np.set_printoptions(linewidth=120)
torch.set_printoptions(linewidth=120)
torch.backends.cudnn.deterministic = True
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def readwav(file):
    """
    Read a wav file.
    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.
    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    return rate, sampwidth, array


def label_str2num(l1, l2, l3, labels):
    """
    This function will turn the string labels in to vector
    :param l1: script01_sid
    :param l2: script02_sid
    :param l3: script03_sid
    :param labels: the labels with string
    :return: matrix of 0 and 1
    """
    pool1 = [i[2] for i in l1]
    pool2 = [i[2] for i in l2]
    pool3 = [i[2] for i in l3]
    pool1.extend(pool2)
    pool1.extend(pool3)
    pool = list(set(pool1))
    pool.sort()

    N = len(labels)  # number of examples
    Y = np.zeros((N, len(pool)))  # initialization
    for i in range(N):
        for ii in labels[i]:  # each ii is a string
            Y[i, pool.index(ii)] = 1
    return Y


def spectro(x, bw=256, overlap=0.1, fs=44.1e3, showplot=True):
    f, t, sx = sg.spectrogram(x, fs=fs, nperseg=bw, noverlap=int(bw*overlap))
    if showplot:
        plt.figure()
        plt.title('Spectrogram')
        plt.pcolormesh(t, f, sx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return sx


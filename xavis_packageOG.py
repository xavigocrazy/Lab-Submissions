import numpy as np

#load arr_0
def load_npz_arr0(filename):
    """Load .npz file and return arr_0."""
    return np.load(filename)["arr_0"]

#time axis
def time_axis(N, fs):
    """Return time axis in seconds."""
    return np.arange(N) / fs

#FFT
def fft_spectrum(x, fs, one_sided=True):
    """
    Compute FFT frequency axis and power spectrum.
    
    Returns:
        f : frequency axis (Hz)
        P : power spectrum |FFT|^2
    """
    x = np.asarray(x)
    N = x.size

    X = np.fft.fft(x)
    P = np.abs(X)**2
    f = np.fft.fftfreq(N, d=1/fs)

    if one_sided:
        mask = f >= 0
        return f[mask], P[mask]

    return f, P


def alias_frequency(f0, fs):
    """
    Predict aliased frequency for a tone f0 sampled at fs.
    """
    fa = abs(f0 - round(f0 / fs) * fs)
    return fa if fa <= fs/2 else fs - fa


#proves 
def dominant_frequency(f, P, fmin=0):
    """
    Return frequency of dominant spectral peak above fmin.
    """
    mask = f >= fmin
    return f[mask][np.argmax(P[mask])]

def rms(x):
    """Root-mean-square value."""
    return np.sqrt(np.mean(np.square(x)))

def snr_db(signal, noise):
    """Signal-to-noise ratio in dB."""
    return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))


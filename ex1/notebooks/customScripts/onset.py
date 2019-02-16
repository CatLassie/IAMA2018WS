import librosa
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import medfilt



######## PART 2 ########

# NOT USED
def normalize_frequencies(spectrogram):
    normalized_spectrogram = librosa.util.normalize(S=spectrogram, axis=1) #axis=0 is columns, but maybe norm=1

    #print(len(normalized_spectrogram), len(normalized_spectrogram[0]))
    #print('spectrogram first frame:\n')
    #for i, f in enumerate(normalized_spectrogram):
    #    print(i, f[0])
    #print('spectrogram first element:\n\n',sorted(normalized_spectrogram[0]), '\n\n\n')

    return normalized_spectrogram


def compute_odf(spectrogram):
    spectral_flux = []
    for j, f in enumerate(spectrogram[0]):
        flux = 0
        for i, g in enumerate(spectrogram):
            if j > 0:
                diff = spectrogram[i][j] - spectrogram[i][j-1]
                # only keep positive difference (increase in energy)
                if diff > 0:
                    flux = flux + (diff**2)

        spectral_flux.append(flux)

    #print(spectral_flux)
    return spectral_flux

def apply_threshold_mean(odf, window_size, inf=False):
    odf = np.array(odf)
    mean = np.mean(odf)
    #minimum_allowed = max(odf)*threshold (cut off all values below this threshold)

    if inf:
        print("max value: ", max(odf))
        print("mean value: ", np.mean(odf))
        print("median value: ", np.median(odf))

    #compute running mean with windows
    running_mean = np.convolve(odf, np.ones((window_size,))/window_size, mode='same')
    threshold = 1.05*np.where(running_mean > mean, running_mean, mean)
    #threshold = running_mean + 0.1*np.std(odf)
    
    #running_median = medfilt(odf, window_size)
    #threshold = running_median + 0.33*np.std(odf)

    if inf:
        print("odf: ", odf)
        print("threshold: ", threshold)

    peaks = np.where(odf > threshold, odf, 0)
    return peaks

def apply_threshold_mean_std(odf, window_size, inf=False):
    odf = np.array(odf)
    mean = np.mean(odf)
    #minimum_allowed = max(odf)*threshold (cut off all values below this threshold)

    if inf:
        print("max value: ", max(odf))
        print("mean value: ", np.mean(odf))
        print("median value: ", np.median(odf))

    #compute running mean with windows
    running_mean = np.convolve(odf, np.ones((window_size,))/window_size, mode='same')
    #threshold = 1.05*np.where(running_mean > mean, running_mean, mean)
    threshold = running_mean + 0.1*np.std(odf)
    
    #running_median = medfilt(odf, window_size)
    #threshold = running_median + 0.33*np.std(odf)

    if inf:
        print("odf: ", odf)
        print("threshold: ", threshold)

    peaks = np.where(odf > threshold, odf, 0)
    return peaks

def apply_threshold_median_std(odf, window_size, inf=False):
    odf = np.array(odf)
    mean = np.mean(odf)
    #minimum_allowed = max(odf)*threshold (cut off all values below this threshold)

    if inf:
        print("max value: ", max(odf))
        print("mean value: ", np.mean(odf))
        print("median value: ", np.median(odf))

    #compute running mean with windows
    #running_mean = np.convolve(odf, np.ones((window_size,))/window_size, mode='same')
    #threshold = 1.05*np.where(running_mean > mean, running_mean, mean)
    #threshold = running_mean + 0.1*np.std(odf)
    
    running_median = medfilt(odf, window_size)
    threshold = running_median + 0.33*np.std(odf)

    if inf:
        print("odf: ", odf)
        print("threshold: ", threshold)

    peaks = np.where(odf > threshold, odf, 0)
    return peaks

def pick_local_peaks(peaks):
    r = 3
    peaks = np.asarray(peaks)
    # pad peaks with 0s
    for i in range(r):
        peaks = np.insert(peaks, 0, 0)
        peaks = np.append(peaks, 0)

    # get the local maximas (+ magic to offset padded 0s)
    maxima = argrelextrema(peaks, np.greater, order=r)[0] - r
    return maxima

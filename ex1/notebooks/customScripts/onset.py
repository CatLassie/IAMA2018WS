import librosa
import numpy as np
from scipy.signal import argrelextrema


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

def apply_threshold(odf, threshold, inf=False):
    odf = np.array(odf)
    minimum_allowed = max(odf)*threshold
    if inf:
        print("lower threshold value", minimum_allowed)
    peaks = np.where(odf > minimum_allowed, odf, 0)
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

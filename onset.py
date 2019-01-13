import librosa
import numpy as np


######## PART 2 ########


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


def apply_threshold(odf, threshold=0):
    odf = np.array(odf)
    peaks = np.where(odf > threshold, odf, 0)
    return peaks

def pick_local_peaks(peaks):
    maxima = []
    for i, p in enumerate(peaks):
        if p > 0:
            for r in range(1,4):
                if i + r < len(peaks):





        maxima.append(p)

    return maxima

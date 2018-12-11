import librosa
import numpy as np


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=96000)
    print("sample values: ", y)
    print("sample #: ", len(y))
    print("sampling rate: ", sr)
    print()
    return y


# split input vector into overlapping frames (unused)
def compute_frames(y):
    frames_matrix = librosa.util.frame(y, frame_length=2048, hop_length=1024)
    print('frame size:', len(frames_matrix))
    print('frame #', len(frames_matrix[0]))
    print()
    """for i, f in enumerate(framesMatrix):
        print(i, f[0])"""
    return frames_matrix


# apply short time fourier transform
# take absolute values and square them
# returns power spectrogram
def compute_spectrogram(y):
    spectrogram = librosa.core.stft(y=y, n_fft=2048, hop_length=1024)
    power_spectrogram = (np.abs(spectrogram))**2
    """"
    print(spectrogram[0])
    print()
    """
    #print('spectrogram first element:\n\n', spectrogram_real[0], '\n\n\n')
    return power_spectrogram


# apply mel scale to spectrogram
def mel_transform(spectrogram):
    mel_spectrogram = librosa.feature.melspectrogram(S=spectrogram)
    #print('mel spectrogram first element:\n\n', mel_spectrogram[0], '\n\n\n')
    return mel_spectrogram


# apply a log10 scale to spectrogram (resulting magnitudes are in decibels)
def log_scale(spectrogram):
    log_mel_spectrogram = librosa.power_to_db(S=spectrogram)
    #print('log mel spectrogram first element:\n\n', log_mel_spectrogram[0], '\n\n\n')
    return log_mel_spectrogram


def main():
    file_path = 'music_data/shortName.flac'
    y = load_audio(file_path)
    spectrogram = compute_spectrogram(y)
    mel_spectrogram = mel_transform(spectrogram)
    log_mel_spectrogram = log_scale(mel_spectrogram)


if __name__ == "__main__":
    main()

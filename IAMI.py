import librosa
import numpy as np


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=96000)
    print("sample values: ", y)
    print("sample #: ", len(y))
    print("sampling rate: ", sr)
    print()
    return y


def compute_frames(y):
    frames_matrix = librosa.util.frame(y, frame_length=2048, hop_length=1024)
    print('frame size:', len(frames_matrix))
    print('frame #', len(frames_matrix[0]))
    print()
    """for i, f in enumerate(framesMatrix):
        print(i, f[0])"""
    return frames_matrix


def compute_spectrogram(y):
    spectrogram = librosa.core.stft(y=y, n_fft=2048, hop_length=1024)
    spectrogram_real = np.abs(spectrogram) # **2 for power spectrogram?
    """"
    print(spectrogram[0])
    print()
    """
    #print('spectrogram first element ', spectrogram_real[0])
    return spectrogram_real


def mel_transform(spectrogram):
    mel_spectrogram = librosa.feature.melspectrogram(S=spectrogram)
    #print('spectrogram first element ', mel_spectrogram[0])
    return mel_spectrogram


def log_scale():
    return


def main():
    file_path = 'music_data/shortName.flac'
    y = load_audio(file_path)
    spectrogram = compute_spectrogram(y)
    mel_spectrogram = mel_transform(spectrogram)


if __name__ == "__main__":
    main()

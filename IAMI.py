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
    spectrogramReal = np.abs(spectrogram)
    """"
    print(spectrogram[0])
    print()
    print(spectrogramReal[0])
    """
    return spectrogramReal


def main():
    file_path = 'music_data/shortName.flac'
    y = load_audio(file_path)
    spectrogram = compute_spectrogram(y)


if __name__ == "__main__":
    main()

import librosa
import numpy as np
import feature as f


def main():
    file_path = 'music_data/shortName.flac'
    y = f.load_audio(file_path)
    log_mel_spectrogram = f.compute_spectrogram(y, 2048, 1024, 40, True, True, True)


if __name__ == "__main__":
    main()
